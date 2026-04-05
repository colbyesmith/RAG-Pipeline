from __future__ import annotations

import asyncio
import logging
import os
import re
import statistics
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.bm25 import BM25Index
from app.chunk_store import store
from app.config import Settings, get_settings
from app.cross_encoder_rerank import rerank_with_cross_encoder
from app.generation import generate_answer
from app.hybrid_search import hybrid_search
from app.vector_ann import configure_vector_ann_from_settings
from app.intent_query import classify_and_rewrite
from app.pdf_ingest import TextChunk, chunk_pages, extract_pages_pdf
from app.policies import evaluate_query_policies
from app.multi_hop import merge_two_hop_candidates, propose_followup_query
from app.mistral_client import MistralError, mistral_embed
from app.attribution import summarize_scores_by_document
from app.schemas import Citation, IngestResponse, IngestTimings, QueryRequest, QueryResponse

_BASE = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
_STATIC = _BASE / "static"


def _safe_name(name: str) -> str:
    base = os.path.basename(name)
    if not re.fullmatch(r"[\w.\- ]+", base) or ".." in base:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not base.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    return base


async def _embed_in_batches(
    client: httpx.AsyncClient,
    settings: Settings,
    texts: list[str],
) -> list[list[float]]:
    out: list[list[float]] = []
    batch_size = settings.mistral_embed_batch_size
    delay_s = settings.mistral_embed_batch_delay_ms / 1000.0
    for i in range(0, len(texts), batch_size):
        if i > 0 and delay_s > 0:
            await asyncio.sleep(delay_s)
        batch = texts[i : i + batch_size]
        vecs = await mistral_embed(client, settings, batch)
        out.extend(vecs)
    return out


def _pdf_to_chunks_sync(tmp_path: str, name: str, settings: Settings) -> list[TextChunk]:
    pages = extract_pages_pdf(
        tmp_path,
        name,
        pdf_fast=settings.pdf_extract_fast,
        pdf_fitz_min_chars_skip_pypdf=settings.pdf_fitz_min_chars_skip_pypdf,
    )
    return chunk_pages(
        pages,
        name,
        settings.chunk_size_chars,
        settings.chunk_overlap_chars,
    )


def _rebuild_bm25() -> BM25Index:
    idx = BM25Index()
    idx.build(store.all_with_embeddings())
    return idx


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_vector_ann_from_settings(get_settings())
    app.state.http = httpx.AsyncClient()
    yield
    await app.state.http.aclose()


app = FastAPI(title="PDF RAG (custom retrieval)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    index = _STATIC / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Place static/index.html or open /docs"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    settings: Settings = Depends(get_settings),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    max_bytes = settings.upload_max_mb * 1024 * 1024
    client: httpx.AsyncClient = app.state.http

    jobs: list[tuple[str, str]] = []
    for uf in files:
        raw_name = uf.filename or "document.pdf"
        name = _safe_name(raw_name)
        content = await uf.read()
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File {name} exceeds upload limit")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            jobs.append((tmp.name, name))

    all_chunks: list[TextChunk] = []
    names = [n for _, n in jobs]
    t_extract0 = time.perf_counter()
    try:
        chunk_lists = await asyncio.gather(
            *[asyncio.to_thread(_pdf_to_chunks_sync, tp, n, settings) for tp, n in jobs]
        )
        for cl in chunk_lists:
            all_chunks.extend(cl)
    finally:
        for tmp_path, _ in jobs:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    extract_s = time.perf_counter() - t_extract0

    if not all_chunks:
        logger.info(
            "ingest timings: pdf_extract+chunk=%.2fs embedding=0.00s chunks=0 files=%d",
            extract_s,
            len(names),
        )
        return IngestResponse(
            ingested_files=names,
            chunks_added=0,
            message="No extractable text found. Scanned PDFs need an OCR tool upstream; "
            "for text PDFs, try re-exporting from the source app.",
            timings=IngestTimings(pdf_extract_and_chunk_s=extract_s, embedding_s=0.0),
        )

    texts = [c.text for c in all_chunks]
    t_embed0 = time.perf_counter()
    try:
        embeddings = await _embed_in_batches(client, settings, texts)
    except MistralError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    embed_s = time.perf_counter() - t_embed0

    added = store.add_chunks(all_chunks, embeddings)
    logger.info(
        "ingest timings: pdf_extract+chunk=%.2fs embedding=%.2fs chunks=%d files=%d",
        extract_s,
        embed_s,
        added,
        len(names),
    )
    return IngestResponse(
        ingested_files=names,
        chunks_added=added,
        message=f"Indexed {added} chunks from {len(names)} file(s).",
        timings=IngestTimings(pdf_extract_and_chunk_s=extract_s, embedding_s=embed_s),
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
):
    q = body.query.strip()

    policy = evaluate_query_policies(q)
    if policy.blocked:
        return QueryResponse(
            answer=policy.message or "Request declined.",
            needs_retrieval=False,
            intent="policy",
            retrieval_skipped_reason="policy_block",
            policy_flags=policy.flags,
        )

    client: httpx.AsyncClient = app.state.http
    debug: dict = {}

    try:
        iq = await classify_and_rewrite(client, settings, q)
    except MistralError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    debug["intent_raw"] = iq
    needs = bool(iq.get("needs_retrieval", True))
    intent = str(iq.get("intent", "factual"))

    if not needs:
        return QueryResponse(
            answer=str(iq.get("direct_reply", "OK.")),
            needs_retrieval=False,
            intent=intent,
            retrieval_skipped_reason="intent_no_kb",
            debug=debug,
        )

    chunks = store.all_with_embeddings()
    if not chunks:
        return QueryResponse(
            answer="No documents ingested yet. Upload PDFs via /ingest first.",
            needs_retrieval=True,
            intent=intent,
            retrieval_skipped_reason="empty_store",
            debug=debug,
        )

    sem_q = str(iq.get("retrieval_query_semantic", q)).strip() or q
    kw_q = str(iq.get("retrieval_query_keywords", q)).strip() or q

    try:
        q_emb = (await mistral_embed(client, settings, [sem_q]))[0]
    except MistralError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    bm25 = _rebuild_bm25()
    pool_k = settings.rag_top_k
    if settings.rag_cross_encoder_enabled:
        pool_k = max(pool_k, settings.rag_retrieve_k)
    if settings.rag_multi_hop_enabled:
        pool_k = max(pool_k, settings.rag_multi_hop_pool_k)

    candidates = hybrid_search(
        chunks,
        bm25,
        sem_q,
        kw_q,
        q_emb,
        top_k=pool_k,
        rrf_k=settings.rrf_k,
        settings=settings,
        retrieve_pool_k=pool_k,
    )

    if settings.rag_multi_hop_enabled and candidates:
        try:
            follow = await propose_followup_query(
                client,
                settings,
                user_question=q,
                retrieval_semantic=sem_q,
                first_hop_ranked=candidates,
            )
        except MistralError:
            follow = None
        if follow:
            try:
                q_emb2 = (await mistral_embed(client, settings, [follow]))[0]
                hop2 = hybrid_search(
                    chunks,
                    bm25,
                    follow,
                    follow,
                    q_emb2,
                    top_k=pool_k,
                    rrf_k=settings.rrf_k,
                    settings=settings,
                    retrieve_pool_k=pool_k,
                )
                candidates = merge_two_hop_candidates(
                    chunks,
                    bm25,
                    candidates,
                    hop2,
                    q_emb,
                    q_emb2,
                    kw_q,
                    follow,
                    merge_top=pool_k,
                )
                debug["multi_hop_followup"] = follow
            except MistralError:
                debug["multi_hop_followup"] = None
        else:
            debug["multi_hop_followup"] = None

    # Similarity gate uses first rag_top_k candidates after any merge (before CE rerank).
    pre_gate = candidates[: settings.rag_top_k]
    sem_scores = [r.semantic_score for r in pre_gate]
    best = max(sem_scores) if sem_scores else 0.0
    mean_sem = statistics.mean(sem_scores) if sem_scores else 0.0
    debug["top_semantic_scores"] = sem_scores[:5]
    debug["semantic_best"] = best
    debug["semantic_mean_topk"] = mean_sem

    # Require a strong best hit and a non-degenerate average across the fused top-k set.
    weak_set = best < settings.rag_similarity_threshold or mean_sem < settings.rag_similarity_threshold * 0.65
    if not candidates or weak_set:
        return QueryResponse(
            answer="insufficient evidence",
            needs_retrieval=True,
            intent=intent,
            insufficient_evidence=True,
            retrieval_skipped_reason="below_similarity_threshold",
            citations=[],
            debug=debug,
        )

    if settings.rag_cross_encoder_enabled:
        ranked = await asyncio.to_thread(
            rerank_with_cross_encoder,
            q,
            candidates,
            settings.rag_top_k,
            settings.cross_encoder_model,
        )
        debug["rerank"] = "cross_encoder"
    else:
        ranked = candidates[: settings.rag_top_k]

    try:
        answer, hallu_flags = await generate_answer(
            client, settings, question=q, intent=intent, ranked=ranked
        )
    except MistralError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    citations = [
        Citation(
            chunk_id=r.stored.chunk.id,
            source_file=r.stored.chunk.source_file,
            page_start=r.stored.chunk.page_start,
            page_end=r.stored.chunk.page_end,
            similarity=r.semantic_score,
            bm25_score=r.bm25_score,
            rrf_score=r.rrf_score,
            cross_encoder_score=r.cross_encoder_score,
        )
        for r in ranked
    ]
    doc_scores = summarize_scores_by_document(ranked)

    if hallu_flags:
        answer = (
            answer
            + "\n\n_Note: some sentences had weak lexical overlap with retrieved passages; "
            "verify critical claims against the cited chunks._"
        )

    return QueryResponse(
        answer=answer,
        needs_retrieval=True,
        intent=intent,
        document_scores=doc_scores,
        citations=citations,
        hallucination_flags=hallu_flags,
        debug=debug,
    )
