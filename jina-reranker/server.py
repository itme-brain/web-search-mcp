import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


MODEL_NAME = os.environ.get("JINA_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
API_KEY = os.environ.get("JINA_API_KEY", "")

# Force CPU so the reranker does not contend with the GPU-backed chat model.
MODEL = CrossEncoder(
    MODEL_NAME,
    device="cpu",
    trust_remote_code=True,
)

app = FastAPI()


class DocumentObject(BaseModel):
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: list[str | DocumentObject]
    model: str | None = None
    top_n: int | None = Field(default=None, ge=1)


def normalize_document(document: str | DocumentObject) -> str:
    if isinstance(document, str):
        return document
    return document.text


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/rerank")
def rerank(
    payload: RerankRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if API_KEY:
        expected = f"Bearer {API_KEY}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    documents = [normalize_document(doc) for doc in payload.documents]
    sentence_pairs = [[payload.query, doc] for doc in documents]
    scores = MODEL.predict(sentence_pairs).tolist()

    results = [
        {
            "index": index,
            "relevance_score": float(score),
            "document": {"text": documents[index]},
        }
        for index, score in enumerate(scores)
    ]
    results.sort(key=lambda item: item["relevance_score"], reverse=True)

    if payload.top_n is not None:
        results = results[: payload.top_n]

    total_tokens = sum(len(payload.query.split()) + len(doc.split()) for doc in documents)

    return {
        "model": payload.model or MODEL_NAME,
        "usage": {"total_tokens": total_tokens},
        "results": results,
    }
