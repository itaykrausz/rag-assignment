import os
from functools import lru_cache
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


###############################################################################
# Configuration
###############################################################################

ANSWER_NOT_FOUND = "I don't know based on the provided Medium articles data."


class Settings:
    """Runtime configuration for the Medium articles RAG API."""

    def __init__(self) -> None:
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
        self.overlap_ratio: float = float(os.getenv("OVERLAP_RATIO", "0.2"))
        self.top_k: int = int(os.getenv("TOP_K", "7"))

        if not 0 <= self.overlap_ratio <= 0.3:
            raise ValueError("OVERLAP_RATIO must be between 0 and 0.3.")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive.")

        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index_name: str = os.getenv(
            "PINECONE_INDEX_NAME", "medium-articles-index"
        )
        self.pinecone_host: str = os.getenv("PINECONE_HOST", "")

        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_org: str = os.getenv("OPENAI_ORGANIZATION", "")
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")

        # Keep the assignment model names configurable, but preserve the required defaults.
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small"
        )
        self.chat_model: str = os.getenv("CHAT_MODEL", "4UHRUIN-gpt-5-mini")
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

        self.system_prompt: str = (
            "You are a Medium-article assistant that answers questions strictly and only "
            "based on the Medium articles dataset context provided to you (metadata and article passages). "
            "You must not use any external knowledge, the open internet, or information that is not "
            "explicitly contained in the retrieved context. "
            f'If the answer cannot be determined from the provided context, respond: "{ANSWER_NOT_FOUND}" '
            "Always explain your answer using the given context, quoting or paraphrasing "
            "the relevant article passage or metadata when helpful."
        )


settings = Settings()


###############################################################################
# API models
###############################################################################

class PromptRequest(BaseModel):
    # Support both names so the API is robust to different assignment wording.
    prompt: Optional[str] = Field(
        default=None,
        description="The natural-language question to ask about the Medium dataset.",
    )
    question: Optional[str] = Field(
        default=None,
        description="Alias for prompt.",
    )

    def normalized_question(self) -> str:
        raw_question = self.prompt if self.prompt is not None else self.question
        return (raw_question or "").strip()


class ContextChunk(BaseModel):
    article_id: str
    title: str
    chunk: str
    score: float


class AugmentedPrompt(BaseModel):
    System: str
    User: str


class PromptResponse(BaseModel):
    response: str
    context: List[ContextChunk]
    Augmented_prompt: AugmentedPrompt


class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int


###############################################################################
# Clients
###############################################################################

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_org:
        kwargs["organization"] = settings.openai_org
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return OpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_pinecone_index() -> Any:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_host:
        return pc.Index(host=settings.pinecone_host)
    return pc.Index(settings.pinecone_index_name)


###############################################################################
# RAG helpers
###############################################################################

def embed_text(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    client = get_openai_client()
    result = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in result.data]


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    client = get_openai_client()
    result = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1,
    )

    content = result.choices[0].message.content
    return (content or "").strip() or ANSWER_NOT_FOUND


def extract_matches(query_result: Any) -> list[Any]:
    if isinstance(query_result, dict):
        return list(query_result.get("matches", []))
    return list(getattr(query_result, "matches", []) or [])


def get_match_score(match: Any) -> float:
    if isinstance(match, dict):
        return float(match.get("score", 0.0))
    return float(getattr(match, "score", 0.0) or 0.0)


def get_match_metadata(match: Any) -> dict[str, Any]:
    if isinstance(match, dict):
        metadata = match.get("metadata", {}) or {}
    else:
        metadata = getattr(match, "metadata", {}) or {}
    return dict(metadata)


def build_context_chunk(match: Any) -> ContextChunk:
    metadata = get_match_metadata(match)
    return ContextChunk(
        article_id=str(metadata.get("article_id", "")),
        title=str(metadata.get("title", "")),
        chunk=str(metadata.get("text", "")),
        score=get_match_score(match),
    )


def format_context_block(chunk: ContextChunk) -> str:
    return f"[Article ID: {chunk.article_id} | Title: {chunk.title}]\n{chunk.chunk}"


def build_user_prompt(question: str, context_chunks: List[ContextChunk]) -> str:
    context_text = "\n\n---\n\n".join(
        format_context_block(chunk) for chunk in context_chunks if chunk.chunk.strip()
    )

    if not context_text:
        context_text = "No relevant context was retrieved from the Medium articles dataset."

    return (
        f"Question:\n{question}\n\n"
        f"Retrieved Medium articles dataset context:\n{context_text}\n\n"
        "Answer using only the retrieved context. If the context does not contain "
        f"the answer, respond exactly: {ANSWER_NOT_FOUND}"
    )


###############################################################################
# FastAPI app
###############################################################################

app = FastAPI(title="Medium Article RAG Assistant", version="1.0.0")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "Medium Article RAG Assistant"}


@app.post("/api/prompt", response_model=PromptResponse)
def handle_prompt(req: PromptRequest) -> PromptResponse:
    question = req.normalized_question()
    if not question:
        raise HTTPException(status_code=400, detail="Request body must include a non-empty prompt or question.")

    try:
        question_embedding = embed_text([question])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {exc}") from exc

    try:
        index = get_pinecone_index()
        query_result = index.query(
            vector=question_embedding,
            top_k=settings.top_k,
            include_metadata=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {exc}") from exc

    context_chunks = [build_context_chunk(match) for match in extract_matches(query_result)]
    user_prompt = build_user_prompt(question, context_chunks)

    try:
        answer = chat_completion(settings.system_prompt, user_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {exc}") from exc

    return PromptResponse(
        response=answer,
        context=context_chunks,
        Augmented_prompt=AugmentedPrompt(
            System=settings.system_prompt,
            User=user_prompt,
        ),
    )


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    return StatsResponse(
        chunk_size=settings.chunk_size,
        overlap_ratio=settings.overlap_ratio,
        top_k=settings.top_k,
    )
