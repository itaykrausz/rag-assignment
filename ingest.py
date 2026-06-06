import os
from typing import Any, List

import pandas as pd
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class IngestSettings:
    def __init__(self) -> None:
        self.dataset_path: str = os.getenv("DATASET_PATH", "")
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
        self.overlap_ratio: float = float(os.getenv("OVERLAP_RATIO", "0.2"))
        self.batch_size: int = int(os.getenv("UPSERT_BATCH_SIZE", "64"))
        self.start_row: int = int(os.getenv("START_ROW", "0"))

        if not 0 <= self.overlap_ratio <= 0.3:
            raise ValueError("OVERLAP_RATIO must be between 0 and 0.3.")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")
        if self.batch_size <= 0:
            raise ValueError("UPSERT_BATCH_SIZE must be positive.")
        if self.start_row < 0:
            raise ValueError("START_ROW must be non-negative.")

        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_org: str = os.getenv("OPENAI_ORGANIZATION", "")
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small"
        )
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index_name: str = os.getenv(
            "PINECONE_INDEX_NAME", "medium-articles-index"
        )
        self.pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
        self.pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")


settings = IngestSettings()


def get_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_org:
        kwargs["organization"] = settings.openai_org
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def get_or_create_index() -> Any:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            vector_type="dense",
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
            deletion_protection="disabled",
        )
    return pc.Index(settings.pinecone_index_name)


def embed_text(texts: List[str]) -> List[List[float]]:
    client = get_openai_client()
    result = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in result.data]


def chunk_text(text: str) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    step = max(1, int(settings.chunk_size * (1 - settings.overlap_ratio)))

    chunks: List[str] = []
    for start in range(0, len(tokens), step):
        end = start + settings.chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        chunks.append(encoding.decode(chunk_tokens))
        if end >= len(tokens):
            break
    return chunks


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def main() -> None:
    if not settings.dataset_path:
        raise RuntimeError("DATASET_PATH is not set.")
    if not os.path.exists(settings.dataset_path):
        raise FileNotFoundError(f"CSV file not found: {settings.dataset_path}")

    df = pd.read_csv(settings.dataset_path)
    required_columns = {"title", "text", "url", "authors", "timestamp", "tags"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df.iloc[settings.start_row:]

    index = get_or_create_index()
    pending_vectors: List[tuple[str, List[float], dict[str, Any]]] = []
    total_articles = 0
    total_chunks = 0

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting articles"):
        article_text = clean_value(row.get("text"))
        if not article_text.strip():
            continue

        total_articles += 1
        article_id = str(row_index)
        title = clean_value(row.get("title"))
        url = clean_value(row.get("url"))
        authors = clean_value(row.get("authors"))
        timestamp = clean_value(row.get("timestamp"))
        tags = clean_value(row.get("tags"))

        chunks = chunk_text(article_text)
        total_chunks += len(chunks)

        for batch_start in range(0, len(chunks), settings.batch_size):
            chunk_batch = chunks[batch_start : batch_start + settings.batch_size]
            embeddings = embed_text(chunk_batch)

            for offset, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
                chunk_index = batch_start + offset
                vector_id = f"article-{article_id}-chunk-{chunk_index}"
                metadata = {
                    "article_id": article_id,
                    "title": title,
                    "url": url,
                    "authors": authors,
                    "timestamp": timestamp,
                    "tags": tags,
                    "text": chunk,
                }
                pending_vectors.append((vector_id, embedding, metadata))

            if len(pending_vectors) >= settings.batch_size:
                index.upsert(vectors=pending_vectors)
                pending_vectors = []

    if pending_vectors:
        index.upsert(vectors=pending_vectors)

    print(f"Finished ingestion: {total_articles} articles, {total_chunks} chunks.")


if __name__ == "__main__":
    main()