import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "medium-articles-index")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not set.")

print(f"Using Pinecone index: {index_name}", flush=True)

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

stats = index.describe_index_stats()
print("Index stats:", stats, flush=True)

pattern = re.compile(r"article-(\d+)-chunk-(\d+)$")

max_article_id = -1
total_ids_seen = 0


def extract_vector_id(item):
    if isinstance(item, str):
        return item

    if hasattr(item, "id"):
        return item.id

    if isinstance(item, dict) and "id" in item:
        return item["id"]

    return str(item)


print("Listing vector IDs...", flush=True)

for ids_page in index.list(prefix="article-"):
    for item in ids_page:
        vector_id = extract_vector_id(item)

        total_ids_seen += 1

        match = pattern.match(vector_id)
        if match:
            article_id = int(match.group(1))
            max_article_id = max(max_article_id, article_id)

print(f"Total vector IDs seen: {total_ids_seen}", flush=True)
print(f"Highest article_id found: {max_article_id}", flush=True)

if max_article_id >= 0:
    print(f"Suggested START_ROW: {max_article_id + 1}", flush=True)
    print(f"Safer START_ROW, with overlap: {max(0, max_article_id - 50)}", flush=True)
else:
    print("No matching article-* vectors found.", flush=True)