"""最小示例：用 Chroma 向量数据库做语义检索。

流程：把几段文字存进 Chroma -> 输入一句话 -> 返回语义最接近的几条。

向量化用 fastembed 的 all-MiniLM-L6-v2，模型统一缓存在 F:\\models，
向量算好后显式喂给 Chroma（add/query 都传 embeddings）。

数据持久化到本地 ./chroma_db 目录，重复运行不会重复入库（用 upsert）。

运行：
    python chroma_demo.py
"""

import chromadb
from fastembed import TextEmbedding

from config import EN_MODEL, MODEL_DIR, DB_PATH

COLLECTION_NAME = "docs"

_model = TextEmbedding(model_name=EN_MODEL, cache_dir=MODEL_DIR)

# 预置的知识库文档（id 与文本一一对应）
DOCUMENTS = {
    "doc1": "Chroma is an open-source vector database for building AI applications.",
    "doc2": "Cats are small carnivorous mammals often kept as pets.",
    "doc3": "Embedding models turn text into numeric vectors for semantic search.",
    "doc4": "The Great Wall of China is over 13,000 miles long.",
    "doc5": "Cosine similarity measures how close two vectors are in direction.",
}


def embed(texts):
    return [v.tolist() for v in _model.embed(texts)]


def main() -> None:
    # 持久化客户端：数据写到本地磁盘，下次运行依然在
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # upsert：按 id 写入，已存在则覆盖，避免重复运行报错
    ids = list(DOCUMENTS.keys())
    texts = list(DOCUMENTS.values())
    collection.upsert(ids=ids, documents=texts, embeddings=embed(texts))

    print(f"当前库中文档数: {collection.count()}\n")

    query = "How do I do semantic search with vectors?"
    print(f"查询: {query}\n")

    results = collection.query(query_embeddings=embed([query]), n_results=3)

    print("最相似的 3 条结果：")
    docs = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    for rank, (doc_id, doc, dist) in enumerate(zip(ids, docs, distances), start=1):
        # distance 越小越相似（默认是 L2 距离的平方）
        print(f"  {rank}. [{doc_id}] 距离={dist:.4f}  {doc}")


if __name__ == "__main__":
    main()
