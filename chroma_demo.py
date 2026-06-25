"""最小示例：用 Chroma 向量数据库做语义检索——文章配套版。

流程：
  1. 把几条中文告警原文向量化后存入 Chroma（带 metadata）
  2. 输入一条新告警
  3. 返回语义最相似的已知问题，附带距离和元数据

向量化用 fastembed 的 all-MiniLM-L6-v2（英文模型，仅演示原理），
模型统一缓存在 MODEL_DIR（见 config.py）。
向量算好后显式喂给 Chroma（upsert / query 都传 embeddings），
这样每一步都看得清清楚楚。

数据持久化到本地 ./chroma_db 目录，重复运行不会重复入库（用 upsert）。

运行：
    python chroma_demo.py
"""

import chromadb
from fastembed import TextEmbedding

from config import EN_MODEL, MODEL_DIR, DB_PATH

COLLECTION_NAME = "demo"

# 预置的告警知识库
IDS = ["alert_001", "alert_002", "alert_003"]
DOCUMENTS = [
    "网关 502 错误，上游服务无响应",
    "order 服务连接池耗尽，请求超时",
    "Redis 主从切换失败，集群状态异常",
]
METADATAS = [
    {"service": "gateway", "severity": "P1"},
    {"service": "order", "severity": "P2"},
    {"service": "redis", "severity": "P1"},
]

_model = TextEmbedding(model_name=EN_MODEL, cache_dir=MODEL_DIR)


def embed(texts):
    """把文字列表转成向量列表（List[List[float]]）。"""
    return [v.tolist() for v in _model.embed(texts)]


def main() -> None:
    # 持久化客户端：数据写到本地磁盘，下次运行依然在
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # upsert：按 id 写入，已存在则覆盖，避免重复运行报错
    collection.upsert(
        ids=IDS,
        documents=DOCUMENTS,
        metadatas=METADATAS,
        embeddings=embed(DOCUMENTS),
    )
    print(f"✅ 当前库中文档数: {collection.count()}\n")

    # 模拟一条新告警，去库里检索最相似的已知问题
    query = "生产网关频繁 502，上游服务无响应"
    print(f"🔍 查询: {query}\n")

    results = collection.query(query_embeddings=embed([query]), n_results=3)

    # 格式化输出
    print(f"{'排名':>4}  {'距离':>8}  {'id':<15}  {'原文'}")
    print(f"{'----':>4}  {'--------':>8}  {'-'*15:>15}  {'-'*20}")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        dist = results["distances"][0][i]
        doc = results["documents"][0][i]
        print(f"{i + 1:>4}  {dist:>8.4f}  {doc_id:<15}  {doc}")

    # 顺便看一眼命中的 metadata
    meta = results["metadatas"][0][0]
    print(f"\n最匹配的服务: {meta.get('service', '?')}  级别: {meta.get('severity', '?')}")


if __name__ == "__main__":
    main()
