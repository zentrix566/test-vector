"""查看 Chroma 本地库里到底存了什么。

Chroma 的数据是 chroma_db/ 下的 sqlite + .bin 文件，直接打开看不懂：
  - chroma.sqlite3  存文档原文、id、metadata（可用 SQLite 工具打开）
  - <集合id>/*.bin   存向量的 HNSW 索引（二进制，不可读）

正确的查看方式是走 Chroma 的 API 把内容读出来，这个脚本就是干这个的：
列出所有集合，并打印每条记录的 id、原文、metadata 和向量预览。

运行：
    python inspect_chroma.py
"""

import chromadb

from config import DB_PATH

PREVIEW_LIMIT = 5  # 每个集合最多预览几条


def main() -> None:
    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()

    if not collections:
        print(f"{DB_PATH} 里还没有任何集合，先跑一下 chroma_demo.py 或 alert_kb.py。")
        return

    for col in collections:
        print(f"== 集合 {col.name} | 共 {col.count()} 条 ==")
        data = col.get(
            include=["documents", "metadatas", "embeddings"],
            limit=PREVIEW_LIMIT,
        )
        for i, doc_id in enumerate(data["ids"]):
            emb = data["embeddings"][i]
            preview = [round(float(x), 4) for x in emb[:3]]
            print(f"  id={doc_id}")
            print(f"    文本: {data['documents'][i]}")
            print(f"    metadata: {data['metadatas'][i]}")
            print(f"    向量: {len(emb)} 维，前 3 维 {preview} ...")
        print()


if __name__ == "__main__":
    main()
