"""告警匹配知识库 MVP。

功能：
  - 来一条新告警 -> 在知识库里找最相似的已知问题
  - 相似度 >= 阈值：判定为"已知问题"，返回根因和处置办法，并累加出现次数
  - 相似度 <  阈值：判定为"新问题"，自动存入知识库（状态=待定位）

向量模型：BAAI/bge-small-zh-v1.5（中文效果好），向量直接喂给 Chroma。
数据持久化在本地 ./chroma_db 的 alert_kb 集合。

运行：
    python alert_kb.py
"""

import re
import time
import uuid

import chromadb
from fastembed import TextEmbedding

from config import ZH_MODEL, MODEL_DIR, DB_PATH

COLLECTION_NAME = "alert_kb"
SIMILARITY_THRESHOLD = 0.75  # 余弦相似度 >= 此值即判定为同一已知问题（需按真实数据校准）

_model = TextEmbedding(model_name=ZH_MODEL, cache_dir=MODEL_DIR)


def normalize(text: str) -> str:
    """归一化告警文本：去掉时间戳、IP、ID、数字等易变部分，提取稳定指纹。"""
    text = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", " ", text)  # 时间戳
    text = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", " ", text)                   # IP
    text = re.sub(r"\b[0-9a-fA-F]{8,}\b", " ", text)                         # 长 hex / id
    text = re.sub(r"\d+", " ", text)                                         # 其余数字
    text = re.sub(r"\s+", " ", text).strip()                                 # 压缩空白
    return text


def embed(text: str) -> list[float]:
    return list(_model.embed([text]))[0].tolist()


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    # 用余弦距离：相似度 = 1 - 距离
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# 预置的初始知识库（实际项目中这部分来自历史告警导入）
SEED_PROBLEMS = [
    {
        "id": "kb-001",
        "alert": "订单服务数据库连接池耗尽，大量请求超时",
        "root_cause": "连接池配置过小，且慢查询占用连接未释放",
        "solution": "1) 临时扩大连接池 maxActive; 2) 定位并优化慢 SQL; 3) 加连接超时回收",
        "service": "order",
        "severity": "P1",
    },
    {
        "id": "kb-002",
        "alert": "Nginx 网关返回大量 502，后端无响应",
        "root_cause": "上游服务进程 OOM 被杀，健康检查未及时摘除",
        "solution": "1) 重启上游服务; 2) 调大内存限制; 3) 检查健康检查间隔",
        "service": "gateway",
        "severity": "P1",
    },
    {
        "id": "kb-003",
        "alert": "磁盘使用率超过 90%，日志目录占满",
        "root_cause": "日志未做轮转，历史日志堆积",
        "solution": "1) 清理过期日志; 2) 配置 logrotate 按天轮转并保留 7 天",
        "service": "host",
        "severity": "P2",
    },
]


def seed_knowledge_base():
    """把初始问题写入知识库（用固定 id，重复运行不会重复入库）。"""
    col = get_collection()
    col.upsert(
        ids=[p["id"] for p in SEED_PROBLEMS],
        embeddings=[embed(normalize(p["alert"])) for p in SEED_PROBLEMS],
        documents=[p["alert"] for p in SEED_PROBLEMS],
        metadatas=[
            {
                "root_cause": p["root_cause"],
                "solution": p["solution"],
                "service": p["service"],
                "severity": p["severity"],
                "status": "已解决",
                "count": 1,
            }
            for p in SEED_PROBLEMS
        ],
    )


def match_alert(raw_alert: str) -> dict:
    """匹配一条告警：命中返回已知问题，未命中则存为新问题。"""
    col = get_collection()
    fingerprint = normalize(raw_alert)
    vec = embed(fingerprint)

    result = col.query(query_embeddings=[vec], n_results=1)

    similarity = 0.0
    if result["ids"][0]:
        distance = result["distances"][0][0]
        similarity = 1 - distance  # 余弦空间：相似度 = 1 - 距离

    if similarity >= SIMILARITY_THRESHOLD:
        hit_id = result["ids"][0][0]
        meta = result["metadatas"][0][0]
        doc = result["documents"][0][0]
        # 累加出现次数、更新最近发生时间
        col.update(
            ids=[hit_id],
            metadatas=[{**meta, "count": meta.get("count", 1) + 1,
                        "last_seen": time.strftime("%Y-%m-%d %H:%M:%S")}],
        )
        return {
            "matched": True,
            "similarity": round(similarity, 4),
            "kb_id": hit_id,
            "known_alert": doc,
            "root_cause": meta["root_cause"],
            "solution": meta["solution"],
        }

    # 未命中：作为新问题入库，等待运维补充根因和处置办法
    new_id = "alert-" + uuid.uuid4().hex[:8]
    col.add(
        ids=[new_id],
        embeddings=[vec],
        documents=[raw_alert],
        metadatas=[{
            "root_cause": "",
            "solution": "",
            "service": "",
            "severity": "",
            "status": "待定位",
            "count": 1,
            "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
        }],
    )
    return {
        "matched": False,
        "similarity": round(similarity, 4),
        "new_id": new_id,
    }


def match_and_advise(raw_alert: str) -> dict:
    """检索 + 生成的完整 RAG：先匹配知识库，再让大模型生成定制化处置建议。"""
    from rag import generate_advice  # 延迟导入：仅 RAG 路径才依赖 openai

    matched = match_alert(raw_alert)
    return generate_advice(raw_alert, matched)


def main():
    # 仅为演示可复现：先清空集合再重建（生产环境不要这样做）
    try:
        chromadb.PersistentClient(path=DB_PATH).delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    seed_knowledge_base()

    incoming = [
        "2026-06-24 10:23:01 order 服务连接池满了，请求大量超时 trace=a1b2c3d4e5f6",
        "网关 192.168.1.5 大面积 502，后端没响应",
        "Redis 集群主从切换失败，缓存大量击穿到数据库",  # 这是新问题
    ]

    for raw in incoming:
        print("=" * 70)
        print("告警:", raw)
        res = match_alert(raw)
        if res["matched"]:
            print(f"  ✅ 命中已知问题 [{res['kb_id']}]  相似度={res['similarity']}")
            print(f"     已知现象: {res['known_alert']}")
            print(f"     根因    : {res['root_cause']}")
            print(f"     处置办法: {res['solution']}")
        else:
            print(f"  🆕 新问题（最高相似度仅 {res['similarity']}，低于阈值 {SIMILARITY_THRESHOLD}）")
            print(f"     已存入知识库 [{res['new_id']}]，状态=待定位，待运维补充根因与处置办法")

    # 完整 RAG 演示：检索 + 大模型生成（未配置密钥时自动降级为返回知识库原文）
    print("\n" + "=" * 70)
    print("【RAG 生成演示】检索命中后，让大模型生成针对性处置建议")
    rag_res = match_and_advise("网关 192.168.1.5 大面积 502，后端没响应")
    print(f"  来源: {rag_res['advice_source']}（llm=大模型生成 / knowledge_base=未配密钥降级）")
    print(f"  建议:\n{rag_res['advice']}")


if __name__ == "__main__":
    main()
