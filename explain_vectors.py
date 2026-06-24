"""原理演示：把"文字 -> 向量 -> 距离 -> 排序"的每一步数字都打印出来。

用 fastembed（all-MiniLM-L6-v2）生成向量，再用 numpy 手动算距离，
让你亲眼看到 Chroma 检索背后到底发生了什么。

运行：
    python explain_vectors.py
"""

import numpy as np
from fastembed import TextEmbedding

from config import EN_MODEL, MODEL_DIR


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离（Chroma 默认度量是它的平方）。越小越相似。"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度：看两个向量方向的夹角。越接近 1 越相似。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    model = TextEmbedding(model_name=EN_MODEL, cache_dir=MODEL_DIR)

    sentences = [
        "How do I search text by meaning?",   # 0 查询句
        "What is semantic search?",           # 1 与 0 含义相近
        "Vector databases store embeddings.", # 2 相关
        "My cat likes to sleep all day.",     # 3 完全无关
    ]
    # sentences = [
    #     "今天午饭吃什么",   # 0 查询句
    #     "天空很蓝",           # 1 与 0 含义相近
    #     "饭菜很好吃", # 2 相关
    #     "江枫渔火对愁眠",     # 3 完全无关
    # ]
    vectors = [np.array(v) for v in model.embed(sentences)]

    # ---------- 第 1 步：看看向量长什么样 ----------
    print("=" * 60)
    print("第 1 步：文字 -> 向量")
    print("=" * 60)
    v0 = vectors[0]
    print(f"句子: {sentences[0]}")
    print(f"  向量维度: {len(v0)}")
    print(f"  前 8 个数字: {np.round(v0[:8], 4)}")
    print(f"  向量长度(模): {np.linalg.norm(v0):.4f}  (模型已归一化，约等于 1)")

    # ---------- 第 2 步：手动算句子两两之间的距离 ----------
    print("\n" + "=" * 60)
    print("第 2 步：句子之间的距离 / 相似度（以句 0 为基准）")
    print("=" * 60)
    query = vectors[0]
    print(f"基准句: \"{sentences[0]}\"\n")
    print(f"{'对比句':<40}{'L2距离':>10}{'余弦相似度':>12}")
    print("-" * 62)
    for i in range(1, len(sentences)):
        dist = l2_distance(query, vectors[i])
        cos = cosine_similarity(query, vectors[i])
        print(f"{sentences[i]:<40}{dist:>10.4f}{cos:>12.4f}")

    # ---------- 第 3 步：按距离排序 = 检索 ----------
    print("\n" + "=" * 60)
    print("第 3 步：按距离排序，就是 Chroma 做的检索")
    print("=" * 60)
    scored = [
        (i, l2_distance(query, vectors[i]))
        for i in range(1, len(sentences))
    ]
    scored.sort(key=lambda x: x[1])  # 距离从小到大
    for rank, (i, dist) in enumerate(scored, start=1):
        print(f"  第{rank}名  距离={dist:.4f}  {sentences[i]}")

    print("\n结论：距离最小的就是语义最接近的，这就是向量检索的全部秘密。")


if __name__ == "__main__":
    main()
