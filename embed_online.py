"""在线向量模型调用 demo：火山方舟 doubao-embedding-vision。

与本地模型（all-MiniLM、bge）不同，在线 embedding 通过 API 调用，
需要联网、消耗 token，但通常维度更高、中文效果更好。

运行前确保 .env 里已配置：
    ARK_API_KEY=your-ark-key-here
    ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
    ARK_EMBED_ENDPOINT=/embeddings        # endpoint 路径，按文档调整
    EMBED_MODEL=doubao-embedding-vision

运行：
    python embed_online.py
"""

import os

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY", "")
BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ENDPOINT = os.getenv("ARK_EMBED_ENDPOINT", "/embeddings/multimodal")
MODEL = os.getenv("EMBED_MODEL", "ep-20260625110146-r8f6b")


def cosine(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """余弦相似度，两个向量方向的夹角有多近。范围 [-1, 1]，越接近 1 越相似。"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed(texts: list[str]) -> list[list[float]]:
    """调用火山方舟 Embedding API，把一批文字转成向量。"""
    if not API_KEY or API_KEY == "your-ark-key-here":
        raise RuntimeError(
            "ARK_API_KEY 未配置。请在 .env 里填入真实的火山方舟 API 密钥，"
            "然后 source .env 或重启终端。"
        )

    url = BASE_URL.rstrip("/") + ENDPOINT
    print(f"[调试] 请求 URL: {url}")

    payload = {
        "model": MODEL,
        "input": texts,
    }
    # doubao-embedding-vision 支持 dimensions 参数，不传则返回默认维度
    # 如需指定维度，在 .env 里加 EMBED_DIMENSIONS=2560
    dim = os.getenv("EMBED_DIMENSIONS")
    if dim:
        payload["dimensions"] = int(dim)

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    # 调试：先打印状态码和部分响应，方便排查 401/404
    print(f"[调试] 状态码: {resp.status_code}")
    if resp.status_code != 200:
        try:
            err = resp.json()
            print(f"[调试] 错误响应: {err}")
        except Exception:
            print(f"[调试] 响应文本: {resp.text[:500]}")
        resp.raise_for_status()

    data = resp.json()
    print(f"[调试] 响应结构键: {list(data.keys())}")

    embeddings = sorted(data["data"], key=lambda x: x["index"])
    return [e["embedding"] for e in embeddings]


def main() -> None:
    print(f"模型: {MODEL}")
    print(f"Base URL: {BASE_URL}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"完整 URL: {BASE_URL.rstrip('/')}{ENDPOINT}\n")

    sentences = [
        "How do I search text by meaning?",
        "What is semantic search?",
        "Vector databases store embeddings.",
        "My cat likes to sleep all day.",
    ]

    print("请求原文:")
    for s in sentences:
        print(f"  - {s}")

    vectors = embed(sentences)
    print(f"\n获取到 {len(vectors)} 条向量，每条 {len(vectors[0])} 维")
    print(f"第 1 条前 5 个数字: {vectors[0][:5]}\n")

    print(f"以『{sentences[0]}』为基准：")
    print(f"{'对比句':<40}  {'余弦相似度':>12}")
    print("-" * 56)
    for i in range(1, len(sentences)):
        sim = cosine(vectors[0], vectors[i])
        print(f"{sentences[i]:<40}  {sim:>12.4f}")


if __name__ == "__main__":
    main()

