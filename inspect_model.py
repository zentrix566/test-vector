"""扒开向量模型的底层：分词器 + ONNX 接口，二合一查看工具。

把文字变成向量分两段，这个脚本分别能看清这两段：
  1. 分词器（tokenizer）：把文字切成词元、查表转成整数 id（input_ids）；
  2. ONNX 模型：吃这些 id，算出 last_hidden_state 向量。
两段串起来就是 文字 → id → 向量 的完整链路。

分词表是从海量语料里统计学出来的高频片段集合，和模型一起训练、绑死，
不能跨模型混用；WordPiece 词元里带 ## 前缀的，表示接在前一个词元后面。

文件默认在 MODEL_DIR 下自动找 tokenizer.json / model.onnx，也可命令行指定：
    python inspect_model.py                 # 两样都看（默认）
    python inspect_model.py tokenizer       # 只看分词器
    python inspect_model.py onnx            # 只看 ONNX 接口
    python inspect_model.py onnx "F:\\...\\model.onnx"   # 指定文件
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config import MODEL_DIR

# 演示样例：英文能看出子词拆分（## 片段），中文基本是逐字成词元
SAMPLES = [
    "Hello, tokenization is fun!",
    "你好，向量数据库的世界",
]


# ============ 分词器部分（原 inspect_tokenizer.py，功能不变）============

def find_tokenizer() -> str:
    """在 MODEL_DIR 下递归找第一个 tokenizer.json，找不到就提示退出。"""
    matches = list(Path(MODEL_DIR).rglob("tokenizer.json"))
    if not matches:
        sys.exit(f"在 {MODEL_DIR} 下没找到 tokenizer.json，请先下载模型或手动传入路径。")
    return str(matches[0])


def show_vocab_samples(tok: Tokenizer, n: int = 8) -> None:
    """从词表里挑几个普通词元和几个 ## 子词，看看它们长什么样。"""
    vocab = tok.get_vocab()  # {词元: 编号}
    plain = sorted(t for t in vocab if not t.startswith("##") and t.isalpha())[:n]
    pieces = sorted(t for t in vocab if t.startswith("##"))[:n]
    print("普通词元示例:", {t: vocab[t] for t in plain})
    print("子词(##)示例:", {t: vocab[t] for t in pieces})


def show_encoding(tok: Tokenizer, text: str) -> None:
    """把一句话过一遍分词器，打印切出的词元和对应编号。"""
    enc = tok.encode(text)
    print(f"\n原文: {text}")
    print(f"  词元: {enc.tokens}")
    print(f"  编号: {enc.ids}")
    print(f"  共 {len(enc.ids)} 个词元（含首尾的 [CLS]/[SEP] 标记）")


def inspect_tokenizer(path: str) -> None:
    """查看分词器：词表多大、一句话被切成哪些词元、编号是多少。"""
    print(f"分词器文件: {path}\n")
    tok = Tokenizer.from_file(path)
    # 有的分词器默认把句子补/截到固定长度，演示时关掉，输出更干净
    tok.no_padding()
    tok.no_truncation()
    print(f"词表大小: {tok.get_vocab_size()} 个词元\n")
    show_vocab_samples(tok)
    for text in SAMPLES:
        show_encoding(tok, text)


# ============ ONNX 部分（原 inspect_onnx.py，功能不变）============

def find_onnx() -> str:
    """在 MODEL_DIR 下递归找第一个 model.onnx，找不到就给出提示退出。"""
    matches = list(Path(MODEL_DIR).rglob("model.onnx"))
    if not matches:
        sys.exit(f"在 {MODEL_DIR} 下没找到 model.onnx，请先下载模型或手动传入路径。")
    return str(matches[0])


def print_interface(sess: ort.InferenceSession) -> None:
    """打印模型接口：输入和输出各自的名字、形状、类型。"""
    print("=== 输入（调用时要喂这些）===")
    for i in sess.get_inputs():
        print(f"  {i.name:16s} shape={i.shape}  {i.type}")
    print("=== 输出（调用后拿到这些）===")
    for o in sess.get_outputs():
        print(f"  {o.name:16s} shape={o.shape}  {o.type}")


def demo_call(sess: ort.InferenceSession) -> None:
    """按接口造一组最小输入跑一次推理，演示"接口怎么调用"。

    注意：这里用的是手工编的假 token id，只为演示调用机制。真正用模型
    要先用配套分词器把文字转成 input_ids，项目里 fastembed 已经把
    分词 + 调用都封装好了（见 embed_local.py），平时无需碰这一层。
    """
    seq_len = 4  # 假装一句话被切成了 4 个词

    # 按接口逐个输入造数据：这个模型三个输入都是 int64、形状 [batch, seq]
    feeds = {i.name: np.ones((1, seq_len), dtype=np.int64) for i in sess.get_inputs()}

    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, feeds)  # ← 真正调用模型就这一行

    print("\n=== 调用一次的结果 ===")
    for name, arr in zip(output_names, outputs):
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")


def inspect_onnx(path: str) -> None:
    """查看 ONNX 模型的接口（输入/输出名字、形状、类型）并演示调用。"""
    print(f"模型文件: {path}\n")
    # 加载模型、建立推理会话，CPU 后端最通用、装上即可用
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print_interface(sess)
    demo_call(sess)


# ============ 入口分发 ============

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("all", "onnx", "tokenizer"):
        sys.exit("用法: python inspect_model.py [all|tokenizer|onnx] [文件路径]")
    path = sys.argv[2] if len(sys.argv) > 2 else None

    if mode in ("all", "tokenizer"):
        inspect_tokenizer(path or find_tokenizer())
    if mode == "all":
        print("\n" + "=" * 50 + "\n")
    if mode in ("all", "onnx"):
        inspect_onnx(path or find_onnx())


if __name__ == "__main__":
    main()
