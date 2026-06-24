"""RAG 的生成（Generation）环节。

补全 alert_kb 缺失的那一步：把向量检索到的"已知问题 + 根因 + 处置办法"
作为上下文喂给大模型，让它针对当前这条具体告警生成定制化的处置建议。

  检索（alert_kb 已有） + 生成（本模块） = 完整的 RAG

大模型走 OpenAI 兼容接口，密钥从环境变量读（见 config.LLM_*）。
未配置密钥时优雅降级：不调用模型，直接返回知识库里的原始处置办法。
"""

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None

SYSTEM_PROMPT = (
    "你是资深 SRE 运维专家。根据给定的知识库参考信息，"
    "针对当前这条具体告警，给出清晰、可执行的处置建议。"
    "用中文回答，分条列出步骤，不要编造知识库里没有的信息。"
)


def _build_user_prompt(raw_alert: str, matched: dict) -> str:
    """把检索结果拼成喂给模型的上下文。命中和未命中用不同的提示。"""
    if matched.get("matched"):
        return (
            f"【当前告警】\n{raw_alert}\n\n"
            f"【知识库命中的相似问题】(相似度 {matched['similarity']})\n"
            f"现象：{matched['known_alert']}\n"
            f"根因：{matched['root_cause']}\n"
            f"参考处置：{matched['solution']}\n\n"
            "请结合当前告警的具体细节，给出针对性的处置步骤。"
        )
    return (
        f"【当前告警】\n{raw_alert}\n\n"
        "知识库中没有足够相似的已知问题。请基于通用运维经验，"
        "给出这条告警的初步排查方向和应急处置步骤。"
    )


def generate_advice(raw_alert: str, matched: dict) -> dict:
    """基于检索结果生成定制化处置建议。

    返回结构在原 matched 基础上增加 advice / advice_source 两个字段。
    """
    if _client is None:
        # 未配置大模型：降级为直接返回知识库原文，不影响主流程
        fallback = matched.get("solution", "") if matched.get("matched") else ""
        return {
            **matched,
            "advice": fallback,
            "advice_source": "knowledge_base",  # 未走大模型
        }

    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(raw_alert, matched)},
        ],
        temperature=0.3,
    )
    return {
        **matched,
        "advice": resp.choices[0].message.content.strip(),
        "advice_source": "llm",  # 由大模型生成
    }
