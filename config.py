"""项目共享配置。

集中管理模型缓存目录、数据库路径和常用模型名，
各脚本从这里 import，避免重复硬编码、改一处即可全局生效。
"""

import os

from dotenv import load_dotenv

# 读取项目根目录的 .env（不存在也不报错），把密钥等运行时配置注入环境变量
load_dotenv()

# 模型统一缓存目录：所有脚本从这里读模型，避免散落在系统临时目录
MODEL_DIR = r"F:\models"

# Chroma 本地持久化目录
DB_PATH = "./chroma_db"

# 常用向量模型
EN_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 英文，384 维
ZH_MODEL = "BAAI/bge-small-zh-v1.5"                  # 中文，512 维

# RAG 生成环节用的大模型（OpenAI 兼容接口）。
# 密钥永远从环境变量读，绝不写进代码；base_url 改一行即可换厂商（DeepSeek/通义/OpenAI 等）。
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
