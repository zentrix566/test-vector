"""项目共享配置。

集中管理模型缓存目录、数据库路径和常用模型名，
各脚本从这里 import，避免重复硬编码、改一处即可全局生效。
"""

# 模型统一缓存目录：所有脚本从这里读模型，避免散落在系统临时目录
MODEL_DIR = r"F:\models"

# Chroma 本地持久化目录
DB_PATH = "./chroma_db"

# 常用向量模型
EN_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 英文，384 维
ZH_MODEL = "BAAI/bge-small-zh-v1.5"                  # 中文，512 维
