# -*- coding: utf-8 -*-
"""
全局配置：模型超参数 + AI 选词参数
"""

# 用来做 token classification 的预训练模型
BASE_MODEL_NAME = "distilbert-base-cased"  # 轻一点，跑得快些

# 训练好之后保存 / 加载模型的目录
MODEL_DIR = "./keyword-bert-inspec"

# 训练相关
MAX_SEQ_LEN = 256
EPOCHS = 5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# AI 选词相关
DEFAULT_TOP_N = 30                 # 最后选出来的词表大小（大概）
MAX_CANDIDATES_BEFORE_LLM = 120    # rule-based + 模型合并前的候选上限
MIN_TOKEN_LEN = 4                  # 候选词最小长度

# 如果未来要接大语言模型精修，可以用这个开关判断
USE_LLM_REFINER = False
