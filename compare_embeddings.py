from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 加载一个轻量级的动态向量模型（会自动下载，约 80MB）
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 准备两个含有 "bank" 但意思完全不同的句子
sentence1 = "I sat on the bank of the river."      # 河岸
sentence2 = "I deposited money in the bank."       # 银行
sentence3 = "The weather is nice today."  # 纯天气



# 3. 把两个句子转成向量（384维）
vec1 = model.encode(sentence1)
vec2 = model.encode(sentence2)
vec3 = model.encode(sentence3)

# 4. 计算这两个句子的相似度
sim = cosine_similarity([vec1], [vec2])[0][0]
sim_weather = cosine_similarity([vec1], [vec3])[0][0]

print(f"句子1: {sentence1}")
print(f"句子2: {sentence2}")
print(f"两个句子的语义相似度: {sim:.4f}")
print(f"‘河岸bank句’ vs ‘天气句’ 相似度: {sim_weather:.4f}")

# 5. 额外验证：取同一个词 "bank"，在不同句子里的向量是否不同？
# 注意：Sentence-Transformers 不能直接取单词向量，但我们可以通过句子看效果。
# 上面两句的相似度如果低于 0.5，就说明模型成功区分了两种含义。