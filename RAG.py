import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 1. 加载动态向量模型（你刚才装的）
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 准备一个小型知识库（模拟公司文档）
knowledge_base = [
    "公司 WiFi 名称为 'Office-5G'，密码为 'Secure@2025'，如遇连接问题请联系 IT 部门分机 1234。",
    "员工入职后需在 HR 系统完成个人信息填写，包括银行卡号、紧急联系人、社保基数。新员工培训在每月第一个周一举行。",
    "报销流程：登录财务系统上传发票，填写事由，经部门经理审批后，财务将在 5 个工作日内打款。单笔超过 5000 元需 CFO 二次审批。",
    "公司邮箱使用 Outlook，服务器地址为 mail.company.com，账户名为工号@company.com，初始密码为身份证后六位。",
    "年假制度：入职满一年享 5 天年假，满三年享 10 天，满五年享 15 天。请假需提前 3 天在 OA 系统提交申请。"
]
# 3. 将知识库向量化（离线索引）
kb_embeddings = model.encode(knowledge_base)

# 4. 用户提问
query = "我想请假，年假怎么计算？"
# 或
query = "怎么连公司 WiFi？密码是多少？"
# 或
query = "报销发票怎么上传？要多久能到账？"

# 5. 将问题转成向量
query_emb = model.encode([query])

# 6. 计算问题与所有知识库文本的余弦相似度
similarities = cosine_similarity(query_emb, kb_embeddings)[0]

# 7. 找出最相似的 Top-2 段落
top_k_indices = np.argsort(similarities)[-2:][::-1]  # 从大到小排序取前2

print(f"用户问题: {query}\n")
print("检索到的相关资料（按相关度排序）：")
for idx in top_k_indices:
    print(f"- 相似度 {similarities[idx]:.4f}：{knowledge_base[idx]}")

# 8. （模拟生成环节）如果有大模型API，就把下面这段拼进Prompt
print("\n--- 模拟LLM回答 ---")
context = "\n".join([knowledge_base[idx] for idx in top_k_indices])
fake_prompt = f"基于以下资料回答问题：\n{context}\n\n问题：{query}\n答案："
print(f"（实际项目中将此 Prompt 发给 GPT/Ollama）\n{fake_prompt}")