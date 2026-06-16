from openai import OpenAI

client = OpenAI(
    api_key="sk-API",  # 替换为从 DeepSeek 平台获取的 key
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "请用一句话解释什么是人工智能"}]
)

print(response.choices[0].message.content)