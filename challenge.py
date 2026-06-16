import ollama
from collections import Counter

MODEL = "qwen3:4b"          # 改成你下载的模型名
QUESTION = "请用一句话解释什么是人工智能。"

print("=" * 50)
print("挑战：同一问题连问 20 次（默认 temperature）")
print("=" * 50)

answers = []
for i in range(20):
    resp = ollama.generate(model=MODEL, prompt=QUESTION)
    ans = resp['response'].strip()
    answers.append(ans)
    print(f"{i+1:2d}. {ans[:60]}{'...' if len(ans)>60 else ''}")

counter = Counter(answers)
print("\n📊 统计结果：")
print(f"共 {len(counter)} 种不同回答\n")
for ans, cnt in counter.most_common():
    print(f"【{cnt} 次】{ans}")

# temperature = 0 对比
print("\n" + "=" * 50)
print("temperature = 0 时，连问 5 次")
print("=" * 50)

temp0_answers = []
for i in range(5):
    resp = ollama.generate(
        model=MODEL,
        prompt=QUESTION,
        options={"temperature": 0}
    )
    ans = resp['response'].strip()
    temp0_answers.append(ans)
    print(f"{i+1}. {ans}")
