import json
with open(r"C:\Users\xiaoy\Downloads\vocab (2).json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
print(len(vocab))