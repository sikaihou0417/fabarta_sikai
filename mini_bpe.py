# interactive_bpe.py
from collections import Counter

# ---------- BPE 训练（使用示例语料）----------
def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe(corpus, num_merges=20):
    word_freq = Counter(corpus)
    vocab = {' '.join(word): freq for word, freq in word_freq.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)
    return merges

# 示例语料（可自行替换）
corpus = [
    "bu","bu","bu","bu","bu",
    "gao","gao","gao","gao","gao","gao","yao","yao","yao","yao","yao",
    "hao","hao","hao","hao","hao","hao", 
    "le","le","le","le","le","le","le",
    "ba","ba","ba","ba","ba","ba","ba","ba","ba","ba",
    "ni","ni","ni","ni","ni","ni","ni","ni",
    
]
merges = train_bpe(corpus, num_merges=30)  # 训练合并规则

# ---------- 分词函数 ----------
def apply_bpe(text, merges):
    """
    对输入的文本应用 BPE 合并规则，返回 token 列表。
    先按空格拆成单词，对每个单词切分为字符，然后按 merges 顺序合并。
    """
    tokens = []
    for word in text.split():
        # 初始为字符列表
        chars = list(word)
        # 对每个合并对，尝试合并
        for pair in merges:
            bigram = ''.join(pair)
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i+1] == pair[1]:
                    # 合并成新的 token
                    chars[i] = bigram
                    del chars[i+1]
                    # 合并后可能形成新的相邻对，但这里我们不立即重新检查（简化）
                    # 为了正确性，应该重新扫描，但为了演示简单，我们只做一次合并
                i += 1
        tokens.extend(chars)
    return tokens

# ---------- 交互界面 ----------
print("BPE 分词器已就绪（基于示例语料训练）")
print("输入文本（拼音），按 Enter 分段，输入 'exit' 退出。\n")

while True:
    user_input = input(">> ")
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input.strip():
        continue
    # 分词
    tokens = apply_bpe(user_input, merges)
    # 输出结果：用 | 分隔显示
    print("分词结果:", " | ".join(tokens))
    print("Token 数量:", len(tokens))
    print()