import tiktoken

enc = tiktoken.get_encoding("o200k_base")


def show_token_steps(text: str) -> None:
    token_ids = enc.encode(text)
    pieces = [enc.decode([token_id]) for token_id in token_ids]

    print(f"\n输入: {text!r}")
    print(f"共分成 {len(token_ids)} 个 token\n")

    for step, (token_id, piece) in enumerate(zip(token_ids, pieces), start=1):
        print(f"步骤 {step}: id={token_id} -> {piece!r}")

    joined = " + ".join(repr(piece) for piece in pieces)
    print(f"\n拼起来: {joined} = {text!r}")

    if len(text) > 1 and " " not in text:
        char_pieces = []
        for ch in text:
            char_pieces.extend(enc.decode([token_id]) for token_id in enc.encode(ch))
        if len(char_pieces) != len(pieces):
            char_joined = " + ".join(repr(piece) for piece in char_pieces)
            print(f"\n对比：先按单字符切分是 {len(char_pieces)} 段")
            print(f"字符级: {char_joined}")
            print(f"BPE 合并后: {joined}")


if __name__ == "__main__":
    text = input("请输入内容：")
    show_token_steps(text)