from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)


def show_token_steps(text: str) -> None:
    token_ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

    print(f"\n分词器: Qwen/Qwen2.5-0.5B")
    print(f"输入: {text!r}")
    print(f"共 {len(token_ids)} 个 token\n")

    for step, (token_id, piece) in enumerate(zip(token_ids, pieces), start=1):
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
        print(f"步骤 {step}: id={token_id} -> {piece!r} (token: {token_str!r})")

    joined = " + ".join(repr(piece) for piece in pieces)
    print(f"\n拼起来: {joined} = {text!r}")
    print(f"token 个数: {len(token_ids)}")


if __name__ == "__main__":
    text = input("请输入内容：")
    show_token_steps(text)
