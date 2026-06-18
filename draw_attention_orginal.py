import torch, matplotlib.pyplot as plt
from matplotlib import font_manager
from transformers import AutoTokenizer, AutoModel

def setup_chinese_font():
    for name in ("PingFang SC", "Songti SC", "Heiti SC", "Arial Unicode MS", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

name = "bert-base-chinese"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, output_attentions=True)

text = "我爱你"
inputs = tok(text, return_tensors="pt")
attn = model(**inputs).attentions            # 每层每头的注意力
tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

def draw_all_heads(layer, ncols=4):
    layer_attn = attn[layer]
    n_heads = layer_attn.shape[1]
    nrows = (n_heads + ncols - 1) // ncols
    layer_idx = layer if layer >= 0 else len(attn) + layer

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.reshape(-1)
    im = None
    for head, ax in enumerate(axes):
        if head >= n_heads:
            ax.axis("off")
            continue
        im = ax.imshow(layer_attn[0, head].detach().numpy(), cmap="viridis")
        ax.set_title(f"head {head}", fontsize=10)
        if head // ncols == nrows - 1:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        else:
            ax.set_xticks([])
        if head % ncols == 0:
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.set_yticks([])

    fig.suptitle(f"layer {layer_idx} — all {n_heads} heads", fontsize=14)
    fig.subplots_adjust(right=0.92, hspace=0.35)
    fig.colorbar(im, ax=axes[:n_heads], fraction=0.02, pad=0.02)
    path = f"attn_L{layer_idx}_all_heads.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {path}")

draw_all_heads(-1)   # 最后一层 12 个头

