import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

# 1. 加载模型（自动下载）
model = api.load("glove-wiki-gigaword-50")

# 2. 类比
print("=== 类比 ===")
print("king - man + woman →", model.most_similar(positive=['king','woman'], negative=['man'], topn=3))
print("paris - france + italy →", model.most_similar(positive=['paris','italy'], negative=['france'], topn=3))
print("woman - girl + boy →", model.most_similar(positive=['woman','boy'], negative=['girl'], topn=3))
print("big - bigger + smaller →", model.most_similar(positive=['big', 'smaller'], negative=['bigger'], topn=3))
print("quick - quickly + slowly →", model.most_similar(positive=['quick', 'slowly'], negative=['quickly'], topn=3))

# 3. 相似度
print("\n=== 相似度 ===")
print("cat-dog:", model.similarity('cat','dog'))
print("cat-car:", model.similarity('cat','car'))
print("good-bad:", model.similarity('good','bad'))
print("apple-huawei",model.similarity('apple','huawei'))
print("apple-orange",model.similarity('apple','orange'))


# 5. PCA 聚类
words = ['apple','huawei','xiaomi','orange','banana','cat','dog','lion','car','bus','airplane','tomato','bat','bank','lender','shore']
# 只保留模型里存在的词
present = [w for w in words if w in model]
vectors = np.array([model[w] for w in present])
coords = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(10,8))
for w, (x,y) in zip(present, coords):
    plt.scatter(x,y)
    plt.annotate(w, (x,y), fontsize=14)
plt.title('Static Embedding PCA (GloVe)')
plt.grid()
plt.show()