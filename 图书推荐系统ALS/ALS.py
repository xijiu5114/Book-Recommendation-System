import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix

# 读取训练数据
train_df = pd.read_csv("图书推荐系统ALS/train_dataset.csv")
# 读取测试用户列表
test_users = pd.read_csv("图书推荐系统ALS/test_dataset.csv")["user_id"].unique()

# 创建用户和物品的映射
user_ids = train_df["user_id"].unique()
item_ids = train_df["item_id"].unique()
user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}

# 构建稀疏矩阵
row = train_df["user_id"].map(user_map).values
col = train_df["item_id"].map(item_map).values
data = np.ones(len(train_df))
matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(item_ids)))

# 初始化ALS模型
model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1)
model.fit(matrix)

# 生成推荐结果
recommendations = []
for user_id in test_users:
    if user_id in user_map:
        user_idx = user_map[user_id]
        # 获取已交互的物品索引
        interacted = matrix[user_idx].indices
        # 预测所有物品分数
        scores = model.user_factors[user_idx] @ model.item_factors.T
        # 排除已交互的
        scores[interacted] = -np.inf
        # 取Top-10
        top_items = np.argpartition(scores, -10)[-10:]
        # 转换为原始item_id
        for item_idx in top_items:
            item_id = item_ids[item_idx]
            recommendations.append({"user_id": user_id, "item_id": item_id})

# 保存结果
pd.DataFrame(recommendations).to_csv("图书推荐系统ALS/als_submission.csv", index=False)