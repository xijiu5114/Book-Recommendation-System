import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 加载训练数据
train_df = pd.read_csv("图书推荐系统Item-CF/train_dataset.csv")
test_users = pd.read_csv("图书推荐系统Item-CF/test_dataset.csv")["user_id"].unique()

# 创建映射字典
user_ids = train_df["user_id"].unique()
item_ids = train_df["item_id"].unique()
user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}

# 构建稀疏矩阵
row = train_df["user_id"].map(user_map)
col = train_df["item_id"].map(item_map)
data = np.ones(len(train_df))
train_matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(item_ids)))

# 计算物品相似度矩阵（使用调整余弦相似度）
item_sim = cosine_similarity(train_matrix.T, dense_output=False)

# 计算物品流行度用于冷启动
item_popularity = np.asarray(train_matrix.sum(axis=0)).ravel()

# 生成推荐结果
recommendations = []
for user_id in test_users:
    if user_id in user_map:
        user_idx = user_map[user_id]
        # 获取用户交互过的物品索引
        interacted_items = train_matrix[user_idx].indices
        
        if len(interacted_items) == 0:
            # 无历史交互用户处理
            top_items = np.argsort(item_popularity)[-10:]
        else:
            # 聚合相似物品得分
            scores = item_sim[interacted_items].sum(axis=0).A1
            # 排除已交互物品
            scores[interacted_items] = -np.inf
            # 获取Top10
            top_items = np.argpartition(scores, -10)[-10:]
    else:
        # 冷启动处理：推荐热门商品
        top_items = np.argsort(item_popularity)[-10:]
    
    # 转换为原始item_id并存储
    recommendations.extend([{
        "user_id": user_id,
        "item_id": item_ids[item_idx]
    } for item_idx in top_items])

# 保存结果
pd.DataFrame(recommendations).to_csv("图书推荐系统Item-CF/itemcf_submission.csv", index=False)