import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset

# 数据加载
train_df = pd.read_csv("图书管理系统BPR/train_dataset.csv")
test_users = pd.read_csv("图书管理系统BPR/test_dataset.csv")["user_id"].unique()

# 创建映射字典
user_ids = train_df["user_id"].unique()
item_ids = train_df["item_id"].unique()
user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}

# 构建交互矩阵
interactions = csr_matrix(
    (np.ones(len(train_df)),
     (train_df["user_id"].map(user_map), train_df["item_id"].map(item_map))),
    shape=(len(user_ids), len(item_ids))
)

# 初始化BPR模型
model = LightFM(
    no_components=64,       # 潜在因子维度
    loss='bpr',             # 使用BPR损失函数
    learning_rate=0.05,     # 学习率
    item_alpha=1e-6,        # 物品L2正则化
    random_state=42
)

# 模型训练
model.fit(
    interactions, 
    epochs=30,             # 训练轮数
    num_threads=4,         # 并行线程
    verbose=True
)

# 计算物品流行度（用于冷启动）
item_popularity = np.asarray(interactions.sum(axis=0)).ravel()

# 生成推荐结果
recommendations = []
for user_id in test_users:
    if user_id in user_map:
        # 处理已知用户
        user_idx = user_map[user_id]
        
        # 获取用户未交互的物品
        interacted_items = interactions[user_idx].indices
        all_items = np.arange(interactions.shape[1])
        non_interacted = np.setdiff1d(all_items, interacted_items)
        
        # 预测分数
        scores = model.predict(user_idx, non_interacted)
        
        # 取Top10
        top_items = non_interacted[np.argsort(-scores)[:10]]
    else:
        # 处理冷启动用户：推荐热门商品
        top_items = np.argsort(-item_popularity)[:10]
    
    # 转换原始ID并存储
    recommendations.extend([{
        "user_id": user_id,
        "item_id": item_ids[item_idx]
    } for item_idx in top_items])

# 保存结果
pd.DataFrame(recommendations).to_csv("bpr_submission.csv", index=False)