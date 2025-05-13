import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 数据加载
train_df = pd.read_csv("图书管理系统XGBoost/train_dataset.csv")
test_users = pd.read_csv("图书管理系统XGBoost/test_dataset.csv")["user_id"].unique()

# 特征工程 ================================================
# 基础统计特征
user_stats = train_df.groupby('user_id')['item_id'].agg([
    'count', 
    'nunique',
    lambda x: len(set(x)) / len(x)  # 多样性
]).reset_index()
user_stats.columns = ['user_id', 'user_activity', 'unique_items', 'diversity']

item_stats = train_df.groupby('item_id')['user_id'].agg([
    'count',
    'nunique'
]).reset_index()
item_stats.columns = ['item_id', 'item_popularity', 'unique_users']

# 合并特征
merged = train_df.merge(user_stats, on='user_id').merge(item_stats, on='item_id')

# 时间衰减特征（假设数据按时间排序）
merged['timestamp_rank'] = merged.groupby('user_id').cumcount(ascending=False)
merged['time_decay'] = 1 / (1 + np.log1p(merged['timestamp_rank']))

# 交叉特征（修复关键点）
merged['user_item_ratio'] = merged['user_activity'] / (merged['item_popularity'] + 1e-6)  # 防止除零
merged['pop_diversity'] = merged['item_popularity'] * merged['diversity']

# 负采样 ==================================================
positive_samples = merged[['user_id', 'item_id', 'user_activity', 'unique_items', 
                          'diversity', 'item_popularity', 'unique_users', 
                          'time_decay', 'user_item_ratio', 'pop_diversity']].copy()
positive_samples['label'] = 1

# 生成负样本
negative_samples = positive_samples.sample(frac=0.2, random_state=42)
negative_samples['item_id'] = np.random.choice(merged['item_id'].unique(), 
                                             len(negative_samples), 
                                             replace=True)
negative_samples['label'] = 0

# 合并数据集
full_data = pd.concat([positive_samples, negative_samples])

# 特征重组（修复关键点：确保保留所有特征）
features = full_data.merge(
    user_stats, on='user_id', suffixes=('', '_y')
).merge(
    item_stats, on='item_id', suffixes=('', '_y')
)

# 填充缺失值
features = features[['user_activity', 'unique_items', 'diversity',
                    'item_popularity', 'unique_users', 'time_decay', 
                    'user_item_ratio', 'pop_diversity']].fillna(0)

# 划分训练集
X = features
y = full_data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练 =================================================
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc'
)

# 使用 evals_result 和早停逻辑
evals_result = {}
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 手动实现早停逻辑
if 'auc' in evals_result and len(evals_result['validation_0']['auc']) > 20:
    best_iteration = max(range(len(evals_result['validation_0']['auc'])), 
                         key=lambda i: evals_result['validation_0']['auc'][i])
    if best_iteration < len(evals_result['validation_0']['auc']) - 20:
        model.set_params(n_estimators=best_iteration + 1)

# 生成推荐 ================================================
# 获取所有item特征
all_items = item_stats['item_id'].unique()
item_features = item_stats.merge(
    pd.DataFrame({'item_id': all_items}),
    on='item_id',
    how='right'
).fillna(0)

recommendations = []

for user_id in test_users:
    try:
        # 获取用户特征
        user_feat = user_stats[user_stats['user_id'] == user_id].iloc[0]
    except IndexError:
        # 冷启动处理：推荐热门商品
        top_items = item_stats.nlargest(10, 'item_popularity')['item_id'].tolist()
        recommendations.extend([{'user_id': user_id, 'item_id': item} for item in top_items])
        continue
    
    # 生成用户-物品特征矩阵
    user_item_matrix = pd.DataFrame({
        'user_id': user_id,
        'item_id': all_items
    }).merge(user_stats, on='user_id').merge(
        item_features, on='item_id'
    )
    
    # 计算交叉特征
    user_item_matrix['user_item_ratio'] = user_feat['user_activity'] / (user_item_matrix['item_popularity'] + 1e-6)
    user_item_matrix['pop_diversity'] = user_item_matrix['item_popularity'] * user_feat['diversity']
    
    # 添加 time_decay 特征
    user_item_matrix['timestamp_rank'] = user_item_matrix.groupby('user_id').cumcount(ascending=False)
    user_item_matrix['time_decay'] = 1 / (1 + np.log1p(user_item_matrix['timestamp_rank']))

    # 填充缺失值
    user_item_matrix = user_item_matrix.fillna(0)
    
    # 预测
    X_pred = user_item_matrix[X.columns]
    preds = model.predict_proba(X_pred)[:, 1]

    # 将预测结果添加到 user_item_matrix
    user_item_matrix['prediction'] = preds

    # 排除已交互物品
    interacted_items = train_df[train_df['user_id'] == user_id]['item_id'].unique()
    mask = ~user_item_matrix['item_id'].isin(interacted_items)

    # 取Top10
    top_items = user_item_matrix[mask].nlargest(10, 'prediction')['item_id'].tolist()

    # 结果存储
    recommendations.extend([{'user_id': user_id, 'item_id': item} for item in top_items])

# 保存结果
pd.DataFrame(recommendations).to_csv("图书管理系统XGBoost/submission - 副本.csv", index=False)