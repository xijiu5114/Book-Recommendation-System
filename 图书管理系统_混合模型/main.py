import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from collections import defaultdict

# 配置参数
MODEL_WEIGHTS = {
    'als': 0.4,
    'itemcf': 0.3,
    'bpr': 0.2,
    'xgb': 0.1
}
TOP_N = 10

def load_model_results(model_names):
    """加载各模型结果文件"""
    results = {}
    for name in model_names:
        try:
            df = pd.read_csv(f"{name}_submission.csv")
            # 统一列名
            df.columns = ['user_id', 'item_id']  
            results[name] = df
            print(f"Loaded {len(df)} recommendations from {name}")
        except FileNotFoundError:
            print(f"Warning: {name}_submission.csv not found")
    return results

def hybrid_recommendation(model_results, weights):
    """混合推荐核心逻辑"""
    # 初始化得分字典
    hybrid_scores = defaultdict(lambda: defaultdict(float))
    
    # 合并各模型结果
    for model_name, df in model_results.items():
        weight = weights.get(model_name, 0)
        if weight <= 0:
            continue
        
        # 为每个推荐位置赋予衰减权重
        for idx, row in df.iterrows():
            user = row['user_id']
            item = row['item_id']
            pos = idx % TOP_N  # 获取在推荐列表中的位置(0-9)
            decay = 1.0 / (pos + 2)  # 位置衰减因子
            hybrid_scores[user][item] += weight * decay
    
    # 生成最终推荐
    final_rec = []
    train_df = pd.read_csv("G:/机器学习课设-图书系统推荐/图书管理系统_混合模型/train_dataset.csv")
    global_popular = train_df['item_id'].value_counts().index.tolist()
    
    for user in pd.read_csv("G:/机器学习课设-图书系统推荐/图书管理系统_混合模型/train_dataset.csv")['user_id'].unique():
        user_scores = hybrid_scores.get(user, {})
        
        # 处理冷启动用户
        if not user_scores:
            top_items = global_popular[:TOP_N]
        else:
            # 排序并去重
            sorted_items = sorted(user_scores.items(), 
                               key=lambda x: (-x[1], x[0]))  # 确保稳定性
            
            seen = set()
            top_items = []
            for item, _ in sorted_items:
                if item not in seen:
                    seen.add(item)
                    top_items.append(item)
                if len(top_items) >= TOP_N:
                    break
            
            # 补充不足的推荐
            if len(top_items) < TOP_N:
                need = TOP_N - len(top_items)
                for item in global_popular:
                    if item not in seen:
                        top_items.append(item)
                        need -= 1
                    if need == 0:
                        break
        
        # 记录结果
        for item in top_items[:TOP_N]:
            final_rec.append({'user_id': user, 'item_id': item})
    
    return pd.DataFrame(final_rec, columns=['user_id', 'item_id'])

def validate_submission(df):
    """验证文件格式"""
    try:
        assert list(df.columns) == ['user_id', 'item_id'], "列名不匹配"
        counts = df['user_id'].value_counts()
        assert counts.nunique() == 1, "推荐数量不一致"
        assert counts.iloc[0] == TOP_N, f"推荐数量应为{TOP_N}"
        print("验证通过！")
        print(f"总用户数: {df['user_id'].nunique()}")
        print(f"总推荐数: {len(df)}")
    except AssertionError as e:
        print(f"验证失败: {str(e)}")

if __name__ == "__main__":
    # 加载各模型结果
    model_results = load_model_results(MODEL_WEIGHTS.keys())
    
    # 生成混合推荐
    final_df = hybrid_recommendation(model_results, MODEL_WEIGHTS)
    
    # 保存结果（确保包含表头）
    final_df.to_csv("G:/机器学习课设-图书系统推荐/图书管理系统_混合模型/hybrid_submission.csv", index=False, header=True)
    print("结果已保存")
    
    # 验证结果
    submission = pd.read_csv("G:/机器学习课设-图书系统推荐/图书管理系统_混合模型/hybrid_submission.csv")
    validate_submission(submission)