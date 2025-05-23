ALS（Alternating Least Squares，交替最小二乘法）是一种基于矩阵分解的协同过滤算法，核心思想是通过分解用户-物品交互矩阵为两个低维隐因子矩阵（用户因子矩阵和物品因子矩阵），从而预测用户对未知物品的偏好。
（1）原理基础：将用户-物品交互矩阵R（可能包含显式评分或隐式行为）分解为两个低维隐因子矩阵：用户矩阵U（m×k）和物品矩阵V（n×k），使得R≈U*（V^T)。其中，k表示隐含特征维度，通常远小于用户和物品数量，以此捕捉用户与物品的潜在关联。
（2）核心步骤：
初始化：随机生成用户矩阵U或物品矩阵V，隐因子维度k需预先指定（如通过交叉验证选择）。
交替优化：固定物品矩阵V，更新用户矩阵U。对每个用户u，利用其评分过的物品对应的V子矩阵，通过最小二乘法求解Uu。其中Ku​是用户Uu的评分物品集合，Ru是评分向量；固定用户矩阵U，更新物品矩阵V同理。
![image](https://github.com/user-attachments/assets/a510ece7-b5f6-475b-9687-c3c9bba4c355)

迭代至收敛：重复上述交替优化步骤，直到总体重构误差（如RMSE）趋于稳定或达到预设的最大迭代次数。通常迭代10~20次即可收敛。
正则化与参数调优：正则化系数λ用于平衡拟合误差与模型复杂度，防止过拟合；隐式反馈中需调节置信度参数α，放大高频行为的权重。
