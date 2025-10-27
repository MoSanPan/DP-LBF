import numpy as np
import pandas as pd
import lightgbm as lgb
from pybloom_live import BloomFilter
import random
import time
import math
import os
import sys

# =========================
# 固定随机性
# =========================
seed = 42
np.random.seed(seed)
random.seed(seed)

start = time.time()

def calculate_accuracy(actual, predicted):
    return np.mean(np.array(actual) == np.array(predicted))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))

# =========================
# 文件路径
# =========================
DATA_PATH = "credit_1_balanced.csv"
QUERY_SAVE_PATH = "query_dataset.csv"

# =========================
# 读取数据
# =========================
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.replace(" ", "_")

urls = data['URL'].tolist()
labels = data['label'].to_numpy()
num_samples = len(data)
X = data.drop(columns=['URL', 'label'], errors="ignore")
y = labels

# =========================
# 构建查询集（10%正 + 10%负）
# =========================
num_query_each = math.floor(len(urls) * 0.10)
rng = np.random.RandomState(seed)

pos_idx = np.where(labels == 1)[0]
neg_idx = np.where(labels == 0)[0]

sampled_pos_idx = rng.choice(pos_idx, size=min(num_query_each, len(pos_idx)), replace=False)
sampled_neg_idx = rng.choice(neg_idx, size=min(num_query_each, len(neg_idx)), replace=False)

Q_indices = np.concatenate([sampled_pos_idx, sampled_neg_idx])
Q_df = data.iloc[Q_indices].reset_index(drop=True)
Q_features = Q_df.drop(columns=['URL', 'label'], errors="ignore")
Q_labels = Q_df['label'].to_numpy()
Q_urls = Q_df['URL'].to_numpy()
Q_df.to_csv(QUERY_SAVE_PATH, index=False)

# =========================
# 训练 LightGBM
# =========================
features = X.columns
model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    learning_rate=0.05,  # 🚀 小步长提升泛化能力
    n_estimators=500,  # 🚀 更多树补偿学习率降低
    num_leaves=31,  # 与 max_depth 配合，控制复杂度
    max_depth=10,  # 略放宽捕捉非线性特征
    min_child_samples=20,  # 允许更多分裂
    subsample=0.9,  # 保留更多样本
    colsample_bytree=0.9,  # 使用更多特征
    lambda_l1=0.1,  # L1 正则，增强稀疏性
    lambda_l2=0.1,  # L2 正则，防止过拟合
    min_split_gain=0.01,  # 分裂增益阈值
    random_state=seed,
    n_jobs=-1  # 使用全部CPU核
)
model.fit(X[features], y)

train_scores = model.predict_proba(X[features])[:, 1]
test_scores = model.predict_proba(Q_features[features])[:, 1]

# =========================
# DP-LBF 参数
# =========================
candidate_taus = np.linspace(0.1, 1, 10)
epsilon_list = [0.2, 0.4, 0.6, 0.8, 1]
BF_budget = len(urls)
num_repeat = 100  # 演示可减小次数

# =========================
# 指标存储
# =========================
avg_acc_all, avg_rmse_all, avg_bf_mem_all, avg_query_time_all = [], [], [], []

all_scores = np.array(train_scores)
all_labels = np.array(y)
all_urls = np.array(urls)

Q_scores = np.array(test_scores)
Q_urls = np.array(Q_urls)
Q_labels = np.array(Q_labels)

for epsilon in epsilon_list:
    acc_list, rmse_list, bf_mem_list, query_time_list = [], [], [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(trial)

        # ---- 1. DP 阈值选择 ----
        utilities = []
        for tau in candidate_taus:
            fn_mask_all = (all_scores < tau) & (all_labels == 1)
            bf_size = fn_mask_all.sum()
            fp_mask_all = (all_scores >= tau) & (all_labels == 0)
            neg_count = (all_labels == 0).sum()
            fpr = (fp_mask_all.sum() / neg_count) if neg_count > 0 else 0.0
            util = -fpr if bf_size <= BF_budget else -np.inf
            utilities.append(util)

        utilities = np.array(utilities)
        finite_mask = np.isfinite(utilities)
        if not finite_mask.any():
            best_index = np.argmin([((all_scores >= tau) & (all_labels == 0)).sum() / ((all_labels == 0).sum() or 1)
                                    for tau in candidate_taus])
        else:
            utilities_shift = utilities[finite_mask] - utilities[finite_mask].min()
            scores = np.exp(epsilon * utilities_shift / 2.0)
            probs = scores / scores.sum()
            indices = np.arange(len(candidate_taus))[finite_mask]
            best_index = rng.choice(indices, p=probs)

        best_tau = candidate_taus[best_index]

        # ---- 2. 构建 Bloom Filter ----
        fn_mask_all = (all_scores < best_tau) & (all_labels == 1)
        if fn_mask_all.sum() > 0:
            bf_capacity = int(fn_mask_all.sum() * 2)
            bf = BloomFilter(capacity=bf_capacity)
            for url in all_urls[fn_mask_all]:
                bf.add(url)
                # 用 sys.getsizeof 统计 bitarray 大小，换算成 KB
            bf_mem_list.append(sys.getsizeof(bf.bitarray) / 1024)
        else:
            bf = None
            bf_mem_list.append(0)

        # ---- 3. 查询并计算时间 ----
        t0 = time.perf_counter()
        y_pred = []
        for i in range(len(Q_urls)):
            url = Q_urls[i]
            score = Q_scores[i]
            if score >= best_tau:
                y_pred.append(1)
            elif bf is not None and url in bf:
                y_pred.append(1)
            else:
                y_pred.append(0)
        t1 = time.perf_counter()
        query_time_list.append((t1 - t0) / len(Q_urls))  # 秒/条

        # ---- 4. 计算指标 ----
        y_pred = np.array(y_pred)
        acc_list.append(calculate_accuracy(Q_labels, y_pred))
        rmse_list.append(calculate_rmse(Q_labels, y_pred))

    # 平均指标
    avg_acc_all.append(np.mean(acc_list))
    avg_rmse_all.append(np.mean(rmse_list))
    avg_bf_mem_all.append(np.mean(bf_mem_list))
    avg_query_time_all.append(np.mean(query_time_list))

    print(f"\n=== ε = {epsilon} ===")
    print(f"选择的平均最优阈值 tau*: {best_tau:.3f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_mem_list):.2f} KB")
    print(f"平均每条查询时间: {np.mean(query_time_list)*1000:.6f} ms")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")

# =========================
# 总结果
# =========================
print("\n=== 总结果 ===")
print("ε:", epsilon_list)
print("ACC:", [round(a, 4) for a in avg_acc_all])
print("RMSE:", [round(r, 4) for r in avg_rmse_all])
print("BloomFilterSize(KB):", [round(s, 2) for s in avg_bf_mem_all])
print("AvgQueryTime(ms/条):", [round(t*1000, 6) for t in avg_query_time_all])

end = time.time()
print(f"\n运行时间: {end - start:.2f} 秒")
