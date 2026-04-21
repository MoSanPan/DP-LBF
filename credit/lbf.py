import numpy as np
import pandas as pd
import lightgbm as lgb
from pybloom_live import BloomFilter
import random
import time
import math
import sys
# 不再需要 bitarray，因为不进行位翻转

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
DATA_PATH = "credit.csv"
QUERY_SAVE_PATH = "query_dataset.csv"

# =========================
# 读取数据
# =========================
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.replace(" ", "_")

urls = data['ID'].tolist()
labels = data['label'].to_numpy()
num_samples = len(data)
X = data.drop(columns=['ID', 'label'], errors="ignore")
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
Q_features = Q_df.drop(columns=['ID', 'label'], errors="ignore")
Q_labels = Q_df['label'].to_numpy()
Q_urls = Q_df['ID'].to_numpy()
Q_df.to_csv(QUERY_SAVE_PATH, index=False)

# =========================
# 训练 LightGBM
# =========================
features = X.columns
model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    learning_rate=0.05,
    n_estimators=15,
    num_leaves=5,
    max_depth=6,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.9,
    lambda_l1=0.1,
    lambda_l2=0.1,
    min_split_gain=0.01,
    random_state=seed,
    n_jobs=-1
)
model.fit(X[features], y)

train_scores = model.predict_proba(X[features])[:, 1]
test_scores = model.predict_proba(Q_features[features])[:, 1]

# =========================
# LBF 参数（不加噪，但保留原框架的循环变量名）
# =========================
candidate_taus = np.linspace(0.1, 1, 10)
epsilon_total_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # 仅用于循环次数，实际不使用
BF_error_rate = 0.01
BF_capacity_limit = 100000   # 容量上限（可调整）
num_repeat = 100

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

# =========================
# 主循环（保留 ε 循环和重复实验，但内部不加噪）
# =========================
for epsilon_total in epsilon_total_list:   # 实际不使用 ε
    # 修改：不再计算 ε₁ 和 ε₂，因为它们未被使用
    acc_list, rmse_list, bf_mem_list, query_time_list = [], [], [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(trial)

        # ---- 1. 阈值选择（确定性，无 DP）----
        # 直接最小化 FPR（无容量约束）
        best_tau = None
        best_fpr = float('inf')
        neg_count = (all_labels == 0).sum()
        for tau in candidate_taus:
            fn_count = ((all_scores < tau) & (all_labels == 1)).sum()
            fp_count = ((all_scores >= tau) & (all_labels == 0)).sum()
            fpr = fp_count / neg_count if neg_count > 0 else 0.0
            if fpr < best_fpr:
                best_fpr = fpr
                best_tau = tau

        # ---- 2. 构建备份 Bloom Filter（无位扰动）----
        fn_mask = (all_scores < best_tau) & (all_labels == 1)
        fn_count = fn_mask.sum()
        if fn_count > 0:
            bf_capacity = int(fn_count)   # 不加噪时直接用实际数量
            bf = BloomFilter(capacity=bf_capacity, error_rate=BF_error_rate)

            indices = np.where(fn_mask)[0]
            for idx in indices:
                url = all_urls[idx]
                bf.add(url)

            # 修改：不再进行位扰动，直接使用原始位数组
            bf_mem_list.append(sys.getsizeof(bf.bitarray) / 1024)  # KB
        else:
            bf = None
            bf_mem_list.append(0)

        # ---- 3. 查询并计时 ----
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
        query_time_list.append((t1 - t0) / len(Q_urls))

        # ---- 4. 计算指标 ----
        y_pred = np.array(y_pred)
        acc_list.append(calculate_accuracy(Q_labels, y_pred))
        rmse_list.append(calculate_rmse(Q_labels, y_pred))

    # 保存当前 ε 下的平均指标（由于不加噪，每个 ε 的结果应几乎相同）
    avg_acc_all.append(np.mean(acc_list))
    avg_rmse_all.append(np.mean(rmse_list))
    avg_bf_mem_all.append(np.mean(bf_mem_list))
    avg_query_time_all.append(np.mean(query_time_list))

    print(f"\n=== ε_total = {epsilon_total:.2f} (LBF 不加噪) ===")
    print(f"选择的平均最优阈值 tau*: {best_tau:.3f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_mem_list):.2f} KB")
    print(f"平均每条查询时间: {np.mean(query_time_list)*1000:.6f} ms")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")

# =========================
# 输出总结果
# =========================
print("\n=== 总结果 ===")
print(len(bf))
print("ε_total:", epsilon_total_list)
print("ACC:", [round(a, 4) for a in avg_acc_all])
print("RMSE:", [round(r, 4) for r in avg_rmse_all])
print("BloomFilterSize(KB):", [round(s, 2) for s in avg_bf_mem_all])
print("AvgQueryTime(ms/条):", [round(t*1000, 6) for t in avg_query_time_all])

end = time.time()
print(f"\n总运行时间: {end - start:.2f} 秒")