import numpy as np
import pandas as pd
import lightgbm as lgb
from pybloom_live import BloomFilter
import random
import time
import math
import os

# =========================
# 固定随机性
# =========================
seed = 42
np.random.seed(seed)
random.seed(seed)

start = time.time()

def calculate_accuracy(actual, predicted):
    correct = np.sum(np.array(actual) == np.array(predicted))
    return correct / len(actual)

# =========================
# 0. 文件路径（如需改动请修改）
# =========================
DATA_PATH = "URL_Dataset1.csv"
QUERY_SAVE_PATH = "query_dataset.csv"

# =========================
# 1. 读取 URL 数据集
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}. 请确认路径。")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.replace(" ", "_")

if 'URL' not in data.columns or 'label' not in data.columns:
    raise KeyError("数据文件必须包含 'URL' 和 'label' 两列。")

urls = data['URL'].tolist()
labels = data['label'].to_numpy()
num_samples = len(data)

# 特征矩阵（训练用）
X = data.drop(columns=['URL', 'label'], errors="ignore")
y = labels

print(f"数据总量: {num_samples}, 特征维度: {X.shape[1]}")

# =========================
# 2. 构建查询集合 Q（原始方式）
#    — 从整个数据中抽取 10% 正样本 + 10% 负样本（按索引）
# =========================
num_query_each = math.floor(len(urls) * 0.10)
rng = np.random.RandomState(seed)

pos_idx = np.where(labels == 1)[0]
neg_idx = np.where(labels == 0)[0]

num_pos_sample = min(num_query_each, len(pos_idx))
num_neg_sample = min(num_query_each, len(neg_idx))

sampled_pos_idx = rng.choice(pos_idx, size=num_pos_sample, replace=False).tolist()
sampled_neg_idx = rng.choice(neg_idx, size=num_neg_sample, replace=False).tolist()

Q_indices = sampled_pos_idx + sampled_neg_idx  # 保持你原来的顺序：正样本在前
# 如果你希望打乱查询集顺序，可以使用 rng.shuffle(Q_indices)

Q_df = data.iloc[Q_indices].reset_index(drop=True)  # 包含 URL, label, 特征列
Q_features = Q_df.drop(columns=['URL', 'label'], errors="ignore")
Q_labels = Q_df['label'].to_numpy()
Q_urls = Q_df['URL'].to_numpy()

# 保存查询集（方便复现/检查）
Q_df.to_csv(QUERY_SAVE_PATH, index=False)
print(f"已保存查询集到: {QUERY_SAVE_PATH}")
print(f"查询集大小: {len(Q_indices)} (正: {num_pos_sample}, 负: {num_neg_sample})")

# =========================
# 3. 训练 LightGBM 模型（使用全量数据，和你原来一致）
# =========================
features = X.columns
model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    learning_rate=0.05,
    n_estimators=30,
    num_leaves=5,
    max_depth=6,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    lambda_l1=0.1,
    lambda_l2=0.1,
    min_split_gain=0.01,
    random_state=seed,
    n_jobs=-1
)
model.fit(X[features], y)

# =========================
# 4. 预测分数
# =========================
train_scores = model.predict_proba(X[features])[:, 1]
test_scores = model.predict_proba(Q_features[features])[:, 1]

print("训练集预测分数 shape:", train_scores.shape)
print("查询集预测分数 shape:", test_scores.shape)

# =========================
# 5. DP-LBF 参数（可按需调整）
# =========================
candidate_taus = np.linspace(0.1, 1, 10)
epsilon_list = [0.5, 1, 1.5, 2, 2.5]
BF_budget = 10000
num_repeat = 100

# =========================
# 6. 全集数据准备（与原逻辑一致）
# =========================
all_scores = np.array(train_scores)
all_labels = np.array(y)
all_urls = np.array(urls)

Q_scores = np.array(test_scores)
Q_urls = np.array(Q_urls)
Q_labels = np.array(Q_labels)

# sanity check
assert len(Q_scores) == len(Q_urls) == len(Q_labels), "查询集各向量长度不一致！"

# =========================
# 7. DP-LBF 主逻辑
# =========================
avg_accs, avg_rmse, avg_bf_sizes, avg_query_times = [], [], [], []

for epsilon in epsilon_list:
    acc_list, rmse_list, bf_size_list, query_time_list = [], [], [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(trial)

        # ---- 7.1 DP 阈值选择 (Exponential Mechanism) ----
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
            best_index = np.argmin([
                ((all_scores >= tau) & (all_labels == 0)).sum() / ((all_labels == 0).sum() or 1)
                for tau in candidate_taus
            ])
        else:
            utilities_shift = utilities[finite_mask] - utilities[finite_mask].min()
            scores = np.exp(epsilon * utilities_shift / 2.0)
            probs = scores / scores.sum()
            indices = np.arange(len(candidate_taus))[finite_mask]
            best_index = rng.choice(indices, p=probs)

        best_tau = candidate_taus[best_index]

        # ---- 7.2 构建 Bloom Filter ----
        fn_mask_all = (all_scores < best_tau) & (all_labels == 1)
        if fn_mask_all.sum() > 0:
            bf_capacity = int(fn_mask_all.sum() * 2)
            bf = BloomFilter(capacity=bf_capacity)
            for url in all_urls[fn_mask_all]:
                bf.add(url)
            bf_size_list.append(fn_mask_all.sum())
        else:
            bf = None
            bf_size_list.append(0)

        # ---- 7.3 测试阶段 ----
        start_query = time.time()  # 统计查询时间
        y_pred = []
        for i in range(len(Q_scores)):
            score = Q_scores[i]
            url = Q_urls[i]
            if score >= best_tau:
                y_pred.append(1)
            elif bf is not None and url in bf:
                y_pred.append(1)
            else:
                y_pred.append(0)
        end_query = time.time()
        query_time_list.append(end_query - start_query)

        y_pred = np.array(y_pred)
        y_true = np.array(Q_labels)

        acc_list.append(calculate_accuracy(y_true, y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse_list.append(rmse)

    # 每个 epsilon 的平均结果
    avg_accs.append(np.mean(acc_list))
    avg_rmse.append(np.mean(rmse_list))
    avg_bf_sizes.append(np.mean(bf_size_list))
    avg_query_times.append(np.mean(query_time_list))

    print(f"\n=== epsilon = {epsilon} ===")
    print(f"选择的平均最优阈值 tau*: {best_tau:.3f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_size_list):.2f}")
    print(f"平均查询时间: {np.mean(query_time_list):.6f} 秒")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")

# =========================
# 8. 输出总结果
# =========================
print("\n=== 总结果 ===")
print("ACC:", [round(a, 4) for a in avg_accs])
print("RMSE:", [round(f, 4) for f in avg_rmse])
print("平均 Bloom Filter 大小:", [round(f, 2) for f in avg_bf_sizes])
print("平均查询时间 (秒):", [round(f, 6) for f in avg_query_times])

