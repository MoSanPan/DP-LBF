import numpy as np
import pandas as pd
import lightgbm as lgb
from pybloom_live import BloomFilter
from sklearn.model_selection import train_test_split
import random
import time
import math

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
# 1. 读取 URL 数据集
# =========================
data = pd.read_csv("URL_Dataset1.csv")
data.columns = data.columns.str.replace(" ", "_")

urls = data['URL'].tolist()
labels = data['label'].to_numpy()
X = data.drop(columns=['URL', 'label'], errors="ignore")
y = labels

print(f"数据总量: {len(data)}")

# =========================
# 2. 划分训练 / 查询集（20% 查询）
# =========================
X_train, X_query, y_train, y_query, urls_train, urls_query = train_test_split(
    X, y, urls, test_size=0.2, stratify=y, random_state=seed
)

print(f"训练集大小: {len(X_train)}, 查询集大小: {len(X_query)}")

# =========================
# 3. 训练 LightGBM 模型
# =========================
features = X_train.columns
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
model.fit(X_train[features], y_train)

# =========================
# 4. 预测分数
# =========================
train_scores = model.predict_proba(X_train[features])[:, 1]
query_scores = model.predict_proba(X_query[features])[:, 1]

print("训练集预测分数 shape:", train_scores.shape)
print("查询集预测分数 shape:", query_scores.shape)

# =========================
# 5. DP-LBF 参数
# =========================
candidate_taus = np.linspace(0.1, 1, 10)
epsilon_list = [0.5, 1, 1.5, 2, 2.5]
BF_budget = 10000
num_repeat = 50  # 可改为 100，但运行较慢

# =========================
# 6. 数据整理
# =========================
all_scores = np.array(train_scores)
all_labels = np.array(y_train)
all_urls = np.array(urls_train)

Q_scores = np.array(query_scores)
Q_labels = np.array(y_query)
Q_urls = np.array(urls_query)

# =========================
# 7. DP-LBF 主逻辑
# =========================
avg_accs, avg_rmse = [], []

for epsilon in epsilon_list:
    acc_list, rmse_list = [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(trial)

        # ---- 7.1 DP 阈值选择 (Exponential Mechanism) ----
        utilities = []
        for tau in candidate_taus:
            fn_mask_all = (all_scores < tau) & (all_labels == 1)
            bf_size = fn_mask_all.sum()
            fp_mask_all = (all_scores >= tau) & (all_labels == 0)
            fpr = fp_mask_all.sum() / (all_labels == 0).sum()
            util = -fpr if bf_size <= BF_budget else -np.inf
            utilities.append(util)

        utilities = np.array(utilities)
        finite_mask = np.isfinite(utilities)

        if not finite_mask.any():
            best_index = np.argmin([
                ((all_scores >= tau) & (all_labels == 0)).sum() / (all_labels == 0).sum()
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
            bf_capacity = fn_mask_all.sum() * 2
            bf = BloomFilter(capacity=bf_capacity)
            for url in all_urls[fn_mask_all]:
                bf.add(url)
            bf_size_output = fn_mask_all.sum()
        else:
            bf = None
            bf_size_output = 0

        # ---- 7.3 测试阶段 ----
        y_pred = []
        for i, score in enumerate(Q_scores):
            url = Q_urls[i]
            if score >= best_tau:
                y_pred.append(1)
            elif bf is not None and url in bf:
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_pred = np.array(y_pred)
        y_true = np.array(Q_labels)

        acc_list.append(calculate_accuracy(y_true, y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse_list.append(rmse)

    avg_accs.append(np.mean(acc_list))
    avg_rmse.append(np.mean(rmse_list))

    print(f"\n=== epsilon = {epsilon} ===")
    print(f"平均最优阈值 tau*: {best_tau:.3f}")
    print(f"平均 Bloom Filter 大小: {bf_size_output}")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")

# =========================
# 8. 输出总结果
# =========================
print("\n=== 总结果 ===")
print("ACC:", [round(a, 4) for a in avg_accs])
print("RMSE:", [round(f, 4) for f in avg_rmse])

end = time.time()
print(f"\n运行时间: {end - start:.4f} 秒")
