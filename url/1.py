import numpy as np
import pandas as pd
import lightgbm as lgb
from pybloom_live import BloomFilter
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
num_samples = len(data)

# 特征
X = data.drop(columns=['URL', 'label'], errors="ignore")
y = labels

# 构建目标集合与非目标集合
S_target = [urls[i] for i in range(len(urls)) if labels[i] == 1]
N_notarget = [urls[i] for i in range(len(urls)) if labels[i] == 0]
print("目标集合大小:", len(S_target))
print("非目标集合大小:", len(N_notarget))

# =========================
# 2. 查询集合 Q（固定随机采样）
# =========================
num_query_each = math.floor(len(urls) * 0.10)
rng = np.random.RandomState(seed)
Q_positive = rng.choice(S_target, size=min(num_query_each, len(S_target)), replace=False).tolist()
Q_negative = rng.choice(N_notarget, size=min(num_query_each, len(N_notarget)), replace=False).tolist()
Q_urls = Q_positive + Q_negative
Q_labels = np.array([1]*len(Q_positive) + [0]*len(Q_negative))

# =========================
# 2. 划分训练/验证/测试集
# =========================

num_train = int(num_samples)


X_train_urls = X.iloc[:num_train]
y_train = y[:num_train]
urls_train = urls[:num_train]


# =========================
# 3. 训练 LightGBM 模型
# =========================
features = X_train_urls.columns
model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    learning_rate=0.05,        # 🚀 小步长提升泛化能力
    n_estimators=30,           # 🚀 更多树补偿学习率降低
    num_leaves=5,              # 与 max_depth 配合，控制复杂度
    max_depth=6,                # 略放宽捕捉非线性特征
    min_child_samples=20,       # 允许更多分裂
    subsample=0.9,              # 保留更多样本
    colsample_bytree=0.9,       # 使用更多特征
    lambda_l1=0.1,              # L1 正则，增强稀疏性
    lambda_l2=0.1,              # L2 正则，防止过拟合
    min_split_gain=0.01,        # 分裂增益阈值
    random_state=seed,
    n_jobs=-1                   # 使用全部CPU核
)
model.fit(X_train_urls[features], y_train)

# =========================
# 4. 预测分数
# =========================
train_scores = model.predict_proba(X_train_urls[features])[:, 1]

test_scores = model.predict_proba(Q_urls[features])[:, 1]

print(test_scores)

# =========================
# 2. DP-LBF 参数
# =========================
candidate_taus = np.linspace(0.1, 1, 10)
epsilon_list =  [0.5, 1, 1.5, 2, 2.5]
BF_budget = 10000
num_repeat = 100

# 全集数据
all_scores = np.concatenate([train_scores])
all_labels = np.concatenate([y_train])
all_urls = np.concatenate([X_train_urls])

# 查询集
Q = np.concatenate([ test_scores])
Q_urls = np.concatenate([Q_urls])
Q_labels = np.concatenate([Q_labels])


# =========================
# 3. DP-LBF 主逻辑
# =========================
avg_accs, avg_rmse = [], []

for epsilon in epsilon_list:
    acc_list, rmse_list = [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(trial)  # 每次重复独立可复现

        # ---- 3.1 DP 阈值选择 (Exponential Mechanism) ----
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

        # ---- 3.2 构建 Bloom Filter (存漏判正样本) ----
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

        # ---- 3.3 测试阶段 ----
        y_pred = []
        for i, score in enumerate(Q):
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
# 4. 输出总结果
# =========================
print("\n=== 总结果 ===")
print("ACC:", [round(a, 4) for a in avg_accs])
print("RMSE:  ", [round(f, 4) for f in avg_rmse])

end = time.time()
print(f"\n运行时间: {end - start:.4f} 秒")
