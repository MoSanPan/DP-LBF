import numpy as np
from pybloom_live import BloomFilter
import random
import time
import sys

start = time.time()
seed = 42
np.random.seed(seed)
random.seed(seed)

# =========================
# 工具函数
# =========================
def calculate_accuracy(actual, predicted):
    return np.sum(np.array(actual) == np.array(predicted)) / len(actual)

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

# =========================
# 1. 模拟数据集
# =========================
num_samples = 10000
urls = [f"url_{i}" for i in range(num_samples)]
labels = np.random.choice([0, 1], size=num_samples)

# 预测分数
pred_scores = np.zeros(num_samples)
pred_scores[labels == 1] = np.random.uniform(0.6, 1.0, size=(labels == 1).sum())
pred_scores[labels == 0] = np.random.uniform(0.0, 0.4, size=(labels == 0).sum())

# 构建目标集合与非目标集合
S_target = [urls[i] for i in range(len(urls)) if labels[i] == 1]
N_notarget = [urls[i] for i in range(len(urls)) if labels[i] == 0]

print("目标集合大小:", len(S_target))
print("非目标集合大小:", len(N_notarget))

# =========================
# 2. 查询集合 Q（固定随机采样）
# =========================
num_query_each = 2000
random.seed(seed)
Q_positive = random.sample(S_target, min(num_query_each, len(S_target)))
Q_negative = random.sample(N_notarget, min(num_query_each, len(N_notarget)))
Q_urls = Q_positive + Q_negative
Q_labels = [1]*len(Q_positive) + [0]*len(Q_negative)
Q_scores = [pred_scores[urls.index(url)] for url in Q_urls]

# =========================
# 3. DP-LBF 参数
# =========================
candidate_taus = np.linspace(0.1, 0.9, 10)
# epsilon_list = [0.5, 1, 1.5, 2, 2.5]
epsilon_list = [0.2, 0.4, 0.6, 0.8, 1]
BF_error_rate = 0.01
num_repeat = 100
BF_budget = len(S_target) + 500  # BF容量控制

# =========================
# 4. DP-LBF 主逻辑
# =========================
avg_accs, avg_rmse = [], []
avg_bf_sizes, avg_query_times = [], []

for epsilon in epsilon_list:
    acc_list, rmse_list, bf_sizes, query_times = [], [], [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(trial)

        # ---- 4.1 构建训练集：全集正样本 + 部分负样本 ----
        train_neg_sample = random.sample(N_notarget, min(len(N_notarget), BF_budget - len(S_target)))
        train_urls = S_target + train_neg_sample
        train_scores = [pred_scores[urls.index(u)] for u in train_urls]
        train_labels = [1]*len(S_target) + [0]*len(train_neg_sample)

        # ---- 4.2 DP 阈值选择（Exponential Mechanism）----
        utilities = []
        for tau in candidate_taus:
            fn_mask = (np.array(train_scores) < tau) & (np.array(train_labels) == 1)
            bf_size = fn_mask.sum()
            fp_mask = (np.array(train_scores) >= tau) & (np.array(train_labels) == 0)
            fpr = fp_mask.sum() / max(1, (np.array(train_labels) == 0).sum())
            util = -fpr if bf_size <= BF_budget else -np.inf
            utilities.append(util)

        utilities = np.array(utilities)
        finite_mask = np.isfinite(utilities)

        if not finite_mask.any():
            best_index = np.argmin([ ((np.array(train_scores) >= tau) & (np.array(train_labels)==0)).sum()
                                     / max(1, (np.array(train_labels)==0).sum())
                                     for tau in candidate_taus ])
        else:
            utilities_shift = utilities[finite_mask] - utilities[finite_mask].min()
            scores = np.exp(epsilon * utilities_shift / 2.0)
            probs = scores / scores.sum()
            indices = np.arange(len(candidate_taus))[finite_mask]
            best_index = rng.choice(indices, p=probs)

        best_tau = candidate_taus[best_index]

        # ---- 4.3 构建 Bloom Filter（存漏判正样本）----
        fn_mask = (np.array(train_scores) < best_tau) & (np.array(train_labels) == 1)
        if fn_mask.sum() > 0:
            bf = BloomFilter(capacity=fn_mask.sum(), error_rate=BF_error_rate)
            for url in np.array(train_urls)[fn_mask]:
                bf.add(url)
            bf_size_output = sys.getsizeof(bf.bitarray) / 1024  # KB
        else:
            bf = None
            bf_size_output = 0

        # ---- 4.4 查询阶段 ----
        t0 = time.time()
        y_pred = []
        for i, score in enumerate(Q_scores):
            url = Q_urls[i]
            if score >= best_tau:
                y_pred.append(1)
            elif bf is not None and url in bf:
                y_pred.append(1)
            else:
                y_pred.append(0)
        t1 = time.time()
        avg_query_time = (t1 - t0) / len(Q_scores)

        y_pred = np.array(y_pred)
        y_true = np.array(Q_labels)

        acc_list.append(calculate_accuracy(y_true, y_pred))
        rmse_list.append(calculate_rmse(y_true, y_pred))
        bf_sizes.append(bf_size_output)
        query_times.append(avg_query_time)

    # 平均指标
    avg_accs.append(np.mean(acc_list))
    avg_rmse.append(np.mean(rmse_list))
    avg_bf_sizes.append(np.mean(bf_sizes))
    avg_query_times.append(np.mean(query_times))

    print(f"\n=== epsilon = {epsilon} ===")
    print(f"平均最优阈值 tau*: {best_tau:.3f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_sizes):.1f} KB")
    print(f"平均查询时间: {np.mean(query_times) * 1000:.4f} ms")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")




# =========================
# 5. 输出总结果
# =========================
print("\n=== 总结果 ===")
print("ε:", epsilon_list)
print("ACC:", [round(a, 4) for a in avg_accs])
print("RMSE:", [round(r, 4) for r in avg_rmse])
print("平均查询时间(ms):", [round(t*1000,4) for t in avg_query_times])
print("平均 Bloom Filter 大小(KB):", [round(s,1) for s in avg_bf_sizes])


end = time.time()

