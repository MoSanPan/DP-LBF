import csv
import os
import random
import sys
import time

import numpy as np
from pybloom_live import BloomFilter

# 算法说明：
# 1. 先用分类分数 pred_scores 为每个 URL 提供一个“直接判为正样本”的依据。
# 2. 确定性选择 FPR 最小的阈值 tau（无容量约束，或可保留容量约束，此处采用无约束最小化 FPR）。
# 3. 对被阈值漏判的正样本构建 Bloom Filter，**不进行任何位扰动**。
# 4. 查询时先看 score 是否超过 tau；若没有超过，再查询原始 Bloom Filter。
# 5. 无隐私保护，所有操作均为确定性过程（重复实验仅用于统计平均，无随机性）。

start = time.time()
seed = 42
np.random.seed(seed)
random.seed(seed)


# =========================
# 工具函数
# =========================
def calculate_accuracy(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.mean(actual == predicted)


def calculate_rmse(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def select_threshold_no_privacy(train_scores, train_labels, candidate_taus, bf_budget=None):
    """
    无隐私保护的确定性阈值选择：
    在所有候选 tau 中，选择假阳性率 (FPR) 最小的那个。
    若提供 bf_budget 且不为 None，则要求假阴性数量不超过预算，否则跳过该 tau。
    若所有 tau 均不满足预算，则忽略预算约束直接选 FPR 最小的。
    """
    train_scores = np.asarray(train_scores)
    train_labels = np.asarray(train_labels)
    negative_mask = train_labels == 0
    negative_count = max(1, negative_mask.sum())

    best_tau = None
    best_fpr = float('inf')

    for tau in candidate_taus:
        fn_count = ((train_scores < tau) & (train_labels == 1)).sum()
        fp_count = ((train_scores >= tau) & negative_mask).sum()
        fpr = fp_count / negative_count

        if bf_budget is not None and fn_count > bf_budget:
            continue   # 超过容量预算，跳过此 tau

        if fpr < best_fpr:
            best_fpr = fpr
            best_tau = tau

    if best_tau is None:   # 所有 tau 都超预算，则忽略预算约束
        for tau in candidate_taus:
            fp_count = ((train_scores >= tau) & negative_mask).sum()
            fpr = fp_count / negative_count
            if fpr < best_fpr:
                best_fpr = fpr
                best_tau = tau

    return best_tau


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
url_to_score = dict(zip(urls, pred_scores))

# 构建目标集合与非目标集合
S_target = [url for url, label in zip(urls, labels) if label == 1]
N_notarget = [url for url, label in zip(urls, labels) if label == 0]

print("目标集合大小:", len(S_target))
print("非目标集合大小:", len(N_notarget))


# =========================
# 2. 查询集合 Q（固定随机采样）
# =========================
num_query_each = 2000
query_rng = random.Random(seed)
Q_positive = query_rng.sample(S_target, min(num_query_each, len(S_target)))
Q_negative = query_rng.sample(N_notarget, min(num_query_each, len(N_notarget)))
Q_urls = Q_positive + Q_negative
Q_labels = [1] * len(Q_positive) + [0] * len(Q_negative)
Q_scores = [url_to_score[url] for url in Q_urls]


# =========================
# 3. 无噪 LBF 参数
# =========================
candidate_taus = np.linspace(0.1, 0.9, 10)
epsilon_total_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # 仅用于循环次数，无实际意义
BF_error_rate = 0.01
num_repeat = 100
BF_budget = len(S_target) + 500   # 可保留容量约束，此处为与 DP 版本一致而保留


# =========================
# 4. 无噪 LBF 主逻辑
# =========================
avg_accs, avg_rmse = [], []
avg_bf_sizes, avg_query_times = [], []
avg_taus = []

for epsilon_total in epsilon_total_list:   # epsilon_total 无实际作用，仅作循环计数
    acc_list, rmse_list, bf_sizes, query_times, tau_list = [], [], [], [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(seed + trial)      # 虽无用但保留
        sample_rng = random.Random(seed + trial)       # 虽无用但保留

        # ---- 4.1 构建训练集：全集正样本 + 部分负样本（与 DP 版本一致）----
        neg_budget = max(0, BF_budget - len(S_target))
        train_neg_sample = sample_rng.sample(N_notarget, min(len(N_notarget), neg_budget))
        train_urls = S_target + train_neg_sample
        train_scores = np.array([url_to_score[u] for u in train_urls])
        train_labels = np.array([1] * len(S_target) + [0] * len(train_neg_sample))

        # ---- 4.2 确定性阈值选择（无隐私保护）----
        best_tau = select_threshold_no_privacy(
            train_scores=train_scores,
            train_labels=train_labels,
            candidate_taus=candidate_taus,
            bf_budget=BF_budget
        )
        tau_list.append(best_tau)

        # ---- 4.3 构建 Bloom Filter（无扰动）----
        fn_mask = (train_scores < best_tau) & (train_labels == 1)
        if fn_mask.sum() > 0:
            bf = BloomFilter(capacity=int(fn_mask.sum()), error_rate=BF_error_rate)
            for url in np.array(train_urls)[fn_mask]:
                bf.add(url)
            # 无扰动，直接使用原始位数组
            bf_size_output = sys.getsizeof(bf.bitarray) / 1024
            noisy_bf = bf   # 保持变量名一致
        else:
            noisy_bf = None
            bf_size_output = 0.0

        # ---- 4.4 查询阶段 ----
        t0 = time.time()
        y_pred = []
        for url, score in zip(Q_urls, Q_scores):
            if score >= best_tau:
                y_pred.append(1)
            elif noisy_bf is not None and url in noisy_bf:
                y_pred.append(1)
            else:
                y_pred.append(0)
        t1 = time.time()
        avg_query_time = (t1 - t0) / len(Q_scores)

        y_true = np.asarray(Q_labels)
        y_pred = np.asarray(y_pred)

        acc_list.append(calculate_accuracy(y_true, y_pred))
        rmse_list.append(calculate_rmse(y_true, y_pred))
        bf_sizes.append(bf_size_output)
        query_times.append(avg_query_time)

    avg_tau = float(np.mean(tau_list))
    avg_taus.append(avg_tau)
    avg_accs.append(float(np.mean(acc_list)))
    avg_rmse.append(float(np.mean(rmse_list)))
    avg_bf_sizes.append(float(np.mean(bf_sizes)))
    avg_query_times.append(float(np.mean(query_times)))

    print(f"\n=== epsilon_total = {epsilon_total:.2f} (No Privacy) ===")
    print(f"平均最优阈值 tau*: {avg_tau:.3f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_sizes):.1f} KB")
    print(f"平均查询时间: {np.mean(query_times) * 1000:.4f} ms")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")


# =========================
# 5. 输出总结果
# =========================
print("\n=== 总结果 ===")
print("epsilon_total:", epsilon_total_list)
print("tau*:", [round(tau, 4) for tau in avg_taus])
print("ACC:", [round(a, 4) for a in avg_accs])
print("RMSE:", [round(r, 4) for r in avg_rmse])
print("平均查询时间(ms):", [round(t * 1000, 4) for t in avg_query_times])
print("平均 Bloom Filter 大小(KB):", [round(s, 1) for s in avg_bf_sizes])


end = time.time()
print(f"总运行时间: {end - start:.2f} s")

