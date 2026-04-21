import csv
import os
import random
import sys
import time

import numpy as np
from bitarray import bitarray
from pybloom_live import BloomFilter


# 算法说明：
# 1. 先用分类分数 pred_scores 为每个 URL 提供一个“直接判为正样本”的依据。
# 2. 对候选阈值 tau 使用指数机制，得到 epsilon_1-DP 的私有阈值选择过程。
#    这一步保护的是训练数据分布以及由训练数据反映出的模型敏感信息。
# 3. 对被阈值漏判的正样本构建 Bloom Filter，再对 Bloom Filter 的 bitarray 做随机响应扰动，
#    得到 epsilon_2-DP 的成员发布过程，保护具体元素是否出现在集合中。
# 4. 查询时先看 score 是否超过 tau；若没有超过，再查询扰动后的 Bloom Filter。
# 5. 整体使用基本组合：epsilon_total = epsilon_1 + epsilon_2。

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


def build_randomized_response_probs(epsilon_bit):
    exp_eps = np.exp(epsilon_bit)
    keep_prob = exp_eps / (exp_eps + 1.0)
    flip_prob = 1.0 / (exp_eps + 1.0)
    return keep_prob, flip_prob


def privatize_bloom_filter_bits(bloom_filter, epsilon_2, rng):
    """
    对 Bloom Filter 的位数组做随机响应。

    一个元素最多影响 k 个 bit，其中 k = num_slices。
    因此把 epsilon_2 平均分配到每个可能受影响的 bit 上，
    每个 bit 使用 epsilon_bit = epsilon_2 / k 的随机响应，
    按基本组合保证整个 Bloom Filter 发布满足 epsilon_2-DP。
    """
    privatized_bf = bloom_filter.copy()
    num_hashes = 10
    epsilon_bit = epsilon_2 / num_hashes
    keep_prob, _ = build_randomized_response_probs(epsilon_bit)

    original_bits = np.asarray(bloom_filter.bitarray.tolist(), dtype=np.int8)
    random_draws = rng.rand(len(original_bits))
    noisy_bits = np.where(random_draws < keep_prob, original_bits, 1 - original_bits)

    privatized_bf.bitarray = bitarray(noisy_bits.tolist())
    return privatized_bf, epsilon_bit


def select_private_threshold(train_scores, train_labels, candidate_taus, bf_budget, epsilon_1, rng):
    """
    使用指数机制选择阈值，保护训练数据分布与模型敏感信息。

    这里把不同 tau 的效用定义为负的假阳性率（FPR），
    同时要求漏判正样本数量不能超过 Bloom Filter 可承载预算。
    效用越大，被选中的概率越高；epsilon_1 越大，越接近最优阈值；
    epsilon_1 越小，随机性越强，隐私保护越强。
    """
    train_scores = np.asarray(train_scores)
    train_labels = np.asarray(train_labels)
    negative_mask = train_labels == 0
    negative_count = max(1, negative_mask.sum())

    utilities = []
    for tau in candidate_taus:
        fn_mask = (train_scores < tau) & (train_labels == 1)
        bf_size = int(fn_mask.sum())
        fp_mask = (train_scores >= tau) & negative_mask
        fpr = fp_mask.sum() / negative_count
        utility = -fpr if bf_size <= bf_budget else -np.inf
        utilities.append(utility)

    utilities = np.asarray(utilities, dtype=float)
    finite_mask = np.isfinite(utilities)

    if not finite_mask.any():
        fallback_costs = []
        for tau in candidate_taus:
            fp_mask = (train_scores >= tau) & negative_mask
            fallback_costs.append(fp_mask.sum() / negative_count)
        best_index = int(np.argmin(fallback_costs))
        return best_index, candidate_taus[best_index], utilities

    utilities_shift = utilities[finite_mask] - utilities[finite_mask].max()
    scores = np.exp((epsilon_1 * utilities_shift) / 2.0)
    probs = scores / scores.sum()
    indices = np.arange(len(candidate_taus))[finite_mask]
    best_index = int(rng.choice(indices, p=probs))
    return best_index, candidate_taus[best_index], utilities


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
# 3. 双层 DP-LBF 参数
# =========================
candidate_taus = np.linspace(0.1, 0.9, 10)
epsilon_total_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
epsilon_1_ratio = 0.1
BF_error_rate = 0.01
num_repeat = 100
BF_budget = len(S_target) + 500


# =========================
# 4. 双层 DP-LBF 主逻辑
# 阈值 / partition -> epsilon_1-DP
# Bloom Filter bits -> epsilon_2-DP
# 总隐私预算 epsilon_total = epsilon_1 + epsilon_2
# =========================
# 这里默认把总预算均分成两部分：
# epsilon_1_ratio = 0.5 表示一半给阈值选择，一半给 Bloom Filter 位扰动。
# 如果后续要研究不同分配策略，只需要调整 epsilon_1_ratio 即可。
avg_accs, avg_rmse = [], []
avg_bf_sizes, avg_query_times = [], []
avg_taus = []
epsilon_1_list, epsilon_2_list = [], []

for epsilon_total in epsilon_total_list:
    epsilon_1 = epsilon_total * epsilon_1_ratio
    epsilon_2 = epsilon_total - epsilon_1
    epsilon_1_list.append(epsilon_1)
    epsilon_2_list.append(epsilon_2)

    acc_list, rmse_list, bf_sizes, query_times, tau_list = [], [], [], [], []

    for trial in range(num_repeat):
        rng = np.random.RandomState(seed + trial)
        sample_rng = random.Random(seed + trial)

        # ---- 4.1 构建训练集：全集正样本 + 部分负样本 ----
        # 正样本全部进入训练集，负样本只抽取一部分用于阈值选择。
        # 这样做的目的，是让阈值选择与 Bloom Filter 容量预算保持一致。
        neg_budget = max(0, BF_budget - len(S_target))
        train_neg_sample = sample_rng.sample(N_notarget, min(len(N_notarget), neg_budget))
        train_urls = S_target + train_neg_sample
        train_scores = np.array([url_to_score[u] for u in train_urls])
        train_labels = np.array([1] * len(S_target) + [0] * len(train_neg_sample))

        # ---- 4.2 epsilon_1-DP 阈值选择（Exponential Mechanism）----
        # 这一层对应“partition / threshold privacy”：
        # 不直接暴露哪个阈值在训练数据上最好，而是按效用概率化选择。
        _, best_tau, _ = select_private_threshold(
            train_scores=train_scores,
            train_labels=train_labels,
            candidate_taus=candidate_taus,
            bf_budget=BF_budget,
            epsilon_1=epsilon_1,
            rng=rng,
        )
        tau_list.append(best_tau)

        # ---- 4.3 构建 Bloom Filter，并对 bitarray 做 epsilon_2-DP 扰动 ----
        # 阈值以下的正样本会被模型漏判，因此补充存入 Bloom Filter。
        # 为了保护“某个具体 URL 是否属于漏判正样本集合”，
        # 我们不直接发布原始 Bloom Filter，而是对每个 bit 做随机响应。
        fn_mask = (train_scores < best_tau) & (train_labels == 1)
        if fn_mask.sum() > 0:
            bf = BloomFilter(capacity=int(fn_mask.sum()), error_rate=BF_error_rate)
            for url in np.array(train_urls)[fn_mask]:
                bf.add(url)
            noisy_bf, epsilon_bit = privatize_bloom_filter_bits(bf, epsilon_2, rng)
            bf_size_output = sys.getsizeof(noisy_bf.bitarray) / 1024
        else:
            noisy_bf = None
            epsilon_bit = 0.0
            bf_size_output = 0.0

        # ---- 4.4 查询阶段 ----
        # 查询规则：
        # 1. 若 score >= tau，直接判为正样本；
        # 2. 否则查询扰动后的 Bloom Filter，命中则补判为正样本；
        # 3. 两者都不满足则判为负样本。
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

    print(f"\n=== epsilon_total = {epsilon_total:.2f} ===")
    print(f"epsilon_1 (threshold / partition): {epsilon_1:.3f}")
    print(f"epsilon_2 (Bloom Filter bits): {epsilon_2:.3f}")
    print(f"平均最优阈值 tau*: {avg_tau:.3f}")
    print(f"每个受影响 bit 的隐私预算 epsilon_bit: {epsilon_bit:.6f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_sizes):.1f} KB")
    print(f"平均查询时间: {np.mean(query_times) * 1000:.4f} ms")
    print(f"平均准确率: {np.mean(acc_list):.4f}, 平均 RMSE: {np.mean(rmse_list):.4f}")


# =========================
# 5. 输出总结果
# =========================
print("\n=== 总结果 ===")
print("epsilon_total:", epsilon_total_list)
print("epsilon_1:", [round(eps, 4) for eps in epsilon_1_list])
print("epsilon_2:", [round(eps, 4) for eps in epsilon_2_list])
print("tau*:", [round(tau, 4) for tau in avg_taus])
print("ACC:", [round(a, 4) for a in avg_accs])
print("RMSE:", [round(r, 4) for r in avg_rmse])
print("平均查询时间(ms):", [round(t * 1000, 4) for t in avg_query_times])
print("平均 Bloom Filter 大小(KB):", [round(s, 1) for s in avg_bf_sizes])


end = time.time()
print(f"总运行时间: {end - start:.2f} s")


# # =========================
# # 6. 保存结果
# # =========================
results = []

for eps_total, eps_1, eps_2, tau, acc, rmse, qt, bf in zip(
    epsilon_total_list,
    epsilon_1_list,
    epsilon_2_list,
    avg_taus,
    avg_accs,
    avg_rmse,
    avg_query_times,
    avg_bf_sizes,
):
    results.append(
        {
            "epsilon_total": eps_total,
            "epsilon_1": round(eps_1, 4),
            "epsilon_2": round(eps_2, 4),
            "avg_tau": round(tau, 4),
            "ACC": round(acc, 4),
            "RMSE": round(rmse, 4),
            "QueryTime(ms)": round(qt * 1000, 4),
            "BloomFilterSize(KB)": round(bf, 1),
        }
    )

print(results)