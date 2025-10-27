import numpy as np
from pybloom_live import BloomFilter
import time
import random
import sys
import math
import pandas as pd

start = time.time()
seed = 42

# =========================
# 工具函数
# =========================
def calculate_accuracy(actual, predicted):
    correct = np.sum(np.array(actual) == np.array(predicted))
    return correct / len(actual)

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

# =========================
# 1. 读取 URL 数据集
# =========================
data = pd.read_csv("URL_Dataset1.csv") # 可根据需要调整行数
data.columns = data.columns.str.replace(" ", "_")
label_col = "label"
urls = data['URL'].tolist()
labels = data[label_col].to_numpy()
num_samples = len(urls)


# 构建目标集合与非目标集合
S_target = [urls[i] for i in range(len(urls)) if labels[i] == 1]  # 插入集合 S
N_notarget = [urls[i] for i in range(len(urls)) if labels[i] == 0]  # 非目标集合

print("目标集合大小:", len(S_target))
print("非目标集合大小:", len(N_notarget))

# 固定随机种子以确保每次取样一致
random.seed(42)
num_query_each = math.floor(len(urls) * 0.10)
print(num_query_each)
Q_positive = random.sample(S_target, min(num_query_each, len(S_target)))
Q_negative = random.sample(N_notarget, min(num_query_each, len(N_notarget)))
Q = Q_positive + Q_negative
actual_values = [1]*len(Q_positive) + [0]*len(Q_negative)

# =========================
# 2. Mangat-Filter 实验
# =========================
# epsilon_list = [0.5, 1, 1.5, 2, 2.5]
epsilon_list = [0.2, 0.4, 0.6, 0.8, 1]
num_repeat = 100

avg_acc_all = []
avg_rmse_all = []
avg_query_time_all = []
avg_bf_mem_all = []

num_query_each = 2000
rng_fixed = np.random.RandomState(seed)  # 查询集合固定随机源

for epsilon in epsilon_list:
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    acc_list, rmse_list, query_time_list, bf_mem_list = [], [], [], []

    for _ in range(num_repeat):

        # Mangat Filter 生成 S'
        S_prime = list(S_target)
        for x in N_notarget:
            if np.random.rand() < (1 - p):
                S_prime.append(x)


        # 构建 Bloom Filter（假阳性率 0.01）
        bf = BloomFilter(capacity=len(S_prime), error_rate=0.01)
        for x in S_prime:
            bf.add(x)

        # 记录 Bloom Filter 内存占用（bitarray 对象）
        bf_mem_list.append(sys.getsizeof(bf.bitarray) / 1024)  # KB



        # 查询并测量平均查询时间
        t0 = time.perf_counter()
        predicted_values = [1 if url in bf else 0 for url in Q]
        t1 = time.perf_counter()
        avg_query_time = (t1 - t0) / len(Q)  # 秒/条
        query_time_list.append(avg_query_time)

        # 计算指标
        acc_list.append(calculate_accuracy(actual_values, predicted_values))
        rmse_list.append(calculate_rmse(actual_values, predicted_values))

    # 平均指标
    avg_acc_all.append(np.mean(acc_list))
    avg_rmse_all.append(np.mean(rmse_list))
    avg_query_time_all.append(np.mean(query_time_list))
    avg_bf_mem_all.append(np.mean(bf_mem_list))

    print(f"\n=== ε = {epsilon}, p = {p:.4f} ===")
    print(f"平均准确率: {np.mean(acc_list):.4f}")
    print(f"平均 RMSE: {np.mean(rmse_list):.4f}")
    print(f"平均 Bloom Filter 大小: {np.mean(bf_mem_list):.2f} KB")
    print(f"平均每条查询时间: {np.mean(query_time_list)*1000:.4f} ms")

# =========================
# 3. 输出总结果
# =========================
print("\n=== MF-总结果 ===")
print("ε:", epsilon_list)
print("ACC:", [round(a, 4) for a in avg_acc_all])
print("RMSE:", [round(r, 4) for r in avg_rmse_all])
print("AvgQueryTime(ms/条):", [round(t*1000, 4) for t in avg_query_time_all])
print("BloomFilterSize(KB):", [round(s, 2) for s in avg_bf_mem_all])

end = time.time()

