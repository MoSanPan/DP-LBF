import math
import random
import numpy as np
import time
import sys
import pandas as pd
from pybloom_live import BloomFilter

# =========================
# 1. 模拟数据集（URL + label）
# =========================
start = time.time()
seed = 42
np.random.seed(seed)
random.seed(seed)

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

print("插入集合 S 大小:", len(S_target))
print("非目标集合大小:", len(N_notarget))

# 固定随机种子以确保每次取样一致
random.seed(42)
num_query_each = math.floor(len(urls) * 0.10)
print(num_query_each)
Q_positive = random.sample(S_target, min(num_query_each, len(S_target)))
Q_negative = random.sample(N_notarget, min(num_query_each, len(N_notarget)))
Q = Q_positive + Q_negative
actual_values = [1]*len(Q_positive) + [0]*len(Q_negative)

print("查询集合 Q 大小:", len(Q))

# =========================
# 2. 定义指标计算函数
# =========================
def calculate_accuracy(actual, predicted):
    correct = np.sum(np.array(actual) == np.array(predicted))
    return correct / len(actual)

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

# =========================
# 3. rappor 实验
# =========================
# epsilon_list = [0.5, 1, 1.5, 2, 2.5]
epsilon_list = [0.2, 0.4, 0.6, 0.8, 1]
avg_acc_all = []
avg_rmse_all = []
avg_time_all = []
avg_space_all = []
alpha = 0.5  # 隐私预算分配参数
num_repeat = 100  # 重复次数（可以调大）

for epsilon in epsilon_list:
    epsilon_PRR = epsilon * alpha
    epsilon_IRR = epsilon - epsilon_PRR
    print(f"\n当前 epsilon: {epsilon}")
    acc_list = []
    rmse_list = []
    query_time_list = []
    space_list = []

    for _ in range(num_repeat):
        # 构建 Bloom Filter（指定假阳性率 0.01）
        bf = BloomFilter(capacity=len(S_target), error_rate=0.01)

        # 插入数据 S
        for element in S_target:
            bf.add(element)

        # 计算 Bloom Filter 空间大小（近似）
        space_bytes = sys.getsizeof(bf.bitarray)  # 内存中bitarray对象大小
        space_list.append(space_bytes / 1024)     # 转换为 KB

        # PRR 随机翻转
        prob_keep_prr = np.exp(epsilon_PRR) / (np.exp(epsilon_PRR) + 1)
        for idx in range(len(bf.bitarray)):
            if np.random.rand() > prob_keep_prr:
                bf.bitarray[idx] = not bf.bitarray[idx]

        # IRR 随机翻转
        prob_keep_lrr = np.exp(epsilon_IRR) / (np.exp(epsilon_IRR) + 1)
        for idx in range(len(bf.bitarray)):
            if np.random.rand() > prob_keep_lrr:
                bf.bitarray[idx] = not bf.bitarray[idx]

        # 查询并计算平均查询时间
        t0 = time.time()
        predicted_values = [1 if url in bf else 0 for url in Q]
        t1 = time.time()
        avg_query_time = (t1 - t0) / len(Q)
        query_time_list.append(avg_query_time)

        # 计算 ACC & RMSE
        acc = calculate_accuracy(actual_values, predicted_values)
        rmse = calculate_rmse(actual_values, predicted_values)
        acc_list.append(acc)
        rmse_list.append(rmse)

    # 平均结果
    avg_acc = np.mean(acc_list)
    avg_rmse = np.mean(rmse_list)
    avg_time = np.mean(query_time_list)
    avg_space = np.mean(space_list)

    avg_acc_all.append(avg_acc)
    avg_rmse_all.append(avg_rmse)
    avg_time_all.append(avg_time)
    avg_space_all.append(avg_space)

    print(f"epsilon = {epsilon}: 平均 ACC = {avg_acc:.4f}, 平均 RMSE = {avg_rmse:.4f}, "
          f"平均查询时间 = {avg_time*1000:.4f} ms, 平均空间 = {avg_space:.2f} KB")

# =========================
# 4. 输出总结果
# =========================
print("\n=== rappor 实验总结果 ===")
print("ε:", epsilon_list)
print("ACC:", [round(a, 4) for a in avg_acc_all])
print("RMSE:", [round(r, 4) for r in avg_rmse_all])
print("AvgQueryTime(ms):", [round(t*1000, 4) for t in avg_time_all])
print("Space(KB):", [round(s, 2) for s in avg_space_all])

end = time.time()

