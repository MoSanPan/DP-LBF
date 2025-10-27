import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter

# =========================
# 自定义颜色
# =========================
customColor1 = [0.9290, 0.6946, 0.1250]
customColor2 = [0.4660, 0.6746, 0.1886]
customColor3 = [1, 0, 0]
customColor4 = [0, 0.4470, 0.7410]
customColor5 = [0.4940, 0.1840, 0.5560]
customColor6 = [0.3010, 0.7450, 0.9330]

# =========================
# 参数
# =========================
CD = 1.5
x = np.array([0.2, 0.4, 0.6, 0.8, 1])
x_dense = np.linspace(x.min(), x.max(), 9)  # 插值点

label_fontsize = 13
tick_fontsize = 11
legend_fontsize = 10

linestyles = ['--', '--', '--', '-', '--', '--']
markers = ['o', 's', '^', 'v', 'D', 'p']
colors = [customColor1, customColor6, customColor2, customColor3, customColor4, customColor5]

# =========================
# 数据 (URL dataset)
# =========================
url_data = {
    'RAPPOR':[0.7071, 0.7069, 0.7067, 0.7062, 0.7058],
    'EBF-LDP':[0.6713, 0.6343, 0.5963, 0.5584, 0.5207],
    'DPBloomFilter':[0.7045, 0.7003, 0.6933, 0.6834, 0.6698],
    'DP-LBF':[0.0874, 0.074, 0.0653, 0.0653, 0.0591],
    'Mangat_filters':[0.4785, 0.4524, 0.426, 0.3993, 0.3728],
    'UltraFilter':[0.2995, 0.3327, 0.3661, 0.4007, 0.4344]
}

# =========================
# 绘图
# =========================
fig, ax = plt.subplots(figsize=(5,5))
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter)

for (label, y), ls, m, c in zip(url_data.items(), linestyles, markers, colors):
    f_interp = interp1d(x, y, kind='linear')
    y_dense = f_interp(x_dense)
    ax.plot(x_dense, y_dense, linestyle=ls, marker=m, markersize=7, linewidth=CD, color=c, label=label)

ax.set_xlim(0.1, 1.1)
ax.set_ylim(0, 1)
ax.set_xticks(x)
ax.set_xticklabels([r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1$'], fontsize=tick_fontsize)
ax.set_xlabel(r'Privacy budget $\epsilon$', fontsize=label_fontsize)
ax.set_ylabel('RMSE', fontsize=label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=legend_fontsize)

plt.show()
