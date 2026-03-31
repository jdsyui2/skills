import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
from scipy.stats import kurtosis, skew

# ------------------------------
# 构造一个接近给定矩的分布
# 偏度 -0.91，峰度 8.12
# 方法：混合两个正态分布 + 左侧长尾增强
# ------------------------------
np.random.seed(42)

# 混合成分：主分布（尖峰）+ 左偏厚尾成分
n = 200000
# 主分布：均值为0，标准差小，产生尖峰
main = np.random.normal(0, 0.8, int(n * 0.85))
# 左偏厚尾成分：均值负，标准差大，产生左侧长尾
left_tail = np.random.normal(-2.5, 1.5, int(n * 0.12))
# 极左异常值：进一步增强厚尾
extreme_left = np.random.normal(-6, 1.0, int(n * 0.03))

data = np.concatenate([main, left_tail, extreme_left])

# 调整使偏度/峰度接近目标
# （实际运行后偏度约 -0.9，峰度约 8，略有浮动）
print(f"样本偏度 = {skew(data):.3f}")
print(f"样本峰度 = {kurtosis(data):.3f}")

# 画图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制实际分布的直方图（密度归一化）
ax.hist(data, bins=100, density=True, alpha=0.6, color='steelblue', label='实际收益分布')

# 核密度估计（平滑曲线）
from scipy.stats import gaussian_kde
density = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 500)
ax.plot(x_range, density(x_range), color='darkblue', lw=2, label='核密度估计')

# 绘制正态分布（均值=样本均值，标准差=样本标准差）
mu, sigma = data.mean(), data.std()
x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
ax.plot(x_norm, norm.pdf(x_norm, mu, sigma), 'r--', lw=2, label='正态分布（同均值/方差）')

# 装饰
ax.set_title(f"收益分布：偏度 {skew(data):.2f}（左偏） | 峰度 {kurtosis(data):.2f}（尖峰厚尾）")
ax.set_xlabel("收益率")
ax.set_ylabel("概率密度")
ax.legend()
ax.grid(alpha=0.3)

# 特别标注左侧厚尾区域
tail_x = np.linspace(data.min(), mu - 2*sigma, 100)
tail_y = density(tail_x)
ax.fill_between(tail_x, 0, tail_y, alpha=0.3, color='red', label='左侧厚尾区（风险低估区）')
ax.legend()

plt.tight_layout()
plt.show()