from scipy import stats

# 定义数组
data = [1.72, 1.65, 1.66, 1.58, 1.42, 1.5, 1.44, 1.37, 1.4, 1.53]

# 执行夏皮罗-威尔克法检验
statistic, p_value = stats.shapiro(data)

# 输出结果
print("Shapiro-Wilk test statistic:", statistic)
print("P-value:", p_value)

# 判断是否拒绝原假设
alpha = 0.05
if p_value > alpha:
    print("样本服从正态分布（接受原假设）")
else:
    print("样本不服从正态分布（拒绝原假设）")