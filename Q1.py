import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from scipy.integrate import quad

def calculate_high_level_volume(h, alpha, a, b, m, n):
    """
    计算油面不同高度情况下的储油体积

    参数说明：
    h - 测量点的油面高度（单位：米）
    alpha - 油罐纵向倾斜角度（单位：弧度）
    a - 椭圆半长轴长度（单位：米）
    b - 椭圆半短轴长度（单位：米）
    m - 测量点距离油罐底部的距离（单位：米）
    n - 油罐另一端到测量点的距离（单位：米）

    """
    tan_alpha = np.tan(alpha)
    cot_alpha = 1 / tan_alpha

    # 分界点
    lower_bound = n * tan_alpha
    upper_bound = 2 * b - m * tan_alpha
    if h < 0 or h > 1.2:
        raise ValueError(f"超出范围")

    # 情况1：较低段
    if h <= lower_bound:
        return _volume_low_level(h, alpha, a, b, m)

    # 情况2：中间段
    elif h <= upper_bound:
        return _volume_mid_level(h, alpha, a, b, m, n)

    # 情况3：较高段
    else:
        return _volume_high_level(h, alpha, a, b, m, n)


# 计算截面面积S(h)
def _cross_section_area(h_prime, a, b):
    if h_prime <= 0:
        return 0.0
    if h_prime >= 2 * b:
        return np.pi * a * b

    t = (h_prime - b) / b
    t_clipped = np.clip(t, -1.0, 1.0)
    return a * b * (np.pi / 2 + t_clipped * np.sqrt(1 - t_clipped ** 2) + np.arcsin(t_clipped))


# 低油面情况函数
def _volume_low_level(h, alpha, a, b, m):
    tan_alpha = np.tan(alpha)
    cot_alpha = 1 / tan_alpha

    # 计算积分上限
    L_upper = m + h * cot_alpha

    # 定义被积函数 S₁(l)
    def integrand_low(l):
        h_prime=l*tan_alpha
        return _cross_section_area(h_prime, a, b)

    # 数值积分
    volume, _ = quad(integrand_low, 0, L_upper)
    return volume


# 中间油面情况函数
def _volume_mid_level(h, alpha, a, b, m, n):

    # 定义被积函数 S₂(l)
    def integrand_mid(l):
        h_prime = h - (l - m) * np.tan(alpha)
        return _cross_section_area(h_prime, a, b)

    # 积分
    volume, _ = quad(integrand_mid, 0, m + n)
    return volume


# 高油面情况函数
def _volume_high_level(h, alpha, a, b, m, n):
    tan_alpha = np.tan(alpha)
    cot_alpha = 1 / tan_alpha

    # 计算积分上限
    L_upper = (2 * b - h) * cot_alpha + n

    # 油罐总体积
    total_volume = np.pi * a * b * (m + n)

    # 定义被积函数 S₃(l)
    def integrand_high(l):
        # 计算空气高度
        h_prime=l*tan_alpha
        return _cross_section_area(h_prime, a, b)

    # 计算空气部分体积
    air_volume, _ = quad(integrand_high, 0, L_upper)
    return total_volume - air_volume


# 主函数
if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

    os.makedirs("Figure", exist_ok=True)

    # 油罐参数 (单位：米)
    a = 1.78 / 2  # 半长轴
    b = 1.2 / 2  # 半短轴
    m = 0.4  # 测量点位置
    n = 2.05  # 另一端距离
    alpha = np.radians(4.1)  # 倾斜角度（弧度）
    lower_bound = n * np.tan(alpha)
    upper_bound = 2 * b - m * np.tan(alpha)
    total_volume = np.pi * a * b * (m + n)

    option=['标定计算','倾斜进油','误差拟合','修正标定']
    option=option[3]

    #计算体积
    if option=='标定计算':
        with pd.ExcelWriter('Q1_未修正标定表.xlsx', engine='openpyxl') as writer:
            df = pd.DataFrame()

            #计算体积
            height=[]
            theoretical_volumes = []
            for h in np.arange(0, 1.2+1e-10, 0.01):
                height.append(1000*h)
                volume = calculate_high_level_volume(h, alpha, a, b, m, n)
                theoretical_volumes.append(1000 * volume)

            #保存文件
            df['Height/mm'] = height
            df['Volume/L'] = theoretical_volumes
            df.to_excel(writer, sheet_name='未修正标定表', index=False)  # 修改为使用writer对象
            print("标定油量计算完成并已添加到Excel文件")

        print(f"总储油体积: {1000*total_volume:.4f} L\n")

    # 计算倾斜进油理论体积
    if option=='倾斜进油':
        # 读取Excel文件
        df = pd.read_excel('Q1_倾斜进油.xlsx', sheet_name='倾斜变位进油误差表')

        # 计算理论油量并添加到第三列
        theoretical_volumes = []
        for h in df.iloc[:, 0]:
            volume = calculate_high_level_volume(h/1000, alpha, a, b, m, n)
            theoretical_volumes.append(1000 * volume)

        df['理论油量/L'] = theoretical_volumes

        # 计算绝对误差（理论-实际）并添加到第四列
        df['绝对误差'] = df['理论油量/L'] - df['累加进油量/L']-215

        # 计算相对误差（绝对误差/实际油量）并添加到第五列
        df['相对误差'] = df['绝对误差'] / (df['累加进油量/L']+215)

        # 保存回Excel文件
        df.to_excel('Q1_倾斜进油.xlsx', sheet_name='倾斜变位进油误差表', index=False)
        print("理论油量计算完成并已添加到Excel文件")

    # 误差拟合
    if option=='误差拟合':
        # 读取Excel文件
        df = pd.read_excel('Q1_倾斜进油.xlsx', sheet_name='倾斜变位进油误差表')

        # 绘制绝对误差和相对误差散点图
        plt.figure(figsize=(14, 6))

        # 绝对误差散点图
        plt.subplot(1, 2, 1)
        plt.scatter(df['油位高度/mm'], df['绝对误差'], label='绝对误差', color='blue', alpha=0.6, edgecolors='w', s=50)
        plt.xlabel('油位高度(mm)')
        plt.ylabel('绝对误差(L)')
        plt.title('绝对误差随油高变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 相对误差散点图
        plt.subplot(1, 2, 2)
        plt.scatter(df['油位高度/mm'], df['相对误差'], label='相对误差', color='red', alpha=0.6, edgecolors='w', s=50)
        plt.xlabel('油位高度(mm)')
        plt.ylabel('相对误差')
        plt.title('相对误差随油高变化')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig("Figure/Q1误差散点图.png")
        plt.show()
        plt.close()

        # 对绝对误差进行多项式拟合
        def poly_func(x, *coeffs):
            """多项式函数，接受系数参数"""
            return np.polyval(coeffs[::-1], x)

        x_data = df['油位高度/mm'].values/1000
        y_data = df['绝对误差'].values/1000

        # 使用3次多项式拟合
        degree = 3  # 多项式阶数
        p0 = [0.0] * (degree + 1)  # 初始猜测值
        popt, pcov = curve_fit(poly_func, x_data, y_data, p0=p0)

        # 生成更密集的x值用于绘制平滑曲线
        x_fit = np.linspace(min(x_data), max(x_data), 200)
        y_fit = poly_func(x_fit, *popt)

        # 计算预测值的标准误差以确定置信区间
        y_pred = poly_func(x_data, *popt)
        residuals = y_data - y_pred
        mse = np.mean(residuals ** 2)
        sigma = np.sqrt(mse)

        # 95%置信区间
        ci = 1.96 * sigma

        # 绘制拟合曲线和置信区间
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, label='原始数据', color='blue', alpha=0.6, edgecolors='w', s=50)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'{degree}次多项式拟合')
        plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='pink', alpha=0.3, label='95%置信区间')

        # 添加多项式方程
        equation = f'拟合方程: y = {popt[3]:.4f}x^3 + {popt[2]:.4f}x^2 + {popt[1]:.4f}x + {popt[0]:.4f}'
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.xlabel('油位高度(m)')
        plt.ylabel('绝对误差(m^3)')
        plt.title('绝对误差多项式拟合曲线')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 优化坐标轴刻度
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig("Figure/Q1绝对误差拟合.png")
        plt.show()
        plt.close()

        # 打印拟合方程
        print(f"{degree}次多项式拟合函数:{equation}")

        #计算拟合的绝对误差
        with pd.ExcelWriter('Q1_已修正标定表.xlsx', engine='openpyxl') as writer:
            df = pd.DataFrame()
            heights=[]
            errors = []
            for h in np.arange(0.42,1.03,0.01):
                heights.append(h*1000)
                ab_error_fit = popt[3]*h**3 + popt[2]*h**2 + popt[1]*h + popt[0]
                errors.append(abs(ab_error_fit*1000))
            df['油位高度/mm'] = heights
            df['拟合绝对误差/L'] = errors
            df.to_excel(writer, sheet_name='已修正标定表', index=False)
            print('Q1_已修正标定表已保存')

    #修正标定
    if option=='修正标定':
        df=pd.read_excel('Q1_已修正标定表.xlsx', sheet_name='已修正标定表')
        # 绘制修正标定图
        plt.scatter(df['Height/mm'], df['修正标定值/L'], label='修正标定值/L', color='gray',alpha=0.6,marker='+', s=20)
        plt.xlabel('油位高度(mm)')
        plt.ylabel('修正标定值/L')
        plt.title('修正标定图')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Figure/Q1修正标定图.png")
        plt.show()
        plt.close()

        # 计算5项滑动平均
        df['修正标定值_平滑/L'] = df['修正标定值/L'].rolling(window=5, center=True).mean()
        df.to_excel('Q1_已修正标定表.xlsx', sheet_name='已修正标定表', index=False)

        # 绘制修正标定图
        plt.scatter(df['Height/mm'], df['修正标定值_平滑/L'], label='原始值', color='gray', alpha=0.6, marker='+', s=20)
        plt.xlabel('油位高度(mm)')
        plt.ylabel('修正标定值/L')
        plt.title('修正标定图（带5项滑动平均）')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Figure/Q1修正标定图_平滑.png")
        plt.show()
        plt.close()


