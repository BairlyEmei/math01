import numpy as np
import pandas as pd
from scipy.integrate import dblquad, quad
import matplotlib.pyplot as plt
import time
import os
from scipy.optimize import minimize
from tqdm import tqdm

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# ====================== 常量定义 ======================
R = 1.5  # 圆柱体半径 (m)
L = 8.0  # 圆柱体长度 (m)
r = 1.625  # 球冠半径 (m)
y0_left = -3.375  # 左球冠中心 y 坐标 (m)
y0_right = 3.375  # 右球冠中心 y 坐标 (m)


# ====================== 辅助函数 ======================
def safe_sqrt(x):
    """值域保护开方函数"""
    return np.sqrt(np.maximum(x, 1e-10))


def timer(func):
    """函数执行时间装饰器"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # print(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result

    return wrapper


# ====================== 核心数学模型 ======================
def _compute_z_A(h, alpha):
    """计算左交点 z_A (公式 1.1)"""
    tan_alpha = np.tan(alpha)
    term1 = 1.375 * tan_alpha + h - R
    term2 = safe_sqrt(
        (R ** 2 * tan_alpha ** 4) +
        ((1.375 ** 2 + R ** 2) * tan_alpha ** 2) +
        (2.75 * (h - R) * tan_alpha) +
        (h - R) ** 2
    )
    denominator = 1 + tan_alpha ** 2
    return (term1 - term2) / denominator


def _compute_z_D(h, alpha):
    """计算右交点 z_D (公式 1.2)"""
    tan_alpha = np.tan(alpha)
    term1 = -5.375 * tan_alpha + h - R
    term2 = safe_sqrt(
        (R ** 2 * tan_alpha ** 4) +
        ((5.375 ** 2 + R ** 2) * tan_alpha ** 2) -
        (10.75 * (h - R) * tan_alpha) +
        (h - R) ** 2
    )
    denominator = 1 + tan_alpha ** 2
    return (term1 + term2) / denominator


def _integrate_cylinder(h, alpha, y_low, y_high, R_val):
    """积分计算圆柱体储油量"""
    # 确保积分区域有效
    if y_low >= y_high:
        return 0.0

    def integrand(z, y):
        return safe_sqrt(R_val ** 2 - z ** 2)

    # 计算积分
    try:
        result, _ = dblquad(
            lambda z, y: 2 * integrand(z, y),
            y_low, y_high,
            lambda y: -R_val,
            lambda y: min(R_val, max(-R_val, -np.tan(alpha) * (y + 2) + h - R_val)),
            epsabs=1e-3,
            epsrel=1e-3
        )
        return max(0, result)  # 确保非负
    except:
        return 0.0


def _integrate_sphere_cap(z_limit, r_val):
    """积分计算球冠储油量"""
    # 确保积分区域有效
    if z_limit < -r_val or z_limit > r_val:
        return 0.0

    def integrand(x, z):
        return safe_sqrt(r_val ** 2 - x ** 2 - z ** 2) - (r_val - R)

    try:
        result, _ = dblquad(
            lambda x, z: 2 * integrand(x, z),
            -r_val, min(z_limit, r_val),
            lambda z: 0,
            lambda z: safe_sqrt(r_val ** 2 - z ** 2),
            epsabs=1e-3,
            epsrel=1e-3
        )
        return max(0, result)  # 确保非负
    except:
        return 0.0


@timer
def V_H_alpha_beta(H, alpha_deg, beta_deg):
    """计算储油量 V(H, α, β)（单位：L）"""
    # 角度转弧度
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)

    # 计算实际油高 h
    h = (H - R) * np.cos(beta) + R

    # 确保h在合理范围内
    h = max(0, min(3.0, h))

    # 计算中间变量
    if alpha > 1e-6:
        z_F_point = max(-4, min(4, h / np.tan(alpha) - 2))  # 边界保护
    else:
        z_F_point = h - R

    z_A = max(-4, min(4, _compute_z_A(h, alpha)))  # 边界保护
    z_D = max(-4, min(4, _compute_z_D(h, alpha)))  # 边界保护
    z_E = max(0, min(3.0, 0.5 * (z_A + (2 * np.tan(alpha) + h - R))))  # 边界保护
    z_F_cap = max(0, min(3.0, 0.5 * (z_D + (-6 * np.tan(alpha) + h - R))))  # 边界保护

    # 计算 V1 (圆柱体部分)
    if z_F_point > 4:  # 油位偏高
        y_low = max(-4, min(4, (h - 3) / np.tan(alpha) - 2))  # 边界保护
        V1 = 8 * np.pi * R ** 2 - _integrate_cylinder(h, alpha, y_low, 4, R)
    elif -2 <= z_F_point <= 4:  # 油位中等
        V1 = _integrate_cylinder(h, alpha, -4, z_F_point, R)
    else:  # 油位偏低
        V1 = _integrate_cylinder(h, alpha, -4, 4, R)

    # 计算 V2 (左球冠)
    V2 = _integrate_sphere_cap(z_E, r)

    # 计算 V3 (右球冠)
    V3 = _integrate_sphere_cap(z_F_cap, r)

    # 总储油量 (m³ 转 L)
    V_total = (V1 + V2 + V3) * 1000
    return max(0, V_total)  # 确保非负


# ====================== 数据处理与分析模块 ======================
class OilTankAnalyzer:
    def __init__(self, data_path):
        self.df = self.load_data(data_path)
        self.alpha = None
        self.beta = None
        self.calibration_table = None
        self.error_report = None

    def load_data(self, path):
        """加载并预处理数据"""
        df = pd.read_excel(path)
        # 毫米转米
        df["显示油高_m"] = df["显示油高/mm"] / 1000
        # 计算实际出油量 (下一行 - 当前行)
        df["实测出油量"] = -df["出油量/L"].diff(periods=-1)
        return df

    @timer
    def find_optimal_params(self, init_alpha=2.1, init_beta=4.2):
        """使用优化算法寻找最优参数"""
        print(f"开始参数优化: 初始值 α={init_alpha}°, β={init_beta}°")

        # 定义目标函数
        def objective(params):
            alpha, beta = params
            return self.calculate_rss(alpha, beta)

        # 使用优化算法
        result = minimize(
            objective,
            x0=[init_alpha, init_beta],
            method='Nelder-Mead',
            bounds=[(0, 10), (0, 20)],
            options={'maxiter': 100, 'disp': True}
        )

        if result.success:
            self.alpha, self.beta = result.x
            print(f"\n优化成功: α = {result.x[0]:.4f}°, β = {result.x[1]:.4f}°, RSS={result.fun:.2f}")
            return result.x
        else:
            print("优化失败，使用默认参数")
            self.alpha, self.beta = 2.1, 4.2
            return self.alpha, self.beta

    def calculate_rss(self, alpha, beta):
        """计算残差平方和"""
        residuals = []
        valid_df = self.df[(self.df["显示油高_m"] >= 1.7) & (self.df["显示油高_m"] <= 2.4)]

        # 预先计算所有高度对应的容积
        volumes = []
        for H in valid_df["显示油高_m"]:
            try:
                volumes.append(V_H_alpha_beta(H, alpha, beta))
            except:
                volumes.append(0)

        # 计算理论出油量
        for i in range(len(valid_df) - 1):
            delta_V_theory = volumes[i + 1] - volumes[i]
            delta_V_actual = valid_df["实测出油量"].iloc[i]

            # 确保数据有效
            if np.isfinite(delta_V_theory) and np.isfinite(delta_V_actual):
                residuals.append((delta_V_actual - delta_V_theory) ** 2)

        return np.sum(residuals) if residuals else float('inf')

    @timer
    def generate_calibration_table(self, H_min=0.1, H_max=3.0, step=0.1):
        """生成罐容表标定值"""
        H_range = np.arange(H_min, H_max + step, step)
        calibration_data = []

        # 使用进度条
        for H in tqdm(H_range, desc="生成罐容表"):
            V = V_H_alpha_beta(H, self.alpha, self.beta)
            calibration_data.append({"油高(m)": H, "储油量(L)": V})

        self.calibration_table = pd.DataFrame(calibration_data)
        return self.calibration_table

    @timer
    def analyze_errors(self):
        """误差分析"""
        if self.alpha is None or self.beta is None:
            raise ValueError("请先运行参数搜索确定最优参数")

        # 计算理论容积
        self.df["理论容积"] = self.df["显示油高_m"].apply(
            lambda H: V_H_alpha_beta(H, self.alpha, self.beta)
        )

        # 计算理论出油量
        self.df["理论出油量"] = -self.df["理论容积"].diff(periods=-1)

        # 计算残差
        self.df["残差"] = self.df["实测出油量"] - self.df["理论出油量"]
        self.df["绝对误差"] = np.abs(self.df["残差"])
        epsilon = 1e-5
        self.df["相对误差(%)"] = np.abs(self.df["残差"]) / (self.df["实测出油量"] + epsilon) * 100

        # 计算总体误差指标
        mae = np.mean(self.df["绝对误差"])
        rmse = np.sqrt(np.mean(self.df["残差"] ** 2))
        avg_rel_error = np.mean(self.df["相对误差(%)"])
        max_error = np.max(self.df["绝对误差"])

        # 创建误差报告
        self.error_report = {
            "MAE (L)": mae,
            "RMSE (L)": rmse,
            "平均相对误差 (%)": avg_rel_error,
            "最大绝对误差 (L)": max_error
        }

        return self.error_report

    def plot_error_analysis(self):
        """绘制误差分析图"""
        if self.error_report is None:
            self.analyze_errors()

        plt.figure(figsize=(10, 4))

        # 残差分布图
        plt.subplot(1, 2, 1)
        plt.hist(self.df["残差"], bins=30, color='lightgray', edgecolor='black')
        plt.axvline(0, color='black', linestyle='--')
        plt.title("残差分布")
        plt.xlabel("残差 (L)")
        plt.ylabel("频数")

        # 残差随油高变化
        plt.subplot(1, 2, 2)
        plt.scatter(self.df["显示油高_m"], self.df["残差"], alpha=0.6,color='gray',marker='x')
        plt.axhline(0, color='black', linestyle='--')
        plt.title("残差随油高变化")
        plt.xlabel("油高 (m)")
        plt.ylabel("残差 (L)")

        plt.tight_layout()
        plt.savefig("Figure/error_analysis.png")
        plt.show()

    def plot_calibration_curve(self):
        """绘制罐容曲线"""
        if self.calibration_table is None:
            self.generate_calibration_table()

        plt.figure(figsize=(10, 6))
        plt.scatter(self.calibration_table["油高(m)"], self.calibration_table["储油量(L)"], alpha=0.6,color='gray',marker='+')
        plt.plot(self.calibration_table["油高(m)"], self.calibration_table["储油量(L)"], color="gray")
        plt.title("罐容曲线")
        plt.xlabel("油高 (m)")
        plt.ylabel("储油量 (L)")
        plt.grid(True)

        # 添加关键点
        critical_points = [0.5, 1.0, 1.5, 2.0, 2.5]
        for point in critical_points:
            idx = np.argmin(np.abs(self.calibration_table["油高(m)"] - point))
            plt.plot(point, self.calibration_table["储油量(L)"].iloc[idx], color="gray",marker='+')
            plt.text(point, self.calibration_table["储油量(L)"].iloc[idx],
                     f'{self.calibration_table["储油量(L)"].iloc[idx]:.0f}L',
                     fontsize=9)

        plt.savefig("Figure/calibration_curve.png")
        plt.show()


# ====================== 主程序 ======================
if __name__ == '__main__':
    # 初始化分析器
    analyzer = OilTankAnalyzer(
        "第1周B题：储油罐的变位识别与罐容表标定(2010年国赛A题)/问题A附件2：实际采集数据表.xls"
    )

    # 设置最优参数（根据之前优化结果）
    analyzer.alpha, analyzer.beta = 2.1, 4.2

    # 执行选项
    options = {
        "1": "参数识别",
        "2": "生成罐容表",
        "3": "误差分析",
        "4": "可视化分析",
        "5": "执行全部流程",
        "6": "退出"
    }

    choice = "4"

    if choice == "1":
        print("\n=== 参数识别 ===")
        alpha, beta = analyzer.find_optimal_params()

    elif choice == "2":
        print("\n=== 罐容表生成 ===")
        calibration_table = analyzer.generate_calibration_table()
        print(calibration_table.head(10))
        calibration_table.to_csv("Q2罐容表标定值.csv", index=False)
        print("罐容表已保存至 'Q2罐容表标定值.csv'")

        # 绘制罐容曲线
        analyzer.plot_calibration_curve()
        print("罐容曲线已保存至 'Figure/calibration_curve.png'")

    elif choice == "3":
        print("\n=== 误差分析 ===")
        error_report = analyzer.analyze_errors()
        print("\n误差分析报告:")
        for metric, value in error_report.items():
            print(f"{metric}: {value:.4f}")

    elif choice == "4":
        print("\n=== 可视化分析 ===")
        analyzer.plot_error_analysis()
        analyzer.plot_calibration_curve()

    elif choice == "5":
        print("\n=== 执行全部流程 ===")
        # # 参数识别
        # alpha, beta = analyzer.find_optimal_params()

        # 生成罐容表
        calibration_table = analyzer.generate_calibration_table()
        calibration_table.to_csv("Q2罐容表标定值.csv", index=False)

        # 误差分析
        error_report = analyzer.analyze_errors()

        # 可视化
        analyzer.plot_error_analysis()
        analyzer.plot_calibration_curve()

        # 输出结果
        print("\n===== 最终结果 =====")
        # print(f"最优参数: α = {alpha:.4f}°, β = {beta:.4f}°")
        print("误差分析报告:")
        for metric, value in error_report.items():
            print(f"{metric}: {value:.4f}")
        print("罐容表已保存至 'Q2罐容表标定值.csv'")