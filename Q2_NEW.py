import numpy as np
import pandas as pd
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import time

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

    def integrand(z, y):
        return safe_sqrt(R_val ** 2 - z ** 2)

    result, _ = dblquad(
        lambda z, y: 2 * integrand(z, y),
        y_low, y_high,
        lambda y: -R_val,
        lambda y: -np.tan(alpha) * (y + 2) + h - R_val,
        epsabs=1e-4,  # 降低精度要求加速计算
        epsrel=1e-4
    )
    return result


def _integrate_sphere_cap(z_limit, r_val):
    """积分计算球冠储油量"""

    def integrand(x, z):
        return safe_sqrt(r_val ** 2 - x ** 2 - z ** 2) - (r_val - R)

    result, _ = dblquad(
        lambda x, z: 2 * integrand(x, z),
        -r_val, z_limit,
        lambda z: 0,
        lambda z: safe_sqrt(r_val ** 2 - z ** 2),
        epsabs=1e-4,
        epsrel=1e-4
    )
    return result


@timer
def V_H_alpha_beta(H, alpha_deg, beta_deg):
    """计算储油量 V(H, α, β)（单位：L）"""
    # 角度转弧度
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)

    # 计算实际油高 h
    h = (H - R) * np.cos(beta) + R

    # 计算中间变量
    if alpha > 1e-6:
        z_F_point = h / np.tan(alpha) - 2  # 圆柱体F点
    else:
        z_F_point = h - R

    z_A = _compute_z_A(h, alpha)  # 左交点 z 坐标
    z_D = _compute_z_D(h, alpha)  # 右交点 z 坐标
    z_E = 0.5 * (z_A + (2 * np.tan(alpha) + h - R))  # 左球冠近似油高
    z_F_cap = 0.5 * (z_D + (-6 * np.tan(alpha) + h - R))  # 右球冠近似油高

    # 计算 V1 (圆柱体部分)
    if z_F_point > 4:  # 油位偏高
        y_low = (h - 3) / np.tan(alpha) - 2
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
    return V_total


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
        # 计算实际出油量 (当前行 - 下一行)
        df["实测出油量"] = -df["出油量/L"].diff(periods=-1)
        return df

    @timer
    def find_optimal_params(self, alpha_range=(0, 10, 1), beta_range=(0, 20, 1)):
        """网格搜索最优参数"""
        best_alpha, best_beta, min_rss = None, None, float('inf')
        alphas = np.arange(*alpha_range)
        betas = np.arange(*beta_range)

        print(f"开始参数搜索: α范围[{alpha_range[0]},{alpha_range[1]}]步长{alpha_range[2]}, "
              f"β范围[{beta_range[0]},{beta_range[1]}]步长{beta_range[2]}")

        for alpha in alphas:
            for beta in betas:
                rss = self.calculate_rss(alpha, beta)
                # 打印当前参数的RSS值
                print(f"当前参数: α={alpha:.2f}°, β={beta:.2f}°, RSS={rss:.2f}")
                if rss < min_rss:
                    min_rss = rss
                    best_alpha, best_beta = alpha, beta
                    print(f"发现更优参数: α={alpha:.2f}°, β={beta:.2f}°, RSS={rss:.2f}")

        # 在最优解附近细化搜索
        alpha_fine = np.linspace(max(0, best_alpha - 0.5), min(10, best_alpha + 0.5), 11)
        beta_fine = np.linspace(max(0, best_beta - 0.5), min(20, best_beta + 0.5), 11)

        for alpha in alpha_fine:
            for beta in beta_fine:
                rss = self.calculate_rss(alpha, beta)
                if rss < min_rss:
                    min_rss = rss
                    best_alpha, best_beta = alpha, beta

        self.alpha, self.beta = best_alpha, best_beta
        print(f"\n最优参数: α = {best_alpha:.4f}°, β = {best_beta:.4f}°")
        return best_alpha, best_beta

    def calculate_rss(self, alpha, beta):
        """计算残差平方和"""
        residuals = []
        valid_df = self.df[(self.df["显示油高_m"] >= 1.7) & (self.df["显示油高_m"] <= 2.4)]

        # 预先计算所有高度对应的容积
        volumes = [V_H_alpha_beta(H, alpha, beta) for H in valid_df["显示油高_m"]]

        # 计算理论出油量
        for i in range(len(valid_df) - 1):
            delta_V_theory = volumes[i + 1] - volumes[i]
            delta_V_actual = valid_df["实测出油量"].iloc[i]
            residuals.append((delta_V_actual - delta_V_theory) ** 2)

        return np.sum(residuals) if residuals else float('inf')

    @timer
    def generate_calibration_table(self, H_min=0.4, H_max=2.6, step=0.1):
        """生成罐容表标定值"""
        H_range = np.arange(H_min, H_max + step, step)
        calibration_data = []

        for H in H_range:
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
        self.df["相对误差(%)"] = np.abs(self.df["残差"] / self.df["实测出油量"]) * 100

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

        plt.figure(figsize=(15, 10))

        # 残差分布图
        plt.subplot(2, 2, 1)
        plt.hist(self.df["残差"], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.title("残差分布")
        plt.xlabel("残差 (L)")
        plt.ylabel("频数")

        # 相对误差分布
        plt.subplot(2, 2, 2)
        plt.hist(self.df["相对误差(%)"], bins=30, color='lightgreen', edgecolor='black')
        plt.title("相对误差分布")
        plt.xlabel("相对误差 (%)")
        plt.ylabel("频数")

        # 残差随油高变化
        plt.subplot(2, 2, 3)
        plt.scatter(self.df["显示油高_m"], self.df["残差"], alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("残差随油高变化")
        plt.xlabel("油高 (m)")
        plt.ylabel("残差 (L)")

        # 箱线图
        plt.subplot(2, 2, 4)
        plt.boxplot(self.df["残差"], vert=True, patch_artist=True)
        plt.title("残差箱线图")
        plt.ylabel("残差 (L)")

        plt.tight_layout()
        plt.savefig("Figure/error_analysis.png")
        plt.show()


# ====================== 主程序 ======================
if __name__ == '__main__':
    # 初始化分析器
    analyzer = OilTankAnalyzer(
        "第1周B题：储油罐的变位识别与罐容表标定(2010年国赛A题)/问题A附件2：实际采集数据表.xls"
    )

    # 执行选项
    options = {
        "1": "参数识别",
        "2": "生成罐容表",
        "3": "误差分析",
        "4": "可视化分析",
        "5": "执行全部流程"
    }

    choice = "1"

    if choice == "1":
        print("\n=== 参数识别 ===")
        alpha, beta = analyzer.find_optimal_params()

    elif choice == "2":
        print("\n=== 罐容表生成 ===")
        calibration_table = analyzer.generate_calibration_table()
        print(calibration_table.head(10))
        calibration_table.to_csv("Q2罐容表标定值.csv", index=False)
        print("罐容表已保存至 'Q2罐容表标定值.csv'")

    elif choice == "3":
        print("\n=== 误差分析 ===")
        error_report = analyzer.analyze_errors()
        print("\n误差分析报告:")
        for metric, value in error_report.items():
            print(f"{metric}: {value:.4f}")

    elif choice == "4":
        analyzer.plot_error_analysis()
        print("误差分析图已保存至 'Figure/error_analysis.png'")

    # elif choice == "5":
    #     print("\n=== 执行全部流程 ===")
    #     analyzer.find_optimal_params()
    #     analyzer.generate_calibration_table()
    #     analyzer.analyze_errors()
    #     analyzer.plot_error_analysis()
    #     print("所有流程执行完成")

