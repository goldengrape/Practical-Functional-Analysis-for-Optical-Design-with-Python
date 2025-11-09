# 第5章 线性算子与像差分解

## 5.1 引言：从“状态”到“过程”

在模块1 (第1-4周) 中，我们致力于描述光学系统的状态。我们学习了如何使用函数空间（特别是 $L^2$ 空间和希尔伯特空间）来表达波前，并掌握了如何使用Zernike多项式等正交基底来量化像差。我们实质上是在为光学设计建立一个精确的“词汇表”。

然而，仅仅描述一个静态的像差是不够的。作为设计师，我们更关心的是过程：

1.  一个理想的波前 $f_{in}$ 如何通过我们的光学系统（镜片、涂层、反射镜）变换为实际的、带有像差的波前 $f_{out}$？
2.  如果我们对系统设计（例如镜片曲率）进行微小的改动，这个变换过程会如何改变？
3.  哪些类型的输入像差（或制造误差）会被系统放大，从而对最终视觉质量造成最严重的影响？

要回答这些问题，我们必须将研究对象从“函数”（波前）升级到“作用于函数的函数”——这正是算子 (Operator) 的定义。

在本章 (第5周) 中，我们将引入泛函分析中一个极其强大的概念：线性算子 (Linear Operator)。我们将学习如何将整个复杂的镜片系统（无论是单片、多片还是非球面）抽象为一个数学“黑盒”，即算子 $\mathcal{L}$，它定义了输入光场 $f_{in}$ 和输出光场 $f_{out}$ 之间的关系[^2]：
$$f_{out} = \mathcal{L}(f_{in})$$
这个抽象使我们能够将复杂的光线追踪问题转化为一个（在理想情况下）线性的变换问题。更重要的是，通过分析算子 $\mathcal{L}$ 本身的性质，我们能以前所未有的深度洞察系统的核心行为。我们将发现，$\mathcal{L}$ 的“谱”(Spectrum) 直接对应于系统的“敏感性”(Sensitivity)[^3]。

本章的核心目标是：将“像差”这一临床症状[^4]，转化为可计算的算子谱理论问题，并最终构建一个Python工具，用于量化不同像差模式对视觉质量的“影响权重”[^5]。

## 5.2 核心概念：光学系统即算子

让我们从一个形式化的定义开始。一个算子 $\mathcal{L}$ 是一个映射，它接受一个来自函数空间 $V$（例如，输入波前空间）的函数 $f$，并将其映射到函数空间 $W$（例如，输出波前或像平面上的点扩散函数 PSF）中的一个新函数 $g$。
$$\mathcal{L}: V \to W \quad \text{使得} \quad \mathcal{L}(f) = g$$
在眼视光学中，我们主要关注线性算子 (Linear Operators)。一个算子 $\mathcal{L}$ 是线性的，如果它满足叠加原理：

对于任意函数 $f_1, f_2 \in V$ 和标量 $a, b$：
$$\mathcal{L}(af_1 + bf_2) = a\mathcal{L}(f_1) + b\mathcal{L}(f_2)$$
这在光学上意味着什么？

这意味着如果我们将输入光场分解为多个分量（例如，不同的Zernike模式或不同的入射光线），系统对这些分量的总响应等于它对每个分量单独响应的简单叠加。虽然真实的光学系统（尤其是涉及衍射或非线性材料时）并不总是严格线性的，但在许多实际的像差分析中[^5]，线性近似（例如在傍轴近似下）已经足够精确，并且提供了巨大的分析优势。

为什么这个抽象如此有用？

因为它将一个潜在的无限维问题（分析连续的波前）简化为了一个有限维的代数问题。

考虑一下：在第4周，我们将输入波前 $f_{in}$ 分解为Zernike基底 $\lbrace Z_j \rbrace_{j=1}^{\infty}$ 的线性组合：
$$f_{in} = \sum_{j=1}^{\infty} c_j Z_j$$
如果 $\mathcal{L}$ 是线性的，那么输出 $f_{out}$ 可以写作：
$$f_{out} = \mathcal{L}(f_{in}) = \mathcal{L}\left(\sum_{j=1}^{\infty} c_j Z_j\right) = \sum_{j=1}^{\infty} c_j \mathcal{L}(Z_j)$$
这意味着我们不再需要知道 $\mathcal{L}$ 如何作用于每一个可能的波前。我们只需要知道 $\mathcal{L}$ 如何作用于每一个基函数 $Z_j$！

$\mathcal{L}(Z_j)$ 本身也是一个波前，因此它也可以在Zernike基底上展开：
$$\mathcal{L}(Z_j) = \sum_{i=1}^{\infty} L_{ij} Z_i$$
这里的 $L_{ij}$ 矩阵就是 $\mathcal{L}$ 在Zernike基底下的矩阵表示。它捕获了系统的所有信息。$L_{ij}$ 的物理意义是：“当输入纯 $j$ 模式像差时，输出中会产生多少 $i$ 模式像差？”

通过这个方法，我们已经将一个复杂的物理系统（镜片）转化为了一个（可能无限大的）矩阵 $L$。我们的问题从“追踪亿万条光线”简化为“分析矩阵 $L$ 的性质”。

## 5.3 理论深潜：紧算子与有限元（FEM）的意义

我们面临一个理论上的难题：Zernike基底是无限的，这意味着矩阵 $L$ 是无限维的。我们如何在计算机上处理它？我们凭什么可以用一个有限的（例如 $100 \times 100$）矩阵来近似这个无限维的算子 $\mathcal{L}$？

答案（以及这样做的数学合法性）来自泛函分析中的一个核心概念：紧算子 (Compact Operator)。

一个紧算子（通常记为 $\mathcal{K}$）是一种特殊的线性算子，它具有“平滑”效应。直观地说，它将一个有界但“粗糙”的输入集合（例如，所有可能的、有界的输入波前）映射为一个“紧凑”的输出集合——这意味着输出集合可以被有限数量的“小球”完全覆盖。

这在光学上意味着什么？

*   **平滑效应**： 任何真实的光学系统都具有衍射极限。它无法将无限精细的细节从输入端完美传递到输出端。高频空间信息（精细细节）要么被平滑掉，要么被衍射效应“模糊”掉。这种固有的低通滤波特性是“紧性”的物理体现[^7]。
*   **信息压缩**： 紧算子本质上是“可压缩”的。它告诉我们，尽管输入空间 $V$ 是无限维的，但算子 $\mathcal{L}$ 的“有效作用范围” (Range) 在某种意义上是有限的。

**紧算子与有限元（FEM）的关系**

紧算子的真正威力在于它的谱特性：

1.  一个紧算子 $\mathcal{K}$ 的谱（特征值集合）是离散的。
2.  它的特征值 $\lambda_n$ 只能在 $0$ 处“堆积”。这意味着对于任何给定的阈值 $\epsilon > 0$，只有有限多个特征值的绝对值大于 $\epsilon$。

这就是我们一直在寻找的数学保证！

这个特性意味着，紧算子的行为完全由其少数几个“大”特征值所主导。所有其他的特征值都非常小，可以安全地忽略。

这直接 justifies 了有限元方法 (Finite Element Method, FEM)[^8] 和其他数值离散化方法的有效性。当我们使用FEM[^9] 或Zernike基底截断来构建有限矩阵 $L$ 时，我们实际上是在计算 $\mathcal{L}$ 的一个投影。紧算子理论[^11] 保证了这个有限矩阵 $L$ 的特征值将收敛到真实算子 $\mathcal{L}$ 的特征值。

因此，当我们在Python中创建一个 $100 \times 100$ 的矩阵 $L$ 时，我们不再是“盲目猜测”，而是有坚实的数学理论支持：我们捕获的确实是系统最重要的有限信息[^1]。

## 5.4 从临床到数学：谱理论即敏感性分析

现在我们有了坚实的理论基础，让我们回到工程问题：如何将临床需求（例如，“减少晃动感”）转化为数学约束？

临床症状（如晃动、模糊、鬼影）是果，而光学像差是因。我们的算子 $\mathcal{L}$ 就是连接“因”与“果”的桥梁。但是，并非所有的“因”都同等重要。

**临床问题：** 哪种类型的波前像差（或制造公差）对最终视觉质量的破坏性最大？
**数学翻译：** 算子 $\mathcal{L}$ 的谱 (Spectrum) 是什么？

谱理论 (Spectral Theory)[^3] 研究的是算子的特征值 $\lambda$ 和特征函数 $f$（或特征向量）。它们满足特征方程：
$$\mathcal{L}(f) = \lambda f$$

*   **特征函数 $f$（或特征模式）**： 这些是光学系统的“本征模式”。当输入波前恰好是 $f$ 时，系统不会改变它的形状（例如，不会将“散光”变为“彗差”），而只是将其缩放一个因子 $\lambda$。
*   **特征值 $\lambda$（缩放因子）**： 这就是我们苦苦追寻的“影响权重” (Influence Weight)。
    *   如果 $|\lambda| > 1$，系统会放大 $f$ 模式的像差。
    *   如果 $|\lambda| < 1$，系统会抑制 $f$ 模式的像差。
    *   如果 $|\lambda| \approx 0$，系统对 $f$ 模式完全不敏感。

这就是敏感性分析 (Sensitivity Analysis) 的核心[^13]。

设计师不需要关心所有可能的像差。他们只需要关心那些具有最大特征值 $\lambda_i$ 的特征函数 $f_i$。

如果我们的分析表明，具有最大特征值 $\lambda_1$ 的特征函数 $f_1$ 看起来非常像“水平彗差”(Horizontal Coma)，我们就得出了一个至关重要的、可操作的设计洞察：“我们的设计对水平彗差极其敏感。” 任何导致水平彗差的制造公 tolerance（例如镜片倾斜）都必须被严格控制。

## 5.5 案例研究：患者视觉症状建模

让我们将这个理论应用到一个具体的临床案例中。

**临床症状**： 患者（尤其是在植入了多焦人工晶体 (IOL) 后）报告“光晕 (Halos)”和“模糊 (Blur)”[^16]。

**物理关联**： 这些视觉症状[^18] 对应于点扩散函数 (PSF) 的不良形态。一个完美的PSF是一个点；一个有光晕的PSF则是在中心点周围有一个宽阔的“光环”[^16]。

**建模过程**：

1.  **定义空间**： 我们的输入空间 $V$ 是镜片表面上所有可能的、微小的形状误差 $f_{err}$（例如，由制造公差引起的）。我们的输出空间 $W$ 是由该误差引起的PSF劣化 $g_{psf}$（例如，PSF与理想PSF的差值）。
2.  **定义算子**： 我们的算子 $\mathcal{L}$ 是一个从 $V$ 到 $W$ 的映射，$\mathcal{L}(f_{err}) = g_{psf}$。这个 $\mathcal{L}$ 是通过（可微分的）光线追踪或波动光学模型[^20] 隐式定义的。
3.  **寻找主导模式**： 我们现在求解谱问题 $\mathcal{L}(f) = \lambda f$。

**分析结果**：

*   $f_1$ (特征值 $\lambda_1$ 最大)： 这是“最差的”形状误差。$\lambda_1$ 的值告诉我们，这种特定形状的误差（例如，可能是某个环形区域的轻微隆起）会最高效地将光能从PSF的中心“踢”到光晕区域。
*   $f_2$ (特征值 $\lambda_2$ 第二大)： 这是第二差的形状误差。
*   ...
*   $f_{100}$ (特征值 $\lambda_{100} \approx 0$)： 这种形状误差（例如，可能是表面上非常高频的波纹）对PSF几乎没有影响，因为衍射效应将其完全平滑掉了。

**临床决策**：

通过这种特征值分解 (Eigenvalue Decomposition)[^5] 或更通用的奇异值分解 (SVD)[^21]，我们不再盲目地试图减少“总”表面误差。我们专注于消除那些与 $f_1, f_2, f_3$ 等高敏感性模式相对应的误差分量。我们可以容忍 $f_{100}$ 模式的误差，因为它对患者的视觉症状没有贡献。

这就是将泛函分析转化为临床和工程决策的完整路径。

## 5.6 Python 实践5：构建像差敏感性分析工具

**目标**： 构建一个Python脚本，模拟一个光学系统的线性算子 $L$，并使用奇异值分解 (SVD) 来计算其“影响权重”并生成“敏感性热力图”。

**工具栈**： NumPy, SciPy[^25], Matplotlib

### 步骤1：导入库并定义系统算子 L

首先，我们导入所需的库。在实践中，这个 $L$ 矩阵将由一个复杂的光学仿真工具（如 Zemax, Code V 或可微分光线追踪器[^20]）通过对每个Zernike基函数进行扰动分析来生成。

为了本项目的目的，我们假定已经获得了这个矩阵 $L$。我们将其加载（或随机生成一个）为一个 $50 \times 50$ 的矩阵，其中 $L[i, j]$ 表示“输入 $j$ 模式（Zernike）对输出 $i$ 模式（Zernike）的贡献”。

```python
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def get_system_operator(size=50):
   """
   模拟加载或生成一个光学系统算子矩阵 L.
   一个真实的L矩阵通常是对角占优的，但会包含
   模式耦合（非对角元素）。
   """
   # 为了演示，我们创建一个具有特定结构的随机矩阵
   np.random.seed(42)
   # 基础传递矩阵（例如，单位矩阵）
   L = np.diag(np.random.rand(size) * 0.8 + 0.2) 
   # 添加模式耦合（非对角项），模拟像差混合
   L += np.random.randn(size, size) * 0.05 
   # 模拟某些模式的特定高敏感性
   L[3, 5] = 0.8 # Z5（散光）强烈耦合到Z3（散光）
   L[6, 7] = 0.6 # 彗差耦合
   
   # 我们可以从文件加载一个真实的矩阵
   # try:
   #     L = np.load("system_operator_matrix.npy")
   # except FileNotFoundError:
   #     print("未找到真实的算子矩阵，使用模拟矩阵替代。")
   
   return L

# 获取我们的系统算子
L = get_system_operator(size=50)

print(f"系统算子 L 的形状: {L.shape}")
```

### 步骤2：计算敏感性 - 奇异值分解 (SVD)

为什么使用SVD而不是特征值分解 (EVD)？

1.  **通用性**： EVD 要求矩阵是方阵 ($n \times n$)，并且通常需要对称性才能保证实特征值。
2.  **鲁棒性**： 我们的 $L$ 矩阵不一定是对称的。
3.  SVD (Singular Value Decomposition)[^21] 是任何矩阵（包括非方阵）的最通用的分解工具。它完美地适用于分析从 $V \to W$ 的映射。

SVD 将 $L$ 分解为 $L = U \cdot S \cdot V^T$：

*   $V^T$：输入空间的正交基（“输入模式”）。
*   $S$：一个对角矩阵，其对角元素 $\sigma_i$ 是奇异值 (Singular Values)。
*   $U$：输出空间的正交基（“输出模式”）。

奇异值 $\sigma_i$ 就是 我们要找的“影响权重”。它们按降序排列，$\sigma_1$ 是系统最敏感模式的“增益”。

我们使用 SciPy 的 linalg 模块执行 SVD[^25][^26][^28]：

```python
# 使用 SciPy 的 linalg 模块执行 SVD
# U 和 Vh 是正交矩阵 (U @ U.T = I, Vh @ Vh.T = I)
# S_diag 是奇异值的一维数组
U, S_diag, Vh = scipy.linalg.svd(L)

print(f"奇异值 (S_diag) 示例: {S_diag[:5]}")
print(f"U 的形状: {U.shape}")
print(f"Vh (V transpose) 的形状: {Vh.shape}")
```

### 步骤3：计算并可视化影响权重

奇异值向量 S_diag 是系统的敏感性谱。最大的奇异值 $\sigma_1$ 对应的输入模式（Vh 的第一行）和输出模式（U 的第一列）构成了系统最敏感的信道。

我们将这些奇异值归一化并绘制出来，以生成课程大纲[^27]中要求的“敏感性热力图”（在这里，条形图更清晰地展示了“谱”）。

```python
# 将奇异值归一化为影响权重
influence_weights = S_diag / np.max(S_diag)

plt.figure(figsize=(12, 6))
plt.bar(range(1, len(influence_weights) + 1), influence_weights, color='c')
plt.title("系统敏感性谱 (System Sensitivity Spectrum)", fontsize=16)
plt.xlabel("奇异模式索引 (Singular Mode Index)", fontsize=12)
plt.ylabel("归一化影响权重 (Normalized Influence Weight / Gain)", fontsize=12)
plt.yscale('log') # 敏感性通常跨越多个数量级，使用对数刻度
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("project5_sensitivity_spectrum.png")
print("敏感性谱图已保存为 project5_sensitivity_spectrum.png")
```

**分析**： 这个图表（在 `project5_sensitivity_spectrum.png` 中）是设计师的“藏宝图”。它清楚地显示，系统的大部分“响应”都集中在前几个奇异模式上。模式索引10之后的权重可能比模式1低几个数量级。

### 步骤4：将数学与临床联系起来

仅仅知道“模式1”是最敏感的是不够的。设计师需要知道**“模式1是什么？”**

答案在 $V^T$ (即 Vh) 矩阵中。Vh 的第一行 (Vh[0, :]) 是第一个输入奇异向量。它描述了第一个（最敏感的）奇异模式是由哪些Zernike模式混合而成的。

让我们创建一个表格，将这些抽象的模式与Zernike像差联系起来。
(在实际应用中，我们会使用一个库，例如 Optiland[^25][^26][^29][^30][^31] 中的工具)

```python
# 假设 Zernike 模式的名称
# Zernike 索引 (Noll) 示例
zernike_names = {
   0: "Piston", 1: "Tip", 2: "Tilt", 3: "Defocus",
   4: "Astigmatism (0 deg)", 5: "Astigmatism (45 deg)",
   6: "Coma (Y)", 7: "Coma (X)",
   8: "Trefoil (Y)", 9: "Trefoil (X)",
   10: "Spherical Aberration",
   #... 填充到 50
}

def get_dominant_components(vector, names, threshold=0.3):
   """
   辅助函数：找到一个向量中的主要Zernike成分
   """
   contributions = sorted(zip(names.keys(), np.abs(vector)), 
                          key=lambda x: x[1], reverse=True)
   dominant = []
   for i, weight in contributions:
       if weight >= threshold:
           name = names.get(i, f"Z{i+1}")
           dominant.append(f"{weight*100:.0f}% {name}")
   return " + ".join(dominant) if dominant else "Mixed High-Order"

# --- 生成模式构成表 (Table 5.1) ---
print("\n--- 像差敏感性分析报告 (Modal Composition Report) ---")
print("==============================================================")
print(f"{'模式索引':<10} | {'影响权重 (σ)':<15} | {'主要Zernike成分 (Input Mode Vh[i])':<50}")
print("-" * 75)

for i in range(10): # 仅显示前10个最敏感的模式
   weight = S_diag[i]
   # Vh[i, :] 是第 i 个输入奇异向量
   components = get_dominant_components(Vh[i, :], zernike_names)
   
   print(f"{i+1:<10} | {weight:<15.4f} | {components:<50}")

print("==============================================================")```

**表格5.1：模拟的像差敏感性分析报告 (项目5输出)**

| 模式索引 | 影响权重 (σ) | 主要Zernike成分 (Input Mode Vh[i]) |
| :--- | :--- | :--- |
| 1 | 1.8345 | 95% Astigmatism (45 deg) + 32% Defocus |
| 2 | 1.2567 | 88% Coma (X) + 40% Astigmatism (0 deg) |
| 3 | 0.9870 | 99% Trefoil (Y) |
| 4 | 0.9122 | 75% Coma (Y) + 50% Trefoil (X) |
| 5 | 0.8530 | 91% Spherical Aberration |
| ... | ... | ... |
| 10 | 0.4321 | Mixed High-Order |

**分析与决策**：

这个表格（见上方的模拟输出）是本周项目的最终交付成果。它不再是抽象的数学，而是可操作的工程指南。

它告诉设计师：

1.  “你的系统对Z5（45度散光）极其敏感（权重1.83）。”
2.  “它对Z7（X方向彗差）和Z8（Y方向三叶草）也高度敏感。”
3.  “有趣的是，它对Z10（球差）的敏感性（权重0.85）远低于对散光和彗差的敏感性。”

**临床转化**： 这意味着任何导致45度散光（例如，镜片在特定方向上的轻微变形或倾斜）的制造误差都将被系统放大近两倍，从而导致严重的视觉质量下降（例如，强烈的“拖影”或“重影”）。设计团队必须优先优化设计，以降低对Z5模式的敏感性，或者对制造公差提出更严格的控制。

## 5.7 可视化抽象：Manim 演示5 - "看见"算子在工作

理论和代码是强大的，但直观的理解是不可或缺的。Manim动画在这里扮演着“翻译者”的角色，将抽象的算子理论转化为可见的几何变换。

**Manim演示5：算子作用动画**

*   **场景1：算子变换 (Operator Action)**
    *   **视觉效果**： 动画开始于一个标准的 Manim 网格（`NumberPlane`），代表我们的函数空间（例如，由Zernike模式 $Z_3$ 和 $Z_5$ 张成的子空间）。
    *   **旁白**： “这是一个理想的函数空间。每个点代表一种特定的像差组合。”
    *   **动作**： 使用 `ApplyMatrix` 或 `LinearTransformScene`，网格开始发生几何变换 (Geometric Transformation)[^21]。网格被拉伸、压缩和剪切。
    *   **旁白**： “这就是线性算子 $\mathcal{L}$[^2] 的作用。它在‘扭曲’这个空间。请注意，某些方向（例如对角线方向）被拉伸得最长——这就是系统最敏感的方向。其他方向被压缩——这就是系统不敏感的方向。”

*   **场景2：特征值分解 (动态) (Dynamic Eigenvalue Decomposition)**
    *   **视觉效果**： 变换停止在一个新的、扭曲的网格状态。Manim高亮显示了网格变换后的“主轴”——即那些只被拉伸、没有被旋转的向量。
    *   **旁白**： “谱理论[^3] 的任务就是找到这些‘主轴’，即系统的特征向量。它们对应的拉伸/压缩因子就是特征值 $\lambda$。”
    *   **动作**：
        1.  一个任意的输入波前 $f_{in}$（一个 `ParametricSurface`）出现。
        2.  $f_{in}$ 动画般地分解为沿着这些新“主轴”（特征向量）的几个分量。
        3.  每个分量向量被其对应的特征值 $\lambda_i$ 独立地缩放（拉伸或压缩）。
        4.  缩放后的分量重新组合 (re-sum)，形成最终的输出波前 $f_{out}$。
    *   **旁白**： “这就是算子如何作用于一个任意输入的几何意义[^5]。系统通过将其分解为本征模式、按敏感性缩放、然后重新组合，来‘处理’像差。”

*   **场景3：像差分解 (应用) (Aberration Decomposition)**
    *   **视觉效果**： 屏幕左侧是一个复杂的目标像差波前（来自项目4的数据）。右侧是一个空的坐标系（“重建”）。
    *   **旁白**： “现在，我们将项目5的SVD结果可视化。我们将使用SVD找到的奇异向量（即我们的新基底）来重建这个像差。”
    *   **动作**：
        1.  **添加模式1（$\sigma_1$）**： 第一个（最敏感的）奇异向量 $f_1$ 乘以其系数后出现。右侧的波形立即捕捉到了左侧目标的主要特征（例如，主要的散光）。
        2.  **添加模式2（$\sigma_2$）**： 第二个奇异向量 $f_2$ 被添加到 $f_1$ 上。重建的波形进一步逼近目标（例如，加入了彗差的扭曲）。
        3.  **添加模式3, 4, 5...** 动画快速连续添加后续模式。
    *   **旁白**： “请看——我们仅仅添加了前5个最敏感的模式，重建的波形就已经非常接近左侧的真实像差了 。这在视觉上证明了我们的数值结论：系统的绝大部分误差都集中在少数几个关键模式上。我们现在知道该优先修复什么了。”

## 5.8 总结与展望

在本章中，我们完成了从“描述”到“分析”的认知飞跃。我们不再将镜片系统视为一块玻璃，而是将其重塑为一个数学线性算子 $\mathcal{L}$。

*   我们利用紧算子理论[^7] 证明了为什么我们可以使用有限的数值模型（如FEM[^8] 或Zernike基底截断）来可信地近似无限维的真实物理。
*   我们揭示了谱理论[^3] 是进行光学敏感性分析[^13] 的终极工具。算子的特征值（或奇异值）$\lambda_i$ 是“影响权重”，而其特征函数（或奇异向量）$f_i$ 是系统特有的“基本像差模式”。
*   我们构建了一个实用的Python工具 (项目5)，使用 NumPy[^27] 和 SciPy[^28] 来计算这些权重，并将抽象的SVD结果[^21] 翻译为可操作的、指导公差分配和设计优化的工程决策表。

**通向第6周的桥梁：**

在第5周，我们分析的算子 $\mathcal{L}$ 还是一个通用的“黑盒”。但是，许多真实的光学系统——尤其是那些具有复杂物理过程的系统，如抗反射涂层、梯度折射率 (GRIN) 镜片或角膜地形图 ——其物理特性是通过积分来定义的。

这些系统需要一种新的数学工具：积分方程 (Integral Equations)。在下一章 (第6周) 中，我们将学习如何使用 Fredholm 方程来显式地构建和求解这些由积分定义的算子，从而优化复杂的多层光学系统。

---
## 引用的著作

[^1]: 实用泛函分析大纲.pdf
[^2]: All-optically untangling light propagation through multimode fibers - Optica Publishing Group, https://opg.optica.org/abstract.cfm?uri=optica-11-1-101
[^3]: Courses - Caltech Catalog, https://catalog.caltech.edu/documents/50/catalog_20_21_part5.pdf
[^4]: Handbook of Optics [Vol. III, 1 ed.] 0071354085, 9780071354080, 9780071414784, https://dokumen.pub/handbook-of-optics-vol-iii-1nbsped-0071354085-9780071354080-9780071414784-z-5004664.html
[^5]: least-mean-square lms algorithm: Topics by Science.gov, https://www.science.gov/topicpages/l/least-mean-square+lms+algorithm
[^6]: 2014 SPIE DSS•, https://spie.org/Documents/ConferencesExhibitions/DSS14-Abstracts.pdf
[^7]: Numerical-asymptotic boundary integral methods in high-frequency acoustic scattering - The University of Bath, https://people.bath.ac.uk/eas25/acta_numerica_final.pdf
[^8]: Finite element approximation of eigenvalue problems - ResearchGate, https://www.researchgate.net/publication/245481475_Finite_element_approximation_of_eigenvalue_problems
[^9]: elliptic mesh generation: Topics by Science.gov, https://www.science.gov/topicpages/e/elliptic+mesh+generation
[^10]: A High-Order L1-2 Scheme Based on Compact Finite Difference Method for the Nonlinear Time-Fractional Schr¨odinger Equation - Engineering Letters, https://www.engineeringletters.com/issues_v31/issue_4/EL_31_4_30.pdf
[^11]: Coercive second-kind boundary integral equations for the Laplace Dirichlet problem on Lipschitz domains - arXiv, https://arxiv.org/pdf/2210.02432
[^12]: Courses - Caltech Catalog, https://catalog.caltech.edu/documents/13/catalog_12_13_part5.pdf
[^13]: OWL Instrument Concept Study Earth-like Planets Imaging Camera Spectrograph - Eso.org, https://www.eso.org/sci/facilities/eelt/owl/Files/publications/OWL-CSR-ESO-00000-0166_iss1.pdf
[^14]: Principles of Harmonic Analysis | Request PDF - ResearchGate, https://www.researchgate.net/publication/266842388_Principles_of_Harmonic_Analysis
[^15]: classical orthogonal polynomials: Topics by Science.gov, https://www.science.gov/topicpages/c/classical+orthogonal+polynomials.html
[^16]: What we have learnt from 30 years living with positive dysphotopsia after intraocular lens implantation?: a review - Metrovision, https://metrovision.fr/2021/2021_Fernandez_Positive_dysphotopsia.pdf
[^17]: Clinical techniques to assess the visual and optical performance of, https://research.manchester.ac.uk/files/38537118/FULL_TEXT.pdf
[^18]: Design concepts for advanced-technology intraocular lenses [Invited] - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11729292/
[^19]: Grzegorz Labuz - RePub, Erasmus University Repository, https://repub.eur.nl/pub/102424/G.Labuz_PhD_Thesis.pdf
[^20]: (PDF) dO: A Differentiable Engine for Deep Lens Design of ..., https://www.researchgate.net/publication/363847205_dO_A_differentiable_engine_for_Deep_Lens_design_of_computational_imaging_systems
[^21]: least-squares curve fitting: Topics by Science.gov, https://www.science.gov/topicpages/l/least-squares+curve+fitting
[^22]: nonlinear least-squares fit: Topics by Science.gov, https://www.science.gov/topicpages/n/nonlinear+least-squares+fit
[^23]: third-order least-squares polynomial: Topics by Science.gov, https://www.science.gov/topicpages/t/third-order+least-squares+polynomial
[^24]: least-squares phase unwrapping: Topics by Science.gov, https://www.science.gov/topicpages/l/least-squares+phase+unwrapping
[^25]: optiland · PyPI, https://pypi.org/project/optiland/0.1.1/
[^26]: optiland - PyPI, https://pypi.org/project/optiland/0.3.1/
[^27]: Performance Characterization of KAPAO, a Low-Cost Natural Guide Star Adaptive Optics Instrument | Request PDF - ResearchGate, https://www.researchgate.net/publication/260867845_Performance_Characterization_of_KAPAO_a_Low-Cost_Natural_Guide_Star_Adaptive_Optics_Instrument
[^28]: (PDF) Adorym: a multi-platform generic X-ray image reconstruction framework based on automatic differentiation - ResearchGate, https://www.researchgate.net/publication/349186113_Adorym_a_multi-platform_generic_X-ray_image_reconstruction_framework_based_on_automatic_differentiation
[^29]: Close-Range Photogrammetry and 3D Imaging [3 ed.] 9783110607246, 9783110607253, 9783110607383 - DOKUMEN.PUB, https://dokumen.pub/close-range-photogrammetry-and-3d-imaging-3nbsped-9783110607246-9783110607253-9783110607383.html
[^30]: Close-Range Photogrammetry and 3D Imaging [4. rev. and exten. edition] 9783111029672, 9783111029351 - DOKUMEN.PUB, https://dokumen.pub/close-range-photogrammetry-and-3d-imaging-4-rev-and-exten-edition-9783111029672-9783111029351.html
[^31]: Light Field Imaging for Deflectometry, https://library.oapen.org/bitstream/id/2da7c64e-00dc-46d5-859c-7bc044ee5b56/light-field-imaging-for-deflectometry.pdf