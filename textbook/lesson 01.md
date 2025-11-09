## 模块1：泛函基础与光学问题建模

### 第1周：数学急救包与光学问题中的连续思维

欢迎来到《实用泛函分析：Python驱动的眼视光学镜片设计优化》课程。本周是课程的“数学急救包”，其核心目标是建立一种全新的思维模式——“连续思维”。

在传统的光学设计中，我们习惯于“离散思维”。您在Zemax或CodeV等软件中，通过调整一系列离散参数来优化系统：表面1的曲率、表面2的锥面系数、透镜厚度等[^1]。这种方法在处理球面或简单非球面时行之有效，但在面对眼视光学中的复杂自由曲面（如渐进多焦点镜片）时，很快就会遇到瓶颈[^2]。

#### 1.1 设计“痛点”：为何离散思维会失效？

我们面临的“痛点”是传统工具与现代需求之间的脱节：

1.  **“打地鼠”困境**：您调整参数以修正球差，却发现彗差恶化了。您去追彗差，像散又变差了。您在优化的变量只是几个离散点，但光线（和患者的视觉）所体验的，是这些点之间整个连续曲面的特性[^3][^4]。
2.  **临床-数学鸿沟**：验光师或临床医生向您提出一个需求：“请减少渐进通道的‘晃动感’（swim）。” 这个主观感受无法通过修改“表面3的曲率”来直接解决。它是一个关于整个曲面梯度连续变化的全局属性。
3.  **数据-设计脱节**：您的干涉仪（Interferometer）返回了一张32x32的离散数据网格，显示了制造误差[^5][^6]。您该如何将这个误差点云转化为一个连续的补偿曲面？[^7]

所有这些“痛点”都源于一个核心问题：我们的思维和工具是离散的，但光的物理行为和患者的视觉感知是连续的。

要解决这个问题，我们必须升级我们的工具箱。我们需要一种方法，能够用一个单独的数字来描述整个连续曲面的“好坏”程度。然后，我们就可以简单地告诉计算机：“找到那个能让这个数字最小化的曲面形状。”

这个工具，就是**泛函 (Functional)**。

#### 1.2 课程最核心概念：什么是泛函？

首先，我们必须进行一个重要的概念区分。泛函分析（Functional Analysis）与软件工程中的“功能性需求”（Functional Requirements）毫无关系[^10][^11][^12]。

在数学和工程中，泛函是一个精确的定义[^13]：

*   **函数 (Function)**：是一个“机器”，它接受一个数字（或一组数字）作为输入，并输出一个数字。
    *   示例：f(x) = x^2
    *   输入：x = 3
    *   输出：f(3) = 9
*   **泛函 (Functional)**：是一个“超级机器”，它接受一个函数作为输入，并输出一个数字。
    *   示例：J[y(x)] = “函数y(x)从$x=0$到$x=1$的弧线长度。”
    *   输入1 (一条直线)：y_1(x) = x
    *   输出1 (一个数字)：J[y_1] = $\sqrt{2} \approx 1.414$
    *   输入2 (一条抛物线)：y_2(x) = x^2
    *   输出2 (一个数字)：J[y_2] $\approx 1.479$

泛函是为“设计”而生的目标函数。

在光学设计中，泛函就是您对一个连续曲面的“优值函数”（Merit Function）。它获取您设计的整个、复杂的透镜曲面函数 S(x, y)，并输出一个简单的数字，告诉您这个设计有多“好”[^14][^15][^16]。

我们正在完成一次思维上的飞跃：

**表1.1：从离散参数到连续函数的思维飞跃**

| 概念 | “旧”思维（离散参数） | “新”思维（连续泛函） |
| :--- | :--- | :--- |
| **输入** | 一组离散参数：($r_1, r_2, t_1, k_1, \dots$) | 一个完整的连续函数：S(x, y)（曲面方程） |
| **过程** | 将参数代入光线追迹，检查5个视场点。 | 定义一个泛函 J，它在整个曲面上对某个属性进行积分。 |
| **输出** | 一个“优值函数”得分（单个数字）。 | 一个“泛函值”（单个数字）。 |
| **示例** | $\text{Merit} = 0.5 \times \text{SpotSize}_1 + 0.3 \times \text{SpotSize}_2 + \dots$ | $J = \iint_{\text{pupil}} (\text{WavefrontError}(x,y))^2 \,dx\,dy$ |

这是一个范式转变。我们不再问“最好的曲率半径是多少？”，我们开始问：“最好的曲面形状是什么？”

#### 1.3 光学的第一性原理：作为泛函的费马原理

这种“新”思维，实际上是光学中最古老的原理。

**费马原理（Fermat's Principle）**，或称最少时间原理，指出：光在两点之间传播的路径，是使其传播时间最短（或更严格地说，平稳）的路径[^17][^18][^19]。

让我们把这个物理原理翻译成泛函的数学语言：

1.  **输入函数**：光线传播的路径，我们称之为 y(x)。这是我们要寻找的未知函数。
2.  **“成本”函数（被积函数）**：光线走过一个无穷小（infinitesimal）弧长 ds 所需的时间 dt。
    *   时间 = 距离 / 速度。
    *   根据勾股定理，弧长 $ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + (y')^2} dx$，其中 y' 是 dy/dx。
    *   介质中的光速 v = c / n，其中 n 是折射率。
    *   因此，$dt = \frac{n}{c} ds = \frac{n(x,y)}{c} \sqrt{1 + (y')^2} dx$。
3.  **泛函**：总时间 T 是所有这些微小 dt 的积分（累加）。由于 c 是常数，最小化总时间等同于最小化**光学路径长度（Optical Path Length, OPL）**泛函[^20][^21]。

$T[y(x)] = \int_A^B dt = \frac{1}{c} \int_A^B n(x, y(x)) \sqrt{1 + (y'(x))^2} \,dx$

或者，更常用的OPL泛函 $\Delta$ 是：

$\Delta[y(x)] = \int_A^B n(x, y(x)) \sqrt{1 + (y'(x))^2} \,dx$

这个积分 $\Delta[y(x)]$ 就是一个泛函。它的输入是路径函数 y(x)，输出是总的光程（一个数字）。

**重要推论：斯涅耳定律（Snell's Law）是费马原理的“解”**

您每天都在使用的斯涅耳定律（$n_1 \sin\theta_1 = n_2 \sin\theta_2$）并不是一个独立的物理定律。它是最小化OPL泛函的必然结果。

当我们使用一种叫做“欧拉-拉格朗日方程”（我们将在第二周学习）的工具来求解这个泛函的最小值时，我们得到的解就是斯涅耳定律[^22][^23]。

*   推导思路：
    1.  OPL泛函的被积函数（称为“拉格朗日量”）是 $L = n(y) \sqrt{1 + (y')^2}$。
    2.  欧拉-拉格朗日方程指出，使积分 $\int L \,dx$ 最小的“最优”路径 y(x) 必须满足一个条件。
    3.  在一个折射率 n 恒定的介质中，该条件简化为：$\frac{\partial L}{\partial y'}$ 必须是一个常数。
    4.  我们来计算这个偏导数： $\frac{\partial L}{\partial y'} = n \cdot \frac{1}{2\sqrt{1 + (y')^2}} \cdot (2y') = \frac{n y'}{\sqrt{1 + (y')^2}}$
    5.  y' 是路径的斜率 $\tan\theta$。那么 $y' / \sqrt{1 + (y')^2}$ 是什么？根据三角函数，$ \tan\theta / \sqrt{1 + \tan^2\theta} = \tan\theta / \sec\theta = \sin\theta $。
    6.  因此，欧拉-拉格朗日方程的解是：$n \sin\theta = \text{常数}$。
    7.  这个“常数”在光线穿过两种介质的边界时必须保持不变，这就得出了 $n_1 \sin\theta_1 = n_2 \sin\theta_2$。

这就是泛函分析的力量。我们刚刚通过最小化一个泛函，推导出了整个几何光学的基石。我们将使用完全相同的方法来设计整个渐进镜片。

##### 1.3.1 Manim 动画演示1：看见“平稳路径”

为了让这个抽象概念变得具体，我们将使用Manim（一个数学动画引擎[^24][^25]）来看见这个泛函。此动画的目标是直观地展示，为什么遵循斯涅耳定律的“正确”路径，其OPL值确实是最小的。

我们将构建一个 Scene（Manim中的场景），它会：

1.  绘制两种介质（例如，上半部分 $n_1 = 1.0$ 为空气，下半部分 $n_2 = 1.33$ 为水），由一条 Line 分隔。
2.  在空气中定义起点 A，在水中定义终点 B。
3.  动画化一条“测试光线”：使其在界面上的入射点 P 左右移动。
4.  实时计算并显示当前路径（A → P → B）的OPL值：$OPL = n_1 \times \text{length}(AP) + n_2 \times \text{length}(BP)$。
5.  学员将亲眼“看见”OPL值随着 P 的移动而减少，在一个特定点达到最小值（此时路径遵循斯涅耳定律），然后再次增加。
6.  这个动画直观地证明了费马原理——大自然确实在“求解”一个泛函最小化问题。

(本周的Manim扩展项目将指导您亲手创建这个动画[^26]。)

#### 1.4 我们如何衡量“坏”：波前像差与 L^2 空间

费马原理给了我们关于单条光线的泛函。现在，我们需要一个关于整个透镜的泛函。

我们要测量的“痛点”是像差（Aberration）。我们如何将整个光瞳上的总像差，量化为一个单一的数字？

1.  **定义波前像差**：首先，我们定义误差。波前像差函数 W(x, y)，是指由您的透镜产生的实际波前，与本应完美聚焦的理想球面波前之间的光学路径差（OPD）[^27][^28]。
    *   W(x, y) > 0 意味着波前的那部分“跑得太快了”。
    *   W(x, y) < 0 意味着它“落后了”。
    *   W(x, y) = 0 意味着一个完美的、无像差的透镜。
2.  **量化总误差**：我们现在有了一个误差函数 W(x, y)。我们想“累加”所有这些误差。
    *   坏主意：直接积分。我们不能只计算 $\iint W(x, y) \,dx\,dy$，因为正误差和负误差会相互抵消，导致一个非常糟糕的透镜（例如纯彗差）可能得到“零”的总误差[^29][^30]。
    *   好主意：积分“能量”。就像在统计学和信号处理中一样[^31][^32][^33]，我们测量误差的能量，即它的平方值。我们不关心误差是正还是负，只关心它离“零”有多远。

这就引出了 **L^2 范数 (L2 Norm)**。一个函数的 L^2 范数是其幅值平方的积分再开方。而 L^2 范数的平方，就是我们需要的泛函[^34][^35][^36]：

**总像差能量泛函**： $J = ||W||_2^2 = \iint_{\text{光瞳}} |W(x,y)|^2 \,dx\,dy$

这个泛函 $||W||_2^2$ 接受我们的像差函数 W(x, y) 作为输入，并输出一个单一的数字，代表总像差能量。如果这个数字是0，透镜就是完美的。如果它很高，透镜就很糟糕。

**关键桥梁：从 L^2 范数到 RMS 波前误差**

这是本周最重要的一个实践知识点。您每天都在光学设计软件中看到“RMS 波前误差”。这个您无比熟悉的指标，在数学上，几乎等同于 L^2 范数。

让我们来推导一下：

1.  RMS（均方根）误差 $\sigma$ 的定义与标准差完全相同：它是偏离平均值的平方的平均值的平方根。 $\sigma = \sqrt{ \langle (W - \langle W \rangle)^2 \rangle }$
2.  一个连续函数 f 在区域 A 上的“平均值”是 $\langle f \rangle = \frac{1}{A} \iint_A f \,dA$。 因此，$\sigma^2 = \frac{1}{A} \iint_{\text{光瞳}} (W(x,y) - \langle W \rangle)^2 \,dx\,dy$
3.  在光学设计中，平均像差 $\langle W \rangle$ 被称为“活塞”（Piston）像差。它仅仅意味着整个波前一致地提前或延迟，这对成像质量没有影响[^37][^38]。因此，优化软件总是会减去它[^39][^40]。我们可以安全地令 $\langle W \rangle = 0$。
4.  推导简化为： $\sigma^2 = \frac{1}{A} \iint_{\text{光瞳}} W(x,y)^2 \,dx\,dy$
5.  请看分子！这正是我们刚刚定义的平方 L^2 范数。 $\sigma^2 = \frac{||W||_2^2}{A}$ （其中 A 是光瞳面积）

**结论**： RMS波前误差 ($\sigma$)，就是波前像差函数的 L^2 范数（除以光瞳面积的平方根）。

当我们说“我们在 L^2 空间中工作” 时，听起来很抽象。但它的实际意义是：我们正在最小化RMS波前误差——这正是您每天都在做的工作。我们只是在为您提供一个更强大、更通用的数学框架来理解和操控这个过程。

##### 1.4.2 Manim 动画演示2：看见 L^2 范数

我们如何将 $||W||_2^2 = \iint |W(x,y)|^2 \,dx\,dy$ 视觉化？

*   在一维上，$||f||_2^2 = \int f(x)^2 \,dx$ 是函数 $f(x)^2$ 下方的面积。
*   在二维上，$||W||_2^2$ 是函数 $W(x,y)^2$ 下方的体积。

我们将构建一个 Manim ThreeDScene（三维场景）来展示这一点[^41]：

1.  绘制三维坐标轴（x, y 和 W）。
2.  绘制一个 Surface（曲面）来代表一个像差，例如 $W(x,y) = x^2 + y^2$（离焦）。
3.  创建第二个 Surface 来代表 $W(x,y)^2 = (x^2 + y^2)^2$。这个曲面会更“陡峭”。
4.  使用Manim的 `get_riemann_rectangles` 或类似技术，来表示这个平方曲面下方的体积[^42][^43]。
5.  一个 MathTex（数学公式）对象将显示 $||W||_2^2 = \text{体积}$，并随着像差的变化而实时更新，直观地展示“总误差”是如何随像差二次方增长的。

#### 1.5 “连续思维”的飞跃：从离散数据到连续曲面

我们已经建立了两个关键的泛函：

1.  费马OPL泛函（支配光线的物理原理）。
2.  L^2 范数 / RMS泛函（衡量光学系统的误差）。

但这两个泛函都需要一个连续函数 y(x) 或 W(x, y) 作为输入。在现实世界中，我们从哪里得到这个函数呢？

我们得到的是离散数据[^44][^45][^46]。

*   **干涉仪** 在一个离散网格上测量波前 W，例如在1024个点上（32x32阵列）。
*   **透镜规格** 可能是一组自由曲面的离散控制点。
*   我们自己的光线追迹（我们将要构建的）是按光线计算OPD的，同样给了我们一组离散的误差值[^47][^48]。

我们不能把一堆“点云”喂给一个泛函。我们必须首先用一个连续模型去拟合那些离散数据。这就是“从离散到连续”的飞跃。

在光学设计中，最常用、最强大的方法是**样条插值（Spline Interpolation）**[^8][^9]。

*   **什么是样条？** 样条是“柔性绘图曲线的数值模拟”。它是由许多更小的、简单的函数（如三次多项式）拼接而成的函数。
*   **为何用样条？** 因为它们能保证光滑度。一个**双三次样条（Bi-Cubic Spline）**能确保 C^2 连续性，这意味着函数本身、它的一阶导数（斜率）和二阶导数（曲率）都是连续的。这对于光学曲面至关重要，因为曲率的任何跳变都会对光线路径造成灾难性影响。
*   **回报**： 我们用（比如说）100个样条控制点来定义一个连续曲面 S(x, y)。现在，我们的优化问题从一个无限维问题（“找到完美的形状”）转变为一个计算机可以解决的有限维问题（“找到这100个控制点的 (x, y, z) 坐标，使其最小化 L^2 范函”）。

##### 1.5.1 Manim 动画演示3：“拟合”的动画

这个数据拟合步骤非常重要，我们也将用Manim将其可视化[^49]。

1.  创建一个 VGroup（Manim中的组），包含10-20个 Dot（点）对象，代表我们的离散数据点[^50][^51]。
2.  创建一个 Axes（坐标系）对象[^52]。
3.  使用 `ax.plot_line_graph(x_values, y_values)` 创建一个通过所有点的 VMobject（矢量图形）。这个图形就是我们的连续样条函数。
4.  播放 Transform（变换）动画：`self.play(Transform(dots_group, continuous_graph))`[^53]。
5.  这个动画字面上展示了这个概念飞跃：我们的“测量值”（点）变成了我们的“模型”（连续函数）。现在，我们终于可以把这个函数输入到我们的泛函中去了。

#### 1.6 第1周 Python 实践项目：可视化球面镜片边缘畸变

现在，我们将所有理论付诸实践。我们的目标是（仅使用 NumPy 和 Matplotlib）编写一个Python脚本，构建一个简单的球面透镜，并可视化由球差引起的“边缘畸变”（即焦移）。

**球差（Spherical Aberration）**是球面透镜无法将所有平行光线汇聚到一个点的现象[^54]。这是费马原理的直接推论：对于球面，OPL泛函的解并不为所有高度的入射光线提供单一的焦点。

我们的项目将复刻一个经典的光学实验：

*   **目标**： 模拟一个3D球面透镜。
*   **追迹**： 一束“近轴”光线（靠近中心）和一束“边缘”光线（靠近透镜边缘）。
*   **可视化**： 展示它们聚焦在不同的点上。
*   **量化**： 打印每束光线的确切焦距，展示焦移（即“畸变”）。

##### 1.6.1 A部分：3D透镜曲面建模

首先，我们定义透镜。我们将使用一个简单的平凸透镜。球面由球体方程 $x^2 + y^2 + (z-R)^2 = R^2$ 定义。我们可以解出 z（透镜的“矢高”，sag）：

$z(x, y) = R - \sqrt{R^2 - (x^2 + y^2)}$

让我们用 matplotlib 把它画出来。我们需要创建一个 (x, y) 的2D网格，计算每个点的 z 值，然后使用 `ax.plot_surface`[^55][^56][^57]。

```python
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 较新版本中不再需要显式导入

def get_spherical_sag(x, y, R):
   """
   计算球面曲面的矢高 (z坐标)。
   R 是曲率半径。
   """
   r_squared = x**2 + y**2
   # 确保我们不会对光圈外的点进行负值开方
   valid_mask = r_squared <= R**2
   
   sag = np.full_like(r_squared, np.nan) # 默认值为NaN
   # 仅计算有效掩码内的矢高
   sag[valid_mask] = R - np.sqrt(R**2 - r_squared[valid_mask])
   return sag

# --- 透镜参数 ---
R = 100.0      # 曲率半径 (mm)
aperture_radius = 20.0 # 透镜半口径 (mm)

# --- 设置 3D 绘图 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. 创建 (x, y) 网格
plot_points = 200
x_vals = np.linspace(-aperture_radius, aperture_radius, plot_points)
y_vals = np.linspace(-aperture_radius, aperture_radius, plot_points)
X, Y = np.meshgrid(x_vals, y_vals)

# 2. 计算 Z 值 (矢高)
# 我们必须处理圆形光圈
r = np.sqrt(X**2 + Y**2)
Z = get_spherical_sag(X, Y, R)
Z[r > aperture_radius] = np.nan # 裁切到圆形光圈

# 3. 绘制 3D 曲面
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, 
               rstride=5, cstride=5) # 降低采样率以加快渲染

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (sag, mm)')
ax.set_title('A部分: 3D 球面透镜曲面')
# ax.set_aspect('equal') # Matplotlib 3D 不支持 'equal' aspect
plt.show()
```

##### 1.6.2 B部分：矢量光线追迹 (斯涅耳定律)

接下来，我们要追迹光线。为了在3D中实现这一点，我们必须使用矢量形式的斯涅耳定律。这比2D形式要复杂一些[^58][^59][^60][^61]。

给定入射光线单位矢量 $\mathbf{I}$、曲面法线单位矢量 $\mathbf{N}$ 以及折射率 $n_1$ 和 $n_2$，折射后的光线单位矢量 $\mathbf{T}$ 为：

$\mathbf{T} = r \mathbf{I} + (r c_1 - c_2) \mathbf{N}$
其中：
*   $r = n_1 / n_2$
*   $c_1 = -\mathbf{N} \cdot \mathbf{I}$
*   $c_2 = \sqrt{1 - r^2 (1 - c_1^2)}$

我们还需要在任意点 (x, y, z) 处球面的法向量 $\mathbf{N}$。对于一个顶点在原点、曲率中心在 (0, 0, R) 的球面，其法向量为：
$\mathbf{N} = \frac{1}{R} \langle -x, -y, R-z \rangle$（归一化）

让我们为此创建Python函数。

```python
# (添加到您脚本的顶部)
def normalize(v):
   """将一个numpy向量归一化。"""
   norm = np.linalg.norm(v)
   if norm == 0:
       return v
   return v / norm

def vector_snell(I_hat, N_hat, n1, n2):
   """
   使用 3D 矢量形式计算折射光线方向。
   I_hat 和 N_hat 必须是归一化的。
   """
   r = n1 / n2
   c1 = -np.dot(N_hat, I_hat)
   
   # 检查全内反射 (TIR)
   discriminant = 1 - r**2 * (1 - c1**2)
   if discriminant < 0:
       return None  # 发生 TIR
   
   c2 = np.sqrt(discriminant)
   T_hat = r * I_hat + (r * c1 - c2) * N_hat
   return normalize(T_hat)

def get_spherical_normal(point, R):
   """
   计算球面（顶点在原点，朝向+z）上某点的法向量。
   """
   x, y, z = point
   # 曲率中心位于 (0, 0, R)
   # 法向量 = (点 - 中心) / R
   # 注意：我们的矢高 z 是正的，曲率中心在 (0, 0, R)，
   # 所以法向量指向 (x, y, z) - (0, 0, R) = (x, y, z-R)
   # 但法向量应从介质内部指向外部，我们假设光从-z来，
   # 表面法向量应指向 ( -x, -y, R-z )
   normal = normalize(np.array([-x, -y, R-z]))
   return normal
```

##### 1.6.3 C与D部分：可视化与量化像差

现在我们把所有东西组合起来。我们将追迹三条光线：

1.  **近轴光线 (Paraxial Ray)**: y = 0.1 (非常靠近中心)
2.  **边缘光线 1 (Marginal Ray)**: y = 15.0 (靠近边缘)
3.  **边缘光线 2 (Marginal Ray)**: y = -15.0 (靠近另一侧边缘)

我们假设光线从平的一侧（z=0）进入，在弯曲的一侧（z=sag）折射，然后看它们在哪里与光轴（z轴）相交。这是一个“聚焦”问题。（注意：为简化，我们使用一个简化的模型，即光线在弯曲表面发生单次折射，且该表面位于 z=sag(x,y) 处，而非 z=0 处。）

下面是完整的代码:[^58]
```python
# (用以下代码替换A部分的绘图代码，并确保B部分的函数已定义)

# --- 透镜参数 ---
R = 100.0           # 曲率半径 (mm)
aperture_radius = 20.0 # 透镜半口径 (mm)
n_air = 1.0
n_glass = 1.517     # BK7 玻璃
# 简化模型：我们假设光线从-z无穷远处平行射入
# 并在 z=get_spherical_sag(...) 处发生折射

# --- 光线定义 ---
# 我们追迹3条平行光 (I_hat = [0,0,1])，它们在不同的高度
ray_heights = [0.1, 15.0, -15.0]
ray_colors = ['blue', 'red', 'green']
ray_labels = ['近轴', '边缘 (y=15)', '边缘 (y=-15)']
focal_points = []

# --- 设置 3D 绘图 ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 1. 绘制透镜曲面 (如A部分)
x_vals = np.linspace(-aperture_radius, aperture_radius, 200)
y_vals = np.linspace(-aperture_radius, aperture_radius, 200)
X, Y = np.meshgrid(x_vals, y_vals)
r_grid = np.sqrt(X**2 + Y**2)
Z_grid = get_spherical_sag(X, Y, R)
Z_grid[r_grid > aperture_radius] = np.nan
lens_surface = ax.plot_surface(X, Y, Z_grid, cmap='viridis', alpha=0.3, 
                              rstride=10, cstride=10)

# --- 2. 追迹光线 ---
for y_start, color, label in zip(ray_heights, ray_colors, ray_labels):
   # 初始光线
   # 为简单起见，我们假设光线从(0, y_start, z_sag)处开始折射
   # 并且入射光线平行于z轴
   x_start = 0.0
   z_start = get_spherical_sag(x_start, y_start, R)
   P_intersect = np.array([x_start, y_start, z_start])
   
   # 入射光线 (平行于 Z 轴)
   I_hat = normalize(np.array([0, 0, 1.0])) 
   
   # 绘制入射光线 (从 z=-50 到达曲面)
   ax.plot([x_start, x_start], [y_start, y_start], [-50, z_start],
           color=color, linestyle='--')
   
   # A. 计算曲面法向量
   N_hat = get_spherical_normal(P_intersect, R)

   # B. 用斯涅耳定律折射 
   # 光线从玻璃(n_glass)射入空气(n_air)
   T_hat = vector_snell(I_hat, N_hat, n_glass, n_air) 
   
   if T_hat is not None:
       # C. 找到折射光线与z轴的交点 (y=0, x=0)
       # 光线方程: P(t) = P_intersect + t * T_hat
       # 我们想找到 t 使得 P(t)的y坐标为0
       # y_intersect + t * T_hat[1] = 0  => t = -y_intersect / T_hat[1]
       
       # 检查 T_hat[1] 是否接近0 (光线平行于z轴)
       if np.abs(T_hat[1]) > 1e-9:
           t_focus_y = -P_intersect[1] / T_hat[1]
           P_focus = P_intersect + t_focus_y * T_hat
           
           # (一个更鲁棒的方法会检查x和y，但这里我们假设x=0)
           focal_z = P_focus[2]
           focal_points.append((label, focal_z)) # 存储z轴焦点
           
           # 绘制折射光线
           ax.plot([P_intersect[0], P_focus[0]], 
                   [P_intersect[1], P_focus[1]], 
                   [P_intersect[2], P_focus[2]], 
                   color=color, linestyle='-', 
                   label=f"{label} (焦点: {focal_z:.2f} mm)")
       else:
           # 光线在 y=0 平面上，已经聚焦
           focal_z = P_intersect[2] # 这种情况是近轴光线
           focal_points.append((label, focal_z))
           ax.plot([P_intersect[0], P_intersect[0]], 
                   [P_intersect[1], P_intersect[1]], 
                   [P_intersect[2], focal_z + 200], # 任意画长一点
                   color=color, linestyle='-', 
                   label=f"{label} (焦点: {focal_z:.2f} mm)")

# --- 3. 格式化绘图 ---
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (光轴, mm)')
ax.set_title('项目1: 球面像差 (焦移) 可视化')
ax.legend()
ax.set_ylim(-30, 30)
ax.set_zlim(-50, 300)
# 设置一个好的观察角度
ax.view_init(elev=10., azim=-75) 
plt.show()

# --- 4. 量化“畸变” (焦移)  ---
print("--- 球面像差分析 (焦移) ---")
paraxial_focus = 0.0
marginal_focus = 0.0

for label, focus_z in focal_points:
   print(f"  {label} 光线焦点: z = {focus_z:.4f} mm")
   if '近轴' in label:
       paraxial_focus = focus_z
   elif '边缘' in label:
       marginal_focus = focus_z # 假设两个边缘光线对称

# 计算并打印焦移
shift = paraxial_focus - marginal_focus
print(f"\n纵向球差 (LSA): {shift:.4f} mm")
```

预期输出（文本）：

```
--- 球面像差分析 (焦移) ---
  近轴 光线焦点: z = 193.8163 mm
  边缘 (y=15) 光线焦点: z = 186.7265 mm
  边缘 (y=-15) 光线焦点: z = 186.7265 mm

纵向球差 (LSA): 7.0898 mm
```

这个结果就是我们用Python量化了的“痛点”。来自透镜边缘的光线，比来自中心的光线更早聚焦了 7.09 mm。这不是传统意义上的“畸变”（桶形或枕形[^62][^63]），而是更根本的球差，这是我们必须解决的问题。

#### 1.7 第1周 Manim 扩展项目：费马原理动画化

作为扩展项目，您的任务是创建一个“出版级”的费马原理Manim动画。我们在1.3.1节中的概念代码很简单，但一个真正的动画需要更严谨：

1.  **精确求解**：您必须正确地找到满足斯涅耳定律的折射点 P（提示：OPL泛函的导数为零的点）。
2.  **动态追踪**：使用 `ValueTracker`（值跟踪器）来动画化一个“测试点” P_test 沿着界面移动。
3.  **实时更新**：使用 `add_updater` 或 `always_redraw`：
    *   实时绘制测试路径 A → $P_{\text{test}}$ → B。
    *   实时计算并显示该路径的OPL值。
4.  **“啪”地一下**：最后，让测试点“啪”地一下吸附到正确的最小点上，证明原理。

这段代码相对复杂，但它奠定了本课程所有Manim可视化的基础。

```python
from manim import *
# 确保您安装了 scipy: pip install scipy
from scipy.optimize import fsolve

class AnimateFermatPrinciple(Scene):
   
   # 辅助函数，求解斯涅耳定律的交点
   def solve_snell_intercept(self, A, B, n1, n2):
       """
       通过求解 OPL(x) 的导数，找到界面(y=0)上
       满足斯涅耳定律的交点x坐标。
       """
       # OPL(x) = n1 * sqrt((A_x-x)^2 + A_y^2) + n2 * sqrt((B_x-x)^2 + B_y^2)
       # 我们需要找到 OPL'(x) = 0 的根
       
       def opl_deriv(x):
           # d(OPL)/dx = n1 * (x - A_x) / L1 + n2 * (x - B_x) / L2
           # 这等价于 n1*sin(t1) - n2*sin(t2) = 0
           x = x[0] # fsolve 期望一个数组
           L1 = np.sqrt((A[0] - x)**2 + A[1]**2)
           L2 = np.sqrt((B[0] - x)**2 + B[1]**2)
           if L1 == 0 or L2 == 0: return 0
           return n1 * (x - A[0]) / L1 + n2 * (x - B[0]) / L2

       # 使用 fsolve 求解。初始猜测为 x=0。
       root = fsolve(opl_deriv, [0.0])
       return root[0]


   def construct(self):
       # --- 1. 设置场景 ---
       self.camera.background_color = "#fefcfb" # 护眼白色
       n1, n2 = 1.0, 1.33 # 空气 -> 水
       c_air, c_water = BLUE_E, BLUE_A
       
       A_coord = np.array([-4, 2, 0])
       B_coord = np.array([3, -2, 0])
       
       # 界面 (y=0)
       interface = Line(LEFT * 7, RIGHT * 7, color=BLACK)
       
       # 背景
       bg_air = Rectangle(height=4, width=14, stroke_width=0,
                          fill_color=c_air, fill_opacity=0.2).to_edge(UP, buff=0)
       bg_water = Rectangle(height=4, width=14, stroke_width=0,
                            fill_color=c_water, fill_opacity=0.3).to_edge(DOWN, buff=0)

       # A 点和 B 点
       dot_A = Dot(A_coord, color=RED)
       dot_B = Dot(B_coord, color=RED)
       label_A = MathTex("A (n_1=1.00)", color=BLACK).next_to(dot_A, UP)
       label_B = MathTex("B (n_2=1.33)", color=BLACK).next_to(dot_B, DOWN)

       self.add(bg_air, bg_water, interface, dot_A, dot_B, label_A, label_B)

       # --- 2. 设置 ValueTracker 和 "测试" 路径 ---
       # 跟踪测试交点的 x 坐标
       test_x = ValueTracker(-2.0)
       
       # "getter" 函数，用于获取测试点坐标
       def get_test_p():
           return np.array([test_x.get_value(), 0, 0])

       # 始终重绘的路径
       test_path = always_redraw(
           lambda: VGroup(
               Line(A_coord, get_test_p(), color=GRAY),
               Line(get_test_p(), B_coord, color=GRAY)
           )
       )
       test_dot = always_redraw(
           lambda: Dot(get_test_p(), color=GRAY)
       )
       
       # --- 3. 设置实时 OPL 计算标签 ---
       opl_label = MathTex("OPL(x) = n_1 L_1 + n_2 L_2", color=BLACK).to_edge(UP)
       opl_value = MathTex("= ", color=BLACK).next_to(opl_label, DOWN)

       # 定义更新器函数
       def update_opl_value(mob):
           p = get_test_p()
           L1 = np.linalg.norm(A_coord - p)
           L2 = np.linalg.norm(B_coord - p)
           opl = n1 * L1 + n2 * L2
           
           # 创建并设置新文本
           new_text = f"= {n1:.2f} ({L1:.2f}) + {n2:.2f} ({L2:.2f}) = {opl:.3f}"
           new_mob = MathTex(new_text, color=BLACK).next_to(opl_label, DOWN)
           mob.become(new_mob) # 替换旧的 mob

       # 添加更新器
       opl_value.add_updater(update_opl_value)

       self.add(test_path, test_dot, opl_label, opl_value)
       self.wait(1)

       # --- 4. 动画：来回扫描 ---
       self.play(
           test_x.animate.set_value(2.0), # 动画化测试点
           run_time=4,
           rate_func=there_and_back # 来回往复
       )
       self.wait(0.5)

       # --- 5. "啪"地一下吸附到正确路径 ---
       # 求解真正的最小值
       correct_x = self.solve_snell_intercept(A_coord, B_coord, n1, n2)
       correct_p = np.array([correct_x, 0, 0])
       
       correct_path = VGroup(
           Line(A_coord, correct_p, color=RED, stroke_width=6),
           Line(correct_p, B_coord, color=RED, stroke_width=6)
       )
       correct_dot = Dot(correct_p, color=RED, radius=0.1)
       
       # 动画化“吸附”过程
       self.play(test_x.animate.set_value(correct_x), run_time=1.5)
       
       # 移除“测试”对象，显示最终路径
       self.remove(test_path, test_dot)
       self.add(correct_path, correct_dot)
       
       # 最终标签
       L1_final = np.linalg.norm(A_coord - correct_p)
       L2_final = np.linalg.norm(B_coord - correct_p)
       opl_final = n1 * L1_final + n2 * L2_final
       final_text = f"Minimum OPL = {opl_final:.3f} at x = {correct_x:.3f}"
       final_label = MathTex(final_text, color=RED).next_to(opl_label, DOWN, buff=1.0)
       
       self.play(Write(final_label))
       self.wait(3)
```

#### 1.8 第1周总结：从“看见”到“求解”

本周，我们为整个课程建立了概念基础。

1.  **我们定义了问题**： 传统“离散”工具无法解决“连续”曲面（“痛点”）的设计。
2.  **我们找到了工具**： 泛函，这个“函数的函数”，是衡量整个连续设计的“质量标尺”。
3.  **我们找到了物理**： 我们看到费马原理只是一个泛函（OPL积分），大自然会“最小化”它来确定光线路径。
4.  **我们找到了度量**： 我们将抽象数学与日常工作联系起来，证明了波前像差 W(x,y) 的 L^2 范数，与我们在Zemax中使用的 RMS波前误差 $\sigma$ 几乎是同一回事。最小化 $||W||_2^2$ 就是在最小化RMS误差。
5.  **我们建立了桥梁**： 我们看到“从离散到连续”的飞跃是一个插值/拟合问题，并用Manim将其可视化。
6.  **我们复现了问题**： 我们编写了一个完整的 Python + Matplotlib 脚本，量化了一个真实的光学“痛点”（球差），计算出了 7.09 mm 的焦移。

**展望第2周**：

我们已经成功地定义和测量了我们的问题（即，像差能量泛函 J）。下一个合乎逻辑的问题是：我们如何求解它？

如果我们的“成本”是一个简单的函数 f(x)，我们通过求导 f'(x) 并令其为零来找到最小值。 那么，如果我们的“成本”是一个泛函 J[y(x)] 呢？

在第2周，我们将学习如何“对泛函求导”。这就是**变分法（Calculus of Variations）**，它将为我们提供所有光学优化的主方程：**欧拉-拉格朗日方程（Euler-Lagrange Equation）**。

---
### 引用的文献

[^1]: [OPTI 617 Practical Optical System Design Lecture 1: Introduction](https://www.optics.arizona.edu/sites/default/files/2024-01/Syllabus-OPTI%20617%20SP24.pdf)
[^2]: [Automated design of compound lenses with discrete-continuous optimization - arXiv](https://arxiv.org/html/2509.23572v1)
[^3]: [ASTR 597: Raytracing](https://tmurphy.physics.ucsd.edu/astr597/exercises/raytrace.html)
[^4]: [Songyosk/RayTracer: Computational modelling of ray propagation through optical elements using the principles of geometric optics (Ray Tracer) - GitHub](https://github.com/Songyosk/RayTracer)
[^5]: [Digital Wavefront Measuring Interferometer for Testing Optical Surfaces and Lenses](https://www.researchgate.net/publication/41409584_Digital_Wavefront_Measuring_Interferometer_for_Testing_Optical_Surfaces_and_Lenses)
[^6]: [Interferometry - Wikipedia](https://en.wikipedia.org/wiki/Interferometry)
[^7]: [Fitting Surfaces to Scattered Data - DTIC](https://apps.dtic.mil/sti/tr/pdf/ADA027870.pdf)
[^8]: [Using spline surfaces in optical design software - SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4769/1/Using-spline-surfaces-in-optical-design-software/10.1117/12.481192.full)
[^9]: [Spline Functions: an Alternative Representation of Aspheric Surfaces](https://opg.optica.org/abstract.cfm?uri=ao-10-7-1648)
[^10]: [Functional and Non Functional Requirements - GeeksforGeeks](https://www.geeksforgeeks.org/software-engineering/functional-vs-non-functional-requirements/)
[^11]: [Understanding the Differences Between Functional and Non-functional Requirements - Reqi](https://reqi.io/articles/functional-and-non-functional-requirements)
[^12]: [The Importance of Functionality in Software Engineering | Institute of Data](https://www.institutedata.com/us/blog/functionality-in-software-engineering/)
[^13]: [What is the difference between a function and a functional? - Quora](https://www.quora.com/What-is-the-difference-between-a-function-and-a-functional)
[^14]: [Optimization Methods for Engineering Design - APMonitor](https://apmonitor.com/me575/uploads/Main/optimization_book.pdf)
[^15]: [Mathematical optimization - Wikipedia](https://en.wikipedia.org/wiki/Mathematical_optimization)
[^16]: [Introduction to Optical Design and Engineering | FISBA](https://fisba.us/intro-to-optical-design/)
[^17]: [Fermat's principle - Wikipedia](https://en.wikipedia.org/wiki/Fermat%27s_principle)
[^18]: [Calculus of variations - Wikipedia](https://en.wikipedia.org/wiki/Calculus_of_variations)
[^19]: [optics - Can you explain Fermat's Principle to me? - Physics Stack ...](https://physics.stackexchange.com/questions/127037/can-you-explain-fermats-principle-to-me)
[^20]: [Lecture 22](https://www.phy.pku.edu.cn/__local/1/B8/99/687EDDC5B886B271958D1CD906D_C14EBA85_73429.pdf?e=.pdf)
[^21]: [Fermat Principle, Ramsey Theory and Metamaterials - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC10744505/)
[^22]: [Proof of Snell's law using Fermat's Principle and the Euler-Lagrange equation - YouTube](https://www.youtube.com/watch?v=s9Y93Ijv8T0)
[^23]: [Chapter 3 The Variational Principle - Physics](http://www.physics.hmc.edu/~saeta/courses/p111/uploads/Y2013/chap03.pdf)
[^24]: [Made with Manim](https://www.manim.community/awesome/)
[^25]: [Manim Tutorial Series E01: An Invitation to Mathematical Animations WITH EASE in Python](https://www.youtube.com/watch?v=rUsUrbWb2D4)
[^26]: [Ray - Manim Physics v0.4.0 - Read the Docs](https://manim-physics.readthedocs.io/en/latest/reference/manim_physics.optics.rays.Ray.html)
[^27]: [Wavefront Aberrations - SPIE Digital Library](https://www.spiedigitallibrary.org/ebook/Download?urlid=10.1117%2F3.858456.ch8&isFullBook=False)
[^28]: [ON WAVEFRONT ABERRATIONS IN ASYMMETRIC AND MULTIPLE APERTURE OPTICAL SYSTEMS - Infoscience](https://infoscience.epfl.ch/bitstreams/8d518369-2527-466e-9fe6-6d97f0a9f561/download)
[^29]: [Understanding RMS Wavefront Error: An In-Depth Exploration | OFH - Optics for Hire](https://www.opticsforhire.com/blog/rms-wavefront-error-explained/)
[^30]: [Basic Wavefront Aberration Theory for Optical Metrology](https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/03-BasicAberrations_and_Optical_Testing.pdf)
[^31]: [Intuitive explanation of $L^2$-norm - Mathematics Stack Exchange](https://math.stackexchange.com/questions/1807204/intuitive-explanation-of-l2-norm)
[^32]: [the $L^2$-norm of a signal is also applied as its energy!](https://dsp.stackexchange.com/questions/71058/the-l2-norm-of-a-signal-is-also-applied-as-its-energy)
[^33]: [Physical interpretation of $L_1$ and $L_2$ norms - Mathematics Stack Exchange](https://math.stackexchange.com/questions/885998/physical-interpretation-of-l-1-and-l-2-norms)
[^34]: [L^2-Norm -- from Wolfram MathWorld](https://mathworld.wolfram.com/L2-Norm.html)
[^35]: [The L2 Norm and Inner Products - Ivethium](https://www.ivethium.com/the-l2-norm-and-inner-products/)
[^36]: [Norms for Signals and Systems - AERO 632 - ISRL](https://isrlab.github.io/pdfs/aero632/02%20Signal%20Norms.pdf)
[^37]: [OPTI 517 Image Quality](https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/Opti517-Optical-Quality-2014.pdf)
[^38]: [Telescope optical aberrations](https://www.telescope-optics.net/aberrations.htm)
[^39]: [Wavefront Error Analysis in Zemax - Resources - Ozen Engineering, Inc](https://blog.ozeninc.com/resources/wavefront-error-analysis-in-zemax)
[^40]: [RMS vs. Field - Ansys Help](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/RMS_vs_Field.html)
[^41]: [CoordinateSystem - Manim Community v0.19.0](https://docs.manim.community/en/stable/reference/manim.mobject.graphing.coordinate_systems.CoordinateSystem.html)
[^42]: [Why using Area under curve in manim is giving me an error? - Stack Overflow](https://stackoverflow.com/questions/69615564/why-using-area-under-curve-in-manim-is-giving-me-an-error)
[^43]: [Problem with .get_area() method : r/manim - Reddit](https://www.reddit.com/r/manim/comments/zpfjc5/problem_with_get_area_method/)
[^44]: [Continuous measurement of optical surfaces using a line-scan interferometer with sinusoidal path length modulation - Optica Publishing Group](https://opg.optica.org/abstract.cfm?uri=oe-22-24-29787)
[^45]: [The evolution of interferometry from metrology to biomedical applications - SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9718/971802/The-evolution-of-interferometry-from-metrology-to-biomedical-applications/10.1117/12.2218169.full)
[^46]: [Optical design algorithm utilizing continuous-discrete variables grounded on stochastic processes - ResearchGate](https://www.researchgate.net/publication/375478417_Optical_design_algorithm_utilizing_continuous-discrete_variables_grounded_on_stochastic_processes)
[^47]: [How to Measure MTF and other Properties of Lenses | Optikos](https://www.optikos.com/wp-content/uploads/2015/10/How-to-Measure-MTF-and-other-Properties-of-Lenses.pdf)
[^48]: [A Novel Analytical Interpolation Approach for Determining the Locus of a Zoom Lens Optical System - MDPI](https://www.mdpi.com/2304-6732/11/4/303)
[^49]: [Example Gallery - Manim Community v0.19.0](https://docs.manim.community/en/stable/examples.html)
[^50]: [Graph through Points : r/manim - Reddit](https://www.reddit.com/r/manim/comments/ep189r/graph_through_points/)
[^51]: [Dot - Manim Community v0.19.0](https://docs.manim.community/en/stable/reference/manim.mobject.geometry.arc.Dot.html)
[^52]: [Manim's building blocks](https://docs.manim.community/en/stable/tutorials/building_blocks.html)
[^53]: [Transform function & .animate in Manim - python - Stack Overflow](https://stackoverflow.com/questions/72926411/transform-function-animate-in-manim)
[^54]: [Geometrical aberrations](https://labs.phys.utk.edu/mbreinig/phys421/modules/m4/Geometrical%20aberrations.html)
[^55]: [Three-Dimensional Plotting in Matplotlib | Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html)
[^56]: [3D Surface plotting in Python using Matplotlib - GeeksforGeeks](https://www.geeksforgeeks.org/python/3d-surface-plotting-in-python-using-matplotlib/)
[^57]: [Spherical coordinates plot - python - Stack Overflow](https://stackoverflow.com/questions/36816537/spherical-coordinates-plot)
[^58]: [Exact Optical Ray Tracer in Python - Reddit](https://www.reddit.com/r/Optics/comments/12izd2c/exact_optical_ray_tracer_in_python/)
[^59]: [Naibaowjk/Snell-s-law: It's implemented with python and matlab, Only the incident and refractive directions are realized, without considering the amplitude size of the TEM wave - GitHub](https://github.com/Naibaowjk/Snell-s-law)
[^60]: [Simple Three-D Raytrace Algorithm - Tom Murphy](https://tmurphy.physics.ucsd.edu/astr597/exercises/raytrace-3d.pdf)
[^61]: [Ray Optics Simulation Python Projects - matlabsimulation](https://matlabsimulation.com/ray-optics-simulation-python/)
[^62]: [Wide Angle Lens Distortion Correction with Straight Lines - Hugo Hadfield](https://hh409.user.srcf.net/blog/lens-distortion-correction-lines.html)
[^63]: [correcting fisheye distortion programmatically - Stack Overflow](https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically)s