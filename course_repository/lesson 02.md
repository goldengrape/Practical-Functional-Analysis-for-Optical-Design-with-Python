## 第二周：变分法——从经典物理到计算设计

### 2.1 引言：超越函数的优化

在第一周，我们回顾了优化的基本概念：寻找使特定函数 $f(x)$ 达到最小值或最大值的点 $x$。然而，在眼视光学、物理学和工程学的许多尖端问题中，我们优化的对象不是一个简单的变量 $x$，而是一个完整的函数 $y(x)$。
我们的目标是找到一条能最小化或最大化某个积分量的路径或曲面。例如：

*   光线在不同介质中会选择哪条路径以最小化传播时间？
*   镜片表面的理想形状应该是什么，才能最小化所有视线的光学像差？

这些问题无法用标准微积分解决。它们属于一个更强大、更深刻的数学领域：**变分法 (Calculus of Variations)**。本周，我们将建立变分法的理论基础，推导其核心工具——**欧拉-拉格朗日方程 (Euler-Lagrange Equation)**，并通过经典物理问题（最短路径和最速降线）将其与光学（费马原理和斯涅尔定律）联系起来。
最重要的是，我们将展示为什么这些经典的解析解在面对现代镜片设计的复杂性时会失效，从而引入本课程的核心——Python 驱动的数值优化方法。

### 2.2 什么是泛函？

在深入探讨之前，我们必须定义我们的核心对象：**泛函 (Functional)**。
简单来说，泛函是“函数的函数”。常规函数（如 $f(x) = x^2$）接受一个数字作为输入，并返回一个数字。而泛函（通常表示为 $I[y]$ 或 $J(y)$）接受一个函数 $y(x)$ 作为输入，并返回一个数字[^2][^4]。
在变分法中，我们最常遇到的泛函形式是一个依赖于未知函数 $y(x)$ 及其导数 $y'(x) = \frac{dy}{dx}$ 的定积分[^5]：
$$I[y] = \int_a^b F(x, y(x), y'(x)) dx$$
这里的 $F$ 是一个普通的多元函数，称为**拉格朗日量 (Lagrangian)**。$I[y]$ 的值取决于在积分区间 $[a, b]$ 上选择的整个函数 $y(x)$ 的形状[^5]。

示例：

*   **弧长泛函**： 如果我们想计算连接点 $(a, y_a)$ 和 $(b, y_b)$ 的曲线 $y(x)$ 的总长度，泛函就是：
    $L[y] = \int_a^b ds = \int_a^b \sqrt{1 + (y'(x))^2} dx$。
    在这里，$F(x, y, y') = \sqrt{1 + (y')^2}$。输入一条曲线（一个函数 $y(x)$），输出它的长度（一个数字）。

变分法的核心问题是：在所有满足特定边界条件（例如 $y(a) = A$ 和 $y(b) = B$）的可能函数 $y(x)$ 中，哪一个函数能使泛函 $I[y]$ 达到极值（最大值或最小值）？[^5]。

### 2.3 欧拉-拉格朗日方程：寻找最优路径

我们如何找到使泛函 $I[y]$ 取极值的函数 $y(x)$？
回想一下标准微积分：要找到 $f(x)$ 的极值，我们求解 $f'(x) = 0$[^6]。我们需要一个类似的工具来处理泛函。这个工具就是欧拉-拉格朗日方程 (Euler-Lagrange Equation)，它代表了泛函导数（或称变分导数）为零的条件[^2]。

#### 2.3.1 直观理解：路径的微扰

我们将通过一种称为**路径微扰 (Path Perturbation)** 的方法来直观地推导这个方程。

1.  **假设存在最优路径**：假设 $y(x)$ 是我们正在寻找的、使 $I[y]$ 最小化的“正确”路径。
2.  **定义“错误”路径**：我们构造一个接近 $y(x)$ 但略有偏差的“错误”路径。我们通过添加一个小的、任意的“扰动函数” $\eta(x)$ 来实现这一点，该函数乘以一个极小的数 $\epsilon$。
    $$y(x, \epsilon) = y(x) + \epsilon \eta(x)$$
    *   $y(x)$ 是我们要求的真实路径 (即 $\epsilon=0$ 时的路径)[^7]。
    *   $\epsilon$ 是一个标量参数。
    *   $\eta(x)$ 是任何满足边界条件的连续函数。由于“正确”路径和“错误”路径都必须在相同的端点开始和结束（即 $y(a)=A, y(b)=B$），扰动函数在端点必须为零：$\eta(a) = 0$ 且 $\eta(b) = 0$[^5]。
3.  **构建关于 $\epsilon$ 的函数**：通过将这条“可变路径” $y(x, \epsilon)$ 代入泛函，我们的泛函 $I[y(x, \epsilon)]$ 暂时变成了一个关于 $\epsilon$ 的普通函数 $I(\epsilon)$：
    $$I(\epsilon) = \int_a^b F(x, y(x, \epsilon), y'(x, \epsilon)) dx$$
4.  **应用极值条件**：如果 $y(x)$ 是真正的最优路径，那么 $I(\epsilon)$ 必须在 $\epsilon = 0$ 时取到最小值。根据标准微积分，这意味着 $I(\epsilon)$ 在 $\epsilon = 0$ 处的导数必须为零[^7]：
    $$\left. \frac{dI}{d\epsilon} \right|_{\epsilon=0} = 0$$
    这被称为**一阶变分 (First Variation)** 为零。

#### 2.3.2 数学推导

现在我们来计算这个导数：

1.  **使用链式法则**：我们将导数 $\frac{d}{d\epsilon}$ 移入积分号内，并对 $F$ 应用链式法则：
    $$\frac{dI}{d\epsilon} = \int_a^b \left( \frac{\partial F}{\partial y} \frac{\partial y}{\partial \epsilon} + \frac{\partial F}{\partial y'} \frac{\partial y'}{\partial \epsilon} \right) dx$$
2.  **计算偏导数**：根据 $y(x, \epsilon) = y(x) + \epsilon \eta(x)$ 的定义：
    *   $\frac{\partial y}{\partial \epsilon} = \eta(x)$
    *   $\frac{\partial y'}{\partial \epsilon} = \frac{\partial}{\partial \epsilon} (\frac{dy}{dx}) = \frac{d}{dx} (\frac{\partial y}{\partial \epsilon}) = \frac{d}{dx}(\eta(x)) = \eta'(x)$
3.  **代入并设 $\epsilon = 0$**：
    $$\left. \frac{dI}{d\epsilon} \right|_{\epsilon=0} = \int_a^b \left( \frac{\partial F}{\partial y} \eta(x) + \frac{\partial F}{\partial y'} \eta'(x) \right) dx = 0$$
    (注意：$\frac{\partial F}{\partial y}$ 和 $\frac{\partial F}{\partial y'}$ 是在 $\epsilon=0$，即在真实路径 $y(x)$ 上计算的。)
4.  **分部积分 (Integration by Parts)**：这是整个推导中最关键的技巧。我们对第二项应用分部积分，以消除 $\eta'(x)$ 并分离出 $\eta(x)$[^7]：
    $$\int_a^b \frac{\partial F}{\partial y'} \eta'(x) dx = \left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b - \int_a^b \eta(x) \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) dx$$
5.  **应用边界条件**：由于我们规定了扰动在端点必须为零，即 $\eta(a) = \eta(b) = 0$，边界项 $\left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b$ 等于零[^7]。
6.  **合并同类项**：将分部积分的结果代回原方程，我们得到：
    $$\int_a^b \left( \frac{\partial F}{\partial y} \eta(x) - \eta(x) \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) \right) dx = 0 $$
    $$ \int_a^b \left( \frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) \right) \eta(x) dx = 0$$
7.  **变分法基本引理 (Fundamental Lemma of Calculus of Variations)**：这个方程告诉我们，括号内的项乘以任何满足条件的扰动函数 $\eta(x)$，其积分都为零。唯一的可能性是括号内的项本身在所有点上都必须为零[^6]。

因此，我们得到了变分法的核心方程：
$$\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0$$
这就是欧拉-拉格朗日方程[^6]。

#### 2.3.3 方程的意义

欧拉-拉格朗日方程是一个革命性的工具。它将一个复杂的、无限维度的泛函最小化问题，转化为了一个（通常是二阶的）常微分方程 (ODE) 求解问题[^9]。我们不再需要“猜测”函数；我们只需要求解这个 ODE，就能找到唯一的候选路径（极值路径）。

### 2.4 应用（一）：两点之间的最短路径

让我们用一个我们已经知道答案的问题来测试这个新工具：平面上两点之间的最短路径是什么？直觉告诉我们是一条直线。让我们用变分法来证明它。

1.  **定义泛函 (弧长)**：如前所述，最小化弧长 $L$ 的泛函为[^12]：
    $$L[y] = \int_a^b \sqrt{1 + (y')^2} dx$$
2.  **确定拉格朗日量 $F$**：
    $$F(x, y, y') = \sqrt{1 + (y')^2}$$
3.  **计算 E-L 方程所需偏导数**[^8]：
    *   $\frac{\partial F}{\partial y} = 0$ (因为 $F$ 不显含 $y$)
    *   $\frac{\partial F}{\partial y'} = \frac{\partial}{\partial y'} (1 + (y')^2)^{1/2} = \frac{1}{2}(1 + (y')^2)^{-1/2} \cdot (2y') = \frac{y'}{\sqrt{1 + (y')^2}}$
4.  **代入欧拉-拉格朗日方程**：
    $$\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0 $$
    $$ 0 - \frac{d}{dx} \left( \frac{y'}{\sqrt{1 + (y')^2}} \right) = 0$$
5.  **求解 ODE**：
    如果某一项的导数为零，则该项本身必须是一个常数[^14]。
    $$\frac{y'}{\sqrt{1 + (y')^2}} = C \quad (\text{其中 } C \text{ 是常数})$$
    现在，我们用代数方法求解 $y'$：
    *   两边平方： $(y')^2 = C^2 (1 + (y')^2)$
    *   展开： $(y')^2 = C^2 + C^2 (y')^2$
    *   整理： $(y')^2 (1 - C^2) = C^2$
    *   解得： $(y')^2 = \frac{C^2}{1 - C^2}$
    由于 $C$ 是常数，$\frac{C^2}{1 - C^2}$ 也是一个常数，我们称之为 $m^2$。
    *   $(y')^2 = m^2 \implies y' = m$
6.  **结论**：
    $y' = \frac{dy}{dx} = m$ (一个常数)。积分一次，我们得到：
    $$y(x) = mx + b$$
    这是一个直线方程。变分法严谨地证明了我们的直觉：平面上两点之间的最短路径是一条直线[^8]。

### 2.5 应用（二）：最速降线问题 (Brachistochrone)

现在我们来解决一个更深刻、更不直观的问题，这个问题标志着变分法的诞生：**最速降线问题**。
问题如下：一个珠子在重力作用下，从点 $A(a, y_a)$ 无摩擦地滑到点 $B(b, y_b)$（$B$点在$A$点下方）。珠子应该沿着什么形状的曲线 $y(x)$ 下滑，才能使所用时间最短？[^16]
直觉可能会告诉我们是直线（最短路径），但事实并非如此。

#### 2.5.1 建立时间泛函

我们的目标是最小化总时间 $T$。

1.  **时间泛函**： $T = \int dt$。
2.  **建立 $dt$, $ds$ 和 $v$ 的关系**：
    *   $ds$ 是微小的弧长： $ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + (y')^2} dx$。
    *   $v$ 是珠子的速度。$v = \frac{ds}{dt}$，因此 $dt = \frac{ds}{v}$。
3.  **计算速度 $v$**：我们使用能量守恒定律。假设珠子从 $y=0$ 处静止出发（设定 $y$ 轴向下为正）。在任意高度 $y$ 处，失去的势能 $mgy$ 转化为动能 $\frac{1}{2}mv^2$[^17]。
    $$mgy = \frac{1}{2}mv^2 \implies v = \sqrt{2gy}$$
4.  **组合泛函**：将 $ds$ 和 $v$ 代入 $dt = \frac{ds}{v}$：
    $$T[y] = \int dt = \int_a^b \frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}} dx$$
5.  **确定拉格朗日量 $F$**：
    常数 $\frac{1}{\sqrt{2g}}$ 不会影响最小化，我们可以将其忽略。拉格朗日量为[^19]：
    $$F(x, y, y') = F(y, y') = \frac{\sqrt{1 + (y')^2}}{\sqrt{y}}$$

#### 2.5.2 求解：贝尔特拉米恒等式 (Beltrami Identity)

我们可以直接将这个 $F$ 代入欧拉-拉格朗日方程。然而，$\frac{\partial F}{\partial y}$ 和 $\frac{d}{dx}(\frac{\partial F}{\partial y'})$ 都会非常复杂，导致一个难以处理的二阶 ODE[^20]。
这里，我们展示一个更强大的技巧。请注意，我们的拉格朗日量 $F(y, y')$ 不显含 $x$ (即 $\frac{\partial F}{\partial x} = 0$)[^21]。
当拉格朗日量不显含 $x$ 时，存在一个“守恒定律”，称为**贝尔特拉米恒等式 (Beltrami Identity)**，它是欧拉-拉格朗日方程的“一阶积分”：
$$F - y' \frac{\partial F}{\partial y'} = C \quad (\text{其中 } C \text{ 是常数})$$
这个恒等式（可以从 E-L 方程推导[^15]）将问题从一个二阶 ODE 降维到一个一阶 ODE，大大简化了求解过程[^24]。

#### 2.5.3 求解最速降线

让我们应用贝尔特拉米恒等式[^19]：

1.  **已知项**：
    *   $F = \frac{\sqrt{1 + (y')^2}}{\sqrt{y}}$
    *   $\frac{\partial F}{\partial y'} = \frac{1}{\sqrt{y}} \cdot \frac{\partial}{\partial y'} (1 + (y')^2)^{1/2} = \frac{1}{\sqrt{y}} \left( \frac{y'}{\sqrt{1 + (y')^2}} \right)$
2.  **代入恒等式 $F - y' \frac{\partial F}{\partial y'} = C$**：
    $$\frac{\sqrt{1 + (y')^2}}{\sqrt{y}} - y' \left( \frac{y'}{\sqrt{y}\sqrt{1 + (y')^2}} \right) = C$$
3.  **代数化简**（目标是求解 $y'$）：
    *   通分，公分母为 $\sqrt{y}\sqrt{1 + (y')^2}$：
        $$\frac{(1 + (y')^2) - (y')^2}{\sqrt{y(1 + (y')^2)}} = C$$
    *   分子上的 $(y')^2$ 被消去：
        $$\frac{1}{\sqrt{y(1 + (y')^2)}} = C$$
    *   两边平方： $1 = C^2 y (1 + (y')^2)$
    *   整理： $1 + (y')^2 = \frac{1}{C^2 y}$
    *   令 $k = \frac{1}{C^2}$ (一个新的常数)： $(y')^2 = \frac{k}{y} - 1 = \frac{k - y}{y}$
    *   解得 $y'$：
        $$y' = \frac{dy}{dx} = \sqrt{\frac{k - y}{y}}$$
4.  **结论**：
    这是一个一阶可分离变量的 ODE。虽然积分需要一些技巧（例如使用三角换元 $y = k \sin^2\theta$[^19]），但它可以被解析求解。最终的解，即最速降线，是**摆线 (Cycloid)**[^25]——一个圆在直线上滚动时，圆周上一个定点的轨迹。

这个深刻的结果表明，最优路径既不是直线也不是圆弧，而是一个非常不直观的曲线。它展示了变分法在解决复杂优化问题时的惊人力量。

### 2.6 统一理论：从粒子到光子

最速降线问题的解引出了一个更深的联系，这个联系是本课程的核心：力学和光学是统一的。
约翰·伯努利 (Johann Bernoulli) 在 1696 年提出最速降线问题时，他自己是通过一个惊人的类比（而不是欧拉-拉格朗日方程）解决的：他把这个问题重新想象成一个光学问题[^16]。

#### 2.6.1 费马原理 (Fermat's Principle)

我们首先回顾一下几何光学的基本原理：费马原理，又称“最少时间原理”。该原理指出，光线在两点之间传播的路径，是所需时间最短的路径[^1]。
这听起来和最速降线问题一模一样！

#### 2.6.2 用费马原理推导斯涅尔定律 (Snell's Law)

我们可以在离散情况下应用费马原理，来推导光线在两种不同介质（如空气和水）的平直界面上的折射定律[^28]。

1.  **建立时间函数**：假设光从介质1中的 $A(0, h_1)$ 传播到介质2中的 $B(d, -h_2)$，在 $x$ 处穿过边界。光在介质1中的速度为 $v_1 = c/n_1$，在介质2中的速度为 $v_2 = c/n_2$。总时间 $T$ 是 $x$ 的函数[^29]：
    $$T(x) = t_1 + t_2 = \frac{\sqrt{h_1^2 + x^2}}{v_1} + \frac{\sqrt{h_2^2 + (d-x)^2}}{v_2}$$
2.  **最小化时间**：这是一个标准的微积分1问题。我们求导并令其为零：$\frac{dT}{dx} = 0$[^29]。
    $$\frac{dT}{dx} = \frac{1}{v_1} \frac{x}{\sqrt{h_1^2 + x^2}} - \frac{1}{v_2} \frac{d-x}{\sqrt{h_2^2 + (d-x)^2}} = 0$$
3.  **识别几何关系**：我们从几何中识别出 $\sin\theta_1 = \frac{x}{\sqrt{h_1^2 + x^2}}$ 和 $\sin\theta_2 = \frac{d-x}{\sqrt{h_2^2 + (d-x)^2}}$[^29]。
4.  **得到斯涅尔定律**：代入上式：
    $$\frac{\sin\theta_1}{v_1} = \frac{\sin\theta_2}{v_2} \implies n_1 \sin\theta_1 = n_2 \sin\theta_2$$
    我们从费马的最少时间原理推导出了斯涅尔定律[^29]。

#### 2.6.3 伯努利的洞察：连续折射

伯努利的绝妙之处在于他将离散的折射推广到了连续介质[^33]。

1.  **力学-光学类比**：
    *   在最速降线问题中，珠子的速度 $v(y) = \sqrt{2gy}$ 随高度 $y$ 连续变化。
    *   在光学问题中，光的速度 $v(y) = c/n(y)$ 随折射率 $n(y)$ 连续变化。
    *   伯努利意识到，最速降线问题在数学上等同于光线在一个折射率连续变化的介质中传播的问题[^16]。
    *   通过类比 $v_{力学} = v_{光学}$，他建立了一个等效的“重力介质”，其折射率 $n(y) \propto 1/v(y)$，即 $n(y) \propto 1/\sqrt{y}$[^16]。
2.  **连续斯涅尔定律**：
    *   想象这个连续介质是由无数个极薄的水平层组成的[^34]。
    *   在每一层 $i$ 和 $i+1$ 之间的界面上，斯涅尔定律都成立： $n_i \sin\theta_i = n_{i+1} \sin\theta_{i+1}$。
    *   这必然意味着，在整个连续路径上， $n(y) \sin\theta(y)$ 的值必须是一个常数[^16]。
3.  **两种解法的等价性**：
    现在我们来比较两种方法得到的结果。
    *   **方法1 (贝尔特拉米)**：$F - y'\frac{\partial F}{\partial y'} = C \implies \frac{1}{\sqrt{y(1 + (y')^2)}} = C$
    *   **方法2 (连续斯涅尔定律)**：$n(y) \sin\theta(y) = K$
    *   我们有 $n(y) \propto 1/\sqrt{y}$。
    *   $\theta$ 是路径与垂直方向（$y$轴）的夹角。因此，$\sin\theta = \frac{dx}{ds} = \frac{dx}{\sqrt{dx^2 + dy^2}} = \frac{1}{\sqrt{1 + (dy/dx)^2}} = \frac{1}{\sqrt{1 + (y')^2}}$。
    *   代入： $(\frac{1}{\sqrt{y}}) \cdot (\frac{1}{\sqrt{1 + (y')^2}}) = K'$ (K' 是吸收了所有比例常数的新常数)。
    *   结果：$\frac{1}{\sqrt{y(1 + (y')^2)}} = K'$

两种方法得出了完全相同的微分方程！这一发现具有深远的意义：最速降线问题就是一个在渐变折射率 (GRIN) 介质中的光线传播问题。欧拉-拉格朗日方程（及其特例贝尔特拉米恒等式）是费马原理和斯涅尔定律在连续介质中的数学体现。这正是本课程的基石：用于经典力学的变分原理，同样是驱动现代光学设计的底层逻辑[^1]。

### 2.7 实验课：Python 驱动的数值解

到目前为止，我们都在处理可以解析求解的“玩具问题”。摆线解虽然漂亮，但极其脆弱。如果我们稍微改变一下问题——比如增加空气阻力或摩擦力——拉格朗日量 $F$ 就会变得非常复杂，导致欧拉-拉格朗日方程无法解析求解[^38]。
在现实世界的眼视光学设计中，我们面对的问题（例如设计一个渐进多焦点镜片）的“拉格朗日量”是一个极其复杂的评价函数 (Merit Function)，它需要最小化数千条光线在不同视角下的像差总和[^40]。在这种情况下，写下或求解欧拉-拉格朗日方程是根本不可能的[^42]。
我们必须转向一种更通用、更强大的方法：**数值优化**。

#### 2.7.1 新的策略：离散化

我们将从根本上改变我们的问题表述：

*   **旧问题 (解析法)**：寻找一个连续函数 $y(x)$（一个无限维对象）。
*   **新问题 (数值法)**：寻找一个离散的点向量 $\mathbf{y} = [y_1, y_2,..., y_N]$，它以足够高的精度逼近真实路径[^43]。

通过离散化，我们将一个（通常无法解决的）泛函最小化问题 $T[y(x)]$ 转换为了一个（可以解决的）多变量函数最小化问题 $T(\mathbf{y})$。
现在，我们可以使用强大的 Python 库（如 SciPy）中的标准数值优化算法（如梯度下降法或其变体）来找到使 $T(\mathbf{y})$ 最小的向量 $\mathbf{y}$[^45]。

#### 2.7.2 Python 实验：数值求解最速降线

本实验的目标是验证我们的数值方法。我们将使用 Python 来数值计算最速降线，并将其与我们之前推导出的解析解（摆线）进行比较。如果两者一致，就证明了我们的数值流程是正确且有效的。

**步骤 1：定义（数值）时间泛函**

我们首先需要一个 Python 函数，它接受一个离散的路径 $\mathbf{y}$，并返回珠子滑过该路径的总时间。这个函数就是我们的数值版“泛函”或“成本函数”。[^46]
```python
import numpy as np

def calculate_descent_time(y_points, x_points, g=9.81):
   """
   计算给定离散路径 (x_points, y_points) 的总下降时间。
   假设 y 轴向下为正，g 为重力加速度。
   """
   
   # 确保 y[0] 为起始点，y[0] 之后的 y 值 > y[0]
   # 为简单起见，我们假设 y_points[0] = 0
   y_path = y_points.copy()
   if y_path[0]!= 0:
       y_path = y_path - y_path[0] # 将起点归零
   
   # 计算每个线段的 dx 和 dy
   dx = np.diff(x_points)
   dy = np.diff(y_points)
   
   # 1. 计算每个线段的弧长 ds = sqrt(dx^2 + dy^2)
   segment_lengths = np.sqrt(dx**2 + dy**2)
   
   # 2. 计算每个线段中点的 y 坐标
   # (y_path[:-1] 是起点, y_path[1:] 是终点)
   midpoint_y = (y_path[:-1] + y_path[1:]) / 2
   
   # 3. 计算每个线段中点的速度 v = sqrt(2gy)
   # 为避免 v=0 (除以零错误)，在 y=0 处添加一个极小值
   midpoint_y[midpoint_y <= 0] = 1e-9
   segment_velocities = np.sqrt(2 * g * midpoint_y)
   
   # 4. 计算每个线段的时间 dt = ds / v
   segment_times = segment_lengths / segment_velocities
   
   # 5. 总时间是所有线段时间的总和
   total_time = np.sum(segment_times)
   
   return total_time
```

**步骤 2：使用 `scipy.optimize.minimize` 进行优化**

现在我们有了“成本函数” `calculate_descent_time`，我们需要一个“求解器”来找到使它最小化的路径 $\mathbf{y}$。我们将使用 `scipy.optimize.minimize`，这是一个强大的、内置了多种优化算法（如 'BFGS'）的工具[^47]。
```python
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 定义问题参数 ---
N_POINTS = 50  # 路径的离散点数
START_POINT = (0.0, 0.0) # (x, y)
END_POINT = (2.0, 1.0)   # (x, y)

# 1. 创建固定的 x 坐标网格
x_path = np.linspace(START_POINT[0], END_POINT[0], N_POINTS)

# 2. 创建一个“目标函数”，供优化器调用
# 优化器只能改变“自由”变量，即 N-2 个内部点
def objective_function(inner_y_points):
   # 将固定的端点和可变的内点重新组合成完整路径
   y_path = np.concatenate(
       ([START_POINT[1]], inner_y_points, [END_POINT[1]])
   )
   # 返回成本（总时间）
   return calculate_descent_time(y_path, x_path)

# 3. 提供一个初始猜测路径
# 最简单的猜测是直线
y_guess_inner = np.linspace(START_POINT[1], END_POINT[1], N_POINTS)[1:-1]

# --- 运行优化器 ---
print("开始数值优化...")
# 'BFGS' 是一种高效的类梯度下降算法
result = minimize(
   objective_function, 
   y_guess_inner, 
   method='BFGS', 
   options={'disp': True, 'maxiter': 1000}
)
print("优化完成！")

# 4. 提取最优路径
if result.success:
   optimal_y_inner = result.x
   optimal_y_path = np.concatenate(
       ([START_POINT[1]], optimal_y_inner, [END_POINT[1]])
   )
   optimal_time = result.fun
   print(f"找到的最短时间: {optimal_time:.4f} s")

   # --- 5. 可视化和验证 --- 
   
   # A. 计算直线路径的时间
   y_linear = np.linspace(START_POINT[1], END_POINT[1], N_POINTS)
   linear_time = calculate_descent_time(y_linear, x_path)
   
   # B. 绘制解析解（摆线）以供比较
   def f(theta):
       return END_POINT[1]/END_POINT[0] - (1-np.cos(theta))/(theta-np.sin(theta))
   from scipy.optimize import newton
   theta2 = newton(f, np.pi/2)
   R = END_POINT[1] / (1 - np.cos(theta2))
   theta_analytic = np.linspace(0, theta2, N_POINTS)
   x_analytic = R * (theta_analytic - np.sin(theta_analytic))
   y_analytic = R * (1 - np.cos(theta_analytic))
   
   plt.figure(figsize=(12, 8))
   plt.plot(x_path, y_linear, 'r--', label=f'直线 (初始猜测) - T={linear_time:.4f} s')
   plt.plot(x_path, optimal_y_path, 'b-', label=f'数值解 (Scipy) - T={optimal_time:.4f} s', linewidth=4)
   plt.plot(x_analytic, y_analytic, 'g:', label='解析解 (摆线)', linewidth=4)
   
   plt.xlabel('X 坐标')
   plt.ylabel('Y 坐标 (向下为正)')
   plt.title('最速降线问题：数值解 vs 解析解')
   plt.legend()
   plt.gca().invert_yaxis() # 反转y轴使零点在顶部
   plt.show()

else:
   print("优化失败:", result.message)
```

**预期结果**：

执行此代码将产生一个图表，显示[^50][^51]：

1.  红色的虚线（直线）路径，其时间最长。
2.  绿色的虚线（摆线）解析解。
3.  蓝色的实线（`scipy.optimize` 的数值解）完美地覆盖在绿色的摆线之上。

#### 2.7.3 实验结论：验证计算流程

这个实验是本课程的第一个关键时刻。它无可辩驳地证明了，一个通用的、“黑盒”的数值优化流程（离散化路径 + `scipy.optimize.minimize`）能够自动发现与高度专业化的欧拉-拉格朗日和贝尔特拉米方程所推导出的完全相同的、非直观的解[^52]。

这给了我们巨大的信心。这意味着，即使我们将来面对没有解析解的、更复杂的问题（如摩擦力、空气阻力，或更复杂的镜片像差），我们仍然可以使用相同的数值流程来找到最优解。

为了更好地理解这两种方法的权衡，请参考下表：

**表2.1：解析法与数值法对比**

| 特性 | 解析法 (欧拉-拉格朗日) | 数值法 (离散化 + 梯度下降) |
| :--- | :--- | :--- |
| **问题表示** | 连续函数 $y(x)$ (无限维)[^54] | 离散向量 $\mathbf{y} = [y_1,..., y_N]$ (有限维)[^44] |
| **数学工具** | 变分法；求解微分方程[^6] | 多元微积分；数值优化[^45] |
| **解的类型** | 精确的、封闭形式的公式（如摆线）[^38] | 近似的、数值点集[^55] |
| **适用性** | 非常有限。需要简单的、可解析积分的拉格朗日量[^39] | 非常通用。可应用于任何可计算的成本函数[^52] |
| **主要挑战** | 解析复杂性：求解 ODE 通常是不可能的[^38] | 计算复杂性：高维优化所需的内存和时间[^57] |

### 2.8 结论：从最速降线到自由曲面镜片

本周，我们从一个古老的数学领域——变分法——开始，推导了其强大的欧拉-拉格朗日方程。我们证明了它如何能解析地解决像最短路径和最速降线这样的经典问题。
更重要的是，我们揭示了经典力学（最速降线）和几何光学（费马原理）之间深刻的数学等价性。我们证明了欧拉-拉格朗日/贝尔特拉米框架本质上是斯涅尔定律在连续介质中的一种表现形式。
然而，我们也看到了解析方法的局限性。对于现实世界中的复杂问题，它们很快就变得无能为力。
这使我们转向了本课程的核心——Python 驱动的计算方法。我们的实验课证明了，通过将问题离散化并应用数值优化算法 (如 `scipy.optimize`)，我们可以重现解析解，从而验证了这种方法的有效性。
这个从最速降线问题中学到的流程，并不是一个简单的类比；它是现代计算光学设计的简化模型[^58]：

1.  在镜片设计中，我们的“路径” $\mathbf{y}$ 是描述镜片自由曲面（非球面、非复曲面、渐进曲面）的一组参数（例如泽尼克多项式系数或 B 样条控制点）[^41]。
2.  我们的“成本函数” `calculate_descent_time` 被一个复杂的评价函数所取代。该函数使用可微分光线追踪 (differentiable ray tracing) 来计算数千条光线通过镜片后的总像差（模糊、畸变等）[^42]。
3.  我们的 `scipy.optimize.minimize` 求解器被先进的梯度下降算法所取代，该算法自动调整镜片表面参数，以迭代方式最小化像差[^45]。

在本周，我们已经建立了解决这些问题所需的所有基本理论和计算构件。在接下来的课程中，我们将不再局限于简单的 $y(x)$ 路径，而是开始扩展这个数值流程，以解决由 Python 驱动的真实眼视光学镜片设计问题。

---
### 引用的著作

[^1]: [Fermat's principle - Wikipedia](https://en.wikipedia.org/wiki/Fermat%27s_principle)
[^2]: [Functional derivative - Wikipedia](https://en.wikipedia.org/wiki/Functional_derivative)
[^3]: [Functional derivative - Wikipedia](https://en.wikipedia.org/wiki/Functional_derivative#:~:text=In%20the%20calculus%20of%20variations,on%20which%20the%20functional%20depends.)
[^4]: [In calculus of variations, what is a functional? - Math Stack Exchange](https://math.stackexchange.com/questions/572316/in-calculus-of-variations-what-is-a-functional)
[^5]: [MATH0043 §2: Calculus of Variations](https://www.ucl.ac.uk/~ucahmto/latex_html/pandoc_chapter2.html)
[^6]: [Calculus of variations - Wikipedia](https://en.wikipedia.org/wiki/Calculus_of_variations)
[^7]: [First variation - University of Utah Math Dept.](https://www.math.utah.edu/~cherk/teach/12calcvar/2euler.pdf)
[^8]: [Euler–Lagrange equation - Wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation)
[^9]: [The Euler-Lagrange equation](https://mathsci.kaist.ac.kr/~nipl/am621/lecturenotes/Euler-Lagrange_equation.pdf)
[^10]: [11.3: Derivation of the Euler-Lagrange Equation - Engineering LibreTexts](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electro-Optics/Direct_Energy_(Mitofsky)/11%3A_Calculus_of_Variations/11.03%3A_Derivation_of_the_Euler-Lagrange_Equation)
[^11]: [Euler Lagrange differential equation. - Mathematics Stack Exchange](https://math.stackexchange.com/questions/1361315/euler-lagrange-differential-equation)
[^12]: [Euler-Lagrange Equation - Richard Fitzpatrick](https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node266.html)
[^13]: [Arc Length - Calculus II - Pauls Online Math Notes](https://tutorial.math.lamar.edu/classes/calcii/arclength.aspx)
[^14]: [Special cases with examples: first integrals - Physics OER](https://oer.physics.manchester.ac.uk/AM/Notes/jsmath/Notesse15.html)
[^15]: [Teddy Rocks Maths Essay](https://tomrocksmaths.com/wp-content/uploads/2021/05/teddy_rocks_maths_essay.pdf)
[^16]: [Brachistochrone curve - Wikipedia](https://en.wikipedia.org/wiki/Brachistochrone_curve)
[^17]: [THE BRACHISTOCHRONE PROBLEM. Imagine a metal bead with a wire threaded through a hole in it, so that the bead can slide with no - UTK Math](https://web.math.utk.edu/~freire/teaching/m231f08/m231f08brachistochrone.pdf)
[^18]: [Brachistochrone Problem](https://archive.lib.msu.edu/crcmath/math/math/b/b355.htm)
[^19]: [A Few Notes on the Brachistochrone Problem - David Meyer](https://davidmeyer.github.io/qc/brachistochrone.pdf)
[^20]: [Show that the full Euler-Lagrange equation of the Brachistochrone is $2y(x)y''(x)+y'(x)^2+1=0$ - Mathematics Stack Exchange](https://math.stackexchange.com/questions/4332151/show-that-the-full-euler-lagrange-equation-of-the-brachistochrone-is-2yxyx)
[^21]: [Beltrami identity - Wikipedia](https://en.wikipedia.org/wiki/Beltrami_identity)
[^22]: [Functionals leading to special cases](https://www.ucl.ac.uk/~ucahmto/latex_html/chapter2_latex2html/node8.html)
[^23]: [Beltrami Identity -- from Wolfram MathWorld](https://mathworld.wolfram.com/BeltramiIdentity.html)
[^24]: [Euler-Lagrange Differential Equation -- from Wolfram MathWorld](https://mathworld.wolfram.com/Euler-LagrangeDifferentialEquation.html)
[^25]: [The Brachistochrone](https://www.ucl.ac.uk/~ucahmto/latex_html/chapter2_latex2html/node7.html)
[^26]: [Brachistochrone Problem -- from Wolfram MathWorld](https://mathworld.wolfram.com/BrachistochroneProblem.html)
[^27]: [The Brachistochrone, with Steven Strogatz - 3Blue1Brown](https://www.3blue1brown.com/lessons/brachistochrone)
[^28]: [Snell's law - Wikipedia](https://en.wikipedia.org/wiki/Snell%27s_law)
[^29]: [Fermat's Principle and the Laws of Reflection and Refraction](http://scipp.ucsc.edu/~haber/ph5B/fermat09.pdf)
[^30]: [Exploration 34.4: Fermat's Principle and Snell's Law - ComPADRE](https://www.compadre.org/physlets/optics/ex34_4.cfm)
[^31]: [Fermat's Principle to Snell's Law (Derivation) - YouTube](https://www.youtube.com/watch?v=3Etj75qzGg0)
[^32]: [Deriving Snell's law - YouTube](https://www.youtube.com/watch?v=8wYkgZKboss)
[^33]: [Johann Bernoulli's brachistochrone solution using Fermat's principle of least time - Mechanical | IISc](https://mecheng.iisc.ac.in/suresh/me256/GalileoBP.pdf)
[^34]: [The brachistochrone. The brachistochrone problem is a… | by ...](https://medium.com/@hamza-ihind/the-brachistochrone-3397143bfb70)
[^35]: [2.1.4 Brachistochrone - Daniel Liberzon](https://liberzon.csl.illinois.edu/teaching/cvoc/node24.html)
[^36]: [ws-ijbc.pdf](https://www.math.rug.nl/~broer/pdf/ws-ijbc.pdf)
[^37]: [Functional Analysis and its relation to mechanics - MathOverflow](https://mathoverflow.net/questions/30120/functional-analysis-and-its-relation-to-mechanics)
[^38]: [Whats the difference between solving something analytically and solving something numerically? - Reddit](https://www.reddit.com/r/learnmath/comments/d8svf1/whats_the_difference_between_solving_something/)
[^39]: [From Analytical to Numerical to Universal Solutions - Ethan Rosenthal](https://www.ethanrosenthal.com/2017/03/20/analytical-numerical-universal/)
[^40]: [Analysis of a Variational Approach to Progressive Lens Design](https://www.researchgate.net/publication/220222531_Analysis_of_a_Variational_Approach_to_Progressive_Lens_Design)
[^41]: [Optimization of freeform spectacle lenses based on high-order aberrations](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12666/1266604/Optimization-of-freeform-spectacle-lenses-based-on-high-order-aberrations/10.1117/12.2676950.short)
[^42]: [Gradient descent-based freeform optics design using algorithmic ...](https://arxiv.org/abs/2302.12031)
[^43]: [Discretize path with numpy array and equal distance between points - Stack Overflow](https://stackoverflow.com/questions/24229585/discretize-path-with-numpy-array-and-equal-distance-between-points)
[^44]: [How to solve calculus of variations problems numerically? - Computational Science Stack Exchange](https://scicomp.stackexchange.com/questions/33632/how-to-solve-calculus-of-variations-problems-numerically)
[^45]: [Why Gradient Descent Works? | Towards Data Science](https://towardsdatascience.com/why-gradient-descent-works-4e487d3c84c1/)
[^46]: [Optimization (scipy.optimize) — SciPy v1.16.2 Manual - Numpy and Scipy Documentation](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
[^47]: [Scientific Python: Using SciPy for Optimization](https://realpython.com/python-scipy-cluster-optimize/)
[^48]: [Optimization and root finding - Numpy and Scipy Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)
[^49]: [Brachistochrone Optimal Control in Python - YouTube](https://www.youtube.com/watch?v=pgJ0jbfFBUE)
[^50]: [The Brachistochrone — Dymos](https://openmdao.github.io/dymos/examples/brachistochrone/brachistochrone.html)
[^51]: [[1904.02539] An introduction to functional analysis for science and engineering - arXiv](https://arxiv.org/abs/1904.02539)
[^52]: [What's the difference between analytical and numerical approaches to problems?](https://math.stackexchange.com/questions/935405/what-s-the-difference-between-analytical-and-numerical-approaches-to-problems)
[^53]: [Numerical analysis - Wikipedia](https://en.wikipedia.org/wiki/Numerical_analysis)
[^54]: [Quantitative Comparison of Analytical Solution and Finite Element Method for Investigation of Near-infrared Light Propagation in Brain Tissue Model - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC10719975/)
[^55]: [Efficient freeform lens optimization for computational caustic displays - Optica Publishing Group](https://opg.optica.org/oe/fulltext.cfm?uri=oe-23-8-10224)
[^56]: [Differentiable optimization of multiple freeform lenses for high-performance tilted illumination](https://opg.optica.org/ol/abstract.cfm?uri=ol-50-11-3505)
[^57]: [Freeform imaging systems: Fermat's principle unlocks “first time right” design - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8102611/)
[^58]: [Deblur or denoise: the role of an aperture in lens and neural network co-design](https://opg.optica.org/ol/abstract.cfm?uri=ol-48-2-231)
[^59]: [Gradient descent method applied for normal minimum and degraded minimum shape. - ResearchGate](https://www.researchgate.net/figure/Gradient-descent-method-applied-for-normal-minimum-and-degraded-minimum-shape_fig10_260152073)
[^60]: [Extrapolating from lens design databases using deep learning - Optica Publishing Group](https://opg.optica.org/OE/fulltext.cfm?uri=oe-27-20-28279)