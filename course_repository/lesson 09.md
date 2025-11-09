# 第9章 弱收敛与多焦点优化

## 9.1 导论：从临床痛点到数学挑战

在眼视光学镜片设计中，尤其是渐进多焦点镜片（Progressive Addition Lenses, PALs）的设计，工程师面临的不仅仅是光学计算问题，更是对患者主观感知的深刻理解与平衡。本课程的核心理念之一是建立“临床-数学”桥梁[^1]，而第9章正是这一理念的集中体现。

渐进镜片设计的核心挑战在于其“多目标”的本质。一方面，镜片需要为佩戴者提供从远用区到近用区平滑、连续的屈光力变化，以满足看远、看中（如电脑屏幕）、看近（如阅读）的需求[^2]。另一方面，根据物理定律，在创造这种平滑过渡的同时，不可避免地会在镜片的两侧（周边区域）引入“不需要的像散”（unwanted astigmatism）[^2]。

这种周边像散是导致患者在适应初期报告最强烈临床痛点的根源：“晃动感”或“游泳效应”（swim effect）[^4]。当佩戴者转头时，周边视野中的物体似乎在不自然地移动或扭曲。科学研究表明，这种“游泳效应”并不仅仅是简单的模糊，它本质上是一种由镜片周边复杂光学特性（特别是球柱面误差）引起的“失真的光学流”（distorted optic flow）[^6]。人脑通过视觉系统中的光学流来感知自身的运动；当光学流被镜片扭曲时，大脑会错误地感知到自身运动或环境在晃动，严重时甚至会导致恶心[^7]。

因此，镜片设计师的职责变成了一场精妙的“杂耍”[^8]。我们必须在多个相互冲突的目标之间做出权衡：
1.  中心视野清晰度：最大化远、中、近三个核心视区的光学性能[^2]。
2.  周边稳定性：最小化“游泳效应”，即最小化周边区域光学流的失真度[^6]。
3.  通道宽度与长度：在有限的镜片空间内，平衡通道的宽度（越宽越好）和长度（越短越适合小镜框，但可能导致像散梯度更大）[^9]。

我们不能将所有目标都同时优化到完美。例如，过度追求更宽的中心通道，几乎总是以牺牲周边区域、增加“游泳效应”为代价[^2]。这就是一个经典且复杂的**多目标优化 (Multi-Objective Optimization, MOO)** 问题[^11]。本章的目标，就是将这个模糊的临床需求（“减少晃动感”）[^1]，转化为可以计算、可以优化的数学模型。

## 9.2 多目标优化的“菜单”：帕累托前沿

当面对多个相互冲突的目标（例如：$f_1$ = 最小化中心像差，$f_2$ = 最小化周边晃动）时，“最佳”解的定义本身就发生了变化。我们不再寻找一个“最好”的设计，而是寻找一组“所有可能的、最好的权衡”[^11]。这一组解，在数学上被称为**帕累托前沿 (Pareto Front)**[^1]。

理解帕累托前沿，首先要理解“支配”（Dominance）的概念[^11]：
*   支配 (Dominance)：假设我们有两个设计方案 A 和 B。如果方案 A 在所有目标上都不劣于方案 B，并且在至少一个目标上严格优于方案 B，我们就说“A 支配 B”。
*   帕累托最优 (Pareto Optimal)：如果一个设计方案不被任何其他可行方案所支配，那么它就是帕累托最优解。这意味着，对于一个帕累托最优方案，你无法在不牺牲另一个目标性能的前提下，提升任何一个目标的性能[^11]。
*   帕累托前沿 (Pareto Front)：所有帕累托最优解在目标空间中构成的集合[^11]。

帕累托前沿为决策者（如产品经理或高级工程师）提供了一份“优化菜单”。如图9.1所示，这条曲线上的每一个点都是一个“最好”的设计。
*   A点：周边稳定性极佳（晃动感低），但中心清晰度较差。
*   B点：中心清晰度极佳，但周边晃动感强烈。
*   C点：一个在清晰度和稳定性之间的“平衡”设计。

选择A、B还是C，不再是一个纯粹的数学问题，而是一个基于产品定位、目标用户和临床反馈的商业决策。而我们作为工程师的核心任务（也是本章Python项目的目标），就是计算并绘制出这条帕累托前沿，将所有“A、B、C”选项清晰地呈现出来，制作一份“设计决策支持图表”[^1]。

## 9.3 理论基石：为什么我们需要“弱收敛”？

在第2周和第3周，我们学习了变分法和泛函梯度下降[^1]。我们建立了一个能量泛函（损失函数） $I[u]$，其中 $u$ 代表镜片曲面（一个函数），我们试图找到 $u^*$ 使得 $I[u]$ 最小。

现在，我们面临一个更深刻的数学问题：我们如何确保这个最小的 $u^*$ 真实存在？

在有限维度优化中（例如 $\min f(x_1, x_2)$），事情很简单：如果函数 $f$ 是连续的，并且我们在一个有界闭集上寻找最小值，Weierstrass定理保证最小值一定存在。但在泛函分析中，我们优化的“变量” $u$ 是一个函数，它位于一个无穷维的函数空间（如Hilbert空间或Banach空间）。

在无穷维空间中，经典的Weierstrass定理失效了。一个核心原因是，无穷维空间中的有界闭集（例如单位球）并不是紧致的（在强拓扑下）[^13]。

为了证明最小化器的存在性，数学家们采用了“变分法中的直接法”（Direct Method in the Calculus of Variations）[^14]。该方法遵循一个关键的逻辑链条[^13]：
1.  取最小化序列：我们从一个“最小化序列” $\{u_n\}$ 开始。这是一个函数序列，其泛函值（损失）收敛到总的下确界：$\lim_{n \to \infty} I[u_n] = \inf_u I[u]$。
2.  强制性 (Coercivity)：我们需要证明，如果 $I[u_n]$ 是有界的（最小化序列的泛函值当然是有界的），那么序列 $\{u_n\}$ 本身也必须在某个范数（如 $W^{1,p}$ 范数）下是有界的（Bounded）[^13]。这是我们设计损失函数时必须满足的一个关键条件。
3.  紧致性 (Compactness)：现在我们有了一个有界序列 $\{u_n\}$。在有限维空间中，这意味着我们可以提取一个强收敛的子序列。但在无穷维Banach空间中，这不成立[^13]。
4.  弱收敛出场：幸运的是，Banach-Alaoglu定理[^15]告诉我们：在一个自反的Banach空间中（我们常用的Sobolev空间 $W^{1,p}$ 和 $L^p$ 空间（当 $1 < p < \infty$）都是自反的[^16]），任何有界序列 $\{u_n\}$ 必定包含一个**弱收敛 (Weakly Convergent)** 的子序列 $\{u_{n_k}\}$。即存在一个极限函数 $u^*$，使得 $u_{n_k} \rightharpoonup u^*$。
5.  弱下半连续性 (W-LSC)：最后，我们需要证明我们的泛函 $I[u]$ 是“弱下半连续”的。这意味着，即使我们只有弱收敛，极限函数的泛函值也不会“逃逸”：$\liminf_{k \to \infty} I[u_{n_k}] \ge I[u^*]$。这通常通过保证泛函 $I$ 对其变量的最高阶导数是凸的（Convex）来实现[^13]。

### 本节核心：弱收敛在优化中的意义[^1]
弱收敛的意义在于，它是我们在无穷维函数空间中“强行”恢复紧致性的工具[^13]。强收敛的要求太高，以至于我们无法从有界序列中提取收敛子序列；而弱收敛的要求“恰到好处”的低，它利用了空间的拓扑结构（弱拓扑[^13]），确保了一个有界序列（由强制性保证）必定有一个极限点（我们的候选 $u^*$）。没有弱收敛，我们甚至无法从理论上证明一个“最优镜片设计”的存在性[^18]。

## 9.4 强收敛 vs. 弱收敛：一个直观对比

弱收敛的概念是泛函分析中最抽象但最关键的概念之一。本课程强调“让数学可见”[^1]，因此我们必须建立一个直观的理解。

定义对比[^20]：
*   **强收敛 (Strong Convergence)**：序列 $\{x_n\}$ 强收敛到 $x$，记作 $x_n \to x$。
    *   定义：$\lim_{n \to \infty} \|x_n - x\| = 0$
    *   直观意义：序列的“长度”或“能量”本身收敛到极限。$x_n$ “真正地”接近 $x$。
*   **弱收敛 (Weak Convergence)**：序列 $\{x_n\}$ 弱收敛到 $x$，记作 $x_n \rightharpoonup x$。
    *   定义：对于对偶空间 $X^*$ 中的每一个连续线性泛函 $f$，都有 $\lim_{n \to \infty} f(x_n) = f(x)$。
    *   直观意义：序列本身可能不会在范数意义下接近 $x$，但所有对它的“连续测量”（泛函 $f$ 就像一个测量工具）的结果都收敛了。

### 为什么弱收敛不意味着强收敛？
让我们看两个经典的例子，这也是我们Manim演示[^1]的核心内容：

**示例1：$L^2$ 空间中的振荡函数 (Manim演示 9.2a)**
*   空间：$X = L^2[0, 2\pi]$，即 $[0, 2\pi]$ 上的平方可积函数空间。
*   序列：$u_n(x) = \sin(nx)$[^22]。
*   分析：
    *   非强收敛：这个序列不强收敛到0。它的 $L^2$ 范数（能量）是一个常数：$\|u_n\|^2 = \int_0^{2\pi} \sin^2(nx) dx = \pi$。它离0的“距离”永远是 $\sqrt{\pi}$，从未减小。
    *   弱收敛：这个序列弱收敛到0函数。根据黎曼-勒贝格引理，对于任何“良好”的函数（泛函） $f(u) = \int_0^{2\pi} u(x)\phi(x) dx$（其中 $\phi \in L^2$），我们有 $\lim_{n \to \infty} \int_0^{2\pi} \sin(nx)\phi(x) dx = 0$。
    *   Manim直观演示：我们将看到 $u_n(x)$ 图像的振荡频率越来越快。它本身并未“消失”，而是在空间中“涂抹”得越来越均匀，导致其与任何固定函数 $\phi$ 的“重叠积分”（内积）都趋于0。

**示例2：$\ell^p$ 空间中的标准基 (Manim演示 9.2b)**
*   空间：$X = \ell^p$（$1 < p < \infty$），即 p-次可和的序列空间[^23]。
*   序列：$e_n = (0, 0, \dots, 1, 0, \dots)$，其中 1 在第 $n$ 个位置[^20]。
*   分析：
    *   非强收敛：这个序列不强收敛到0序列。它到0序列的范数（距离）始终为1：$\|e_n - 0\|_{\ell^p} = (1^p)^{1/p} = 1$[^20]。
    *   弱收敛：这个序列弱收敛到0序列[^20]。为什么？$\ell^p$ 空间的对偶空间是 $\ell^q$（其中 $\frac{1}{p} + \frac{1}{q} = 1$）。$\ell^q$ 空间中的任何泛函 $f$ 都可以表示为一个序列 $y = (y_i) \in \ell^q$。
        $f(e_n) = \sum_{i=1}^\infty (e_n)_i y_i = y_n$
        因为 $y \in \ell^q$，这意味着 $\sum |y_i|^q < \infty$，这必然要求 $\lim_{n \to \infty} y_n = 0$。因此，对于任何 $f \in X^*$，都有 $\lim_{n \to \infty} f(e_n) = \lim_{n \to \infty} y_n = 0 = f(0)$。
    *   Manim直观演示：我们将看到一个“脉冲”在序列中向右“移动”。这个脉冲的“能量”（范数）始终是1，但它“跑”到了无穷远处，因此任何固定的“测量”（只关心有限个坐标的 $y_n$）最终都会将其视为0。

## 9.5 策略与实践：设计多目标损失函数

理论保证了最优解的存在性，但我们如何在实践中找到它？我们必须设计一个损失函数（或一个算法）来计算帕累托前沿[^1]。

### 策略1：标量化（Scalarization）- 加权和法

这是最直观的方法：将多个目标通过加权的方式“合并”成一个单一的损失函数[^24]。

$$L_{\text{total}} = w_1 L_{\text{clarity}} + w_2 L_{\text{stability}}$$

其中 $w_1 + w_2 = 1$，$w_i > 0$。
*   工作原理：我们现在只有一个目标 $L_{\text{total}}$ 需要最小化。通过改变 $w_1$ 和 $w_2$ 的权重（例如，从 $(w_1=0.1, w_2=0.9)$ 扫到 $(w_1=0.9, w_2=0.1)$），我们希望能够描绘出整个帕累托前沿[^25]。
*   致命缺陷：加权和法有一个众所周知且在工程上非常危险的缺陷：它无法找到非凸（Non-Convex / Concave）区域的帕累托解[^12]。
*   Manim演示 9.3：如图9.2所示，加权和法 $L_{\text{total}}$ 在几何上等同于用一条斜率 $k = -w_1/w_2$ 的直线去“触碰”可行域。如果帕累托前沿是“向内凹”的（非凸），那么无论我们如何改变权重（改变直线斜率），这条直线都只会触碰到这个“凹坑”的两个端点，而永远无法找到凹坑内部的任何解[^12]。在镜片设计中，这种“凹坑”区域的解（可能是性能极佳的平衡点）将被完全错过。

### 策略2：$\epsilon$-约束法（Epsilon-Constraint Method）

这种方法更稳健，它能正确处理非凸问题[^8]。
*   工作原理：我们不再“合并”目标，而是只优化一个目标，同时将其余目标转化为约束[^26]。
*   $\min L_{\text{clarity}}$
*   约束条件：$L_{\text{stability}} \le \epsilon$
*   寻找前沿：通过在一个循环中“扫描” $\epsilon$ 的值（例如，从 $\epsilon = 0.5$ 逐步减小到 $\epsilon = 0.1$），我们可以逐点地描绘出整个帕累托前沿，包括非凸部分[^26]。
*   优势：非常稳健，可以控制帕累托前沿上点的分布和间距[^26]。

### 策略3：演化算法（Evolutionary Algorithms）- NSGA-II

这是目前多目标优化领域的“黄金标准”，也是我们将在Python项目中采用的方法[^28]。
*   工作原理：NSGA-II（Non-dominated Sorting Genetic Algorithm II）[^28] 是一种基于种群的遗传算法。它不像前两种方法那样一次只找一个点，而是同时维护一个完整的解集（种群）[^12]。
*   核心机制：
    1.  非支配排序（Non-Dominated Sorting）：在每一代中，算法根据帕累托支配关系将整个种群“分层”。第1层是当前最好的帕累托前沿，第2层是去掉第1层后剩余解的帕累托前沿，以此类推。
    2.  拥挤度计算（Crowding Distance）：为了保持解的多样性（即在帕累托前沿上均匀分布，而不是都挤在一起），算法会计算同一层中每个解的“拥挤度”。
    3.  选择：算法优先选择“层级更低”（更优）的解，在层级相同时，优先选择“拥挤度更低”（处于稀疏区域）的解。
*   优势：NSGA-II 旨在一次性找到整个帕累托前沿，并且通过拥挤度计算来确保解的多样性[^30]。它天生就是为多目标问题设计的，无需进行标量化或约束转化。

## 9.6 Python 实践项目 9：双焦点镜片平衡优化

目标：[^1]
1.  构建一个简化的双焦点镜片多目标损失函数。
2.  使用 `pymoo` 库和 NSGA-II 算法实现帕累托最优前沿的计算。
3.  使用 `matplotlib` 生成设计决策支持图表。

工具链：`numpy`, `scipy`, `matplotlib`, `pymoo`[^28]。

### 9.6.1 步骤1：定义 Problem

首先，我们需要使用 `pymoo` 的 `Problem` 接口来定义我们的优化问题[^30]。我们的“设计变量” $x$ 可以是一个向量，代表镜片曲面的一组Zernike系数或B样条控制点。

我们将定义两个目标（$f_1, f_2$），且都要求最小化：
*   $f_1(x)$：清晰度误差。模拟计算在“近用区”的光焦度，并计算其与目标值（例如 +2.0D）的均方根误差（RMSE）。
*   $f_2(x)$：稳定性误差。模拟计算镜片周边区域的“平均像散梯度”。梯度越大，“游泳效应”越强。

```python
import numpy as np
from pymoo.core.problem import Problem

# 假设我们有一个外部函数来模拟镜片性能
# from lens_simulator import simulate_lens_performance
def simulate_lens_performance(designs):
   # 输入: designs (N_POP, N_VARS)
   # 这是一个占位符。在实际应用中，这里会调用光学仿真引擎（如Zemax, Code V）
   # 或一个高性能的Python光线追迹库（如Optiland）
   
   # 模拟计算清晰度误差 (f1) 和 稳定性误差 (f2)
   # 我们使用随机数来模拟这个过程
   
   # f1: 清晰度误差 (0.0 到 1.5 之间)
   f1_error = 0.5 * (designs[:, 0] - 0.5)**2 + 0.8 * designs[:, 1]**2 + 0.1
   
   # f2: 稳定性误差 (0.0 到 1.5 之间)
   f2_error = 0.7 * (designs[:, 0] + 0.3)**2 + 0.5 * (designs[:, 1] - 0.7)**2 + 0.3
   
   # 模拟一些非线性和非凸特性
   f1_error = f1_error + 0.1 * np.sin(5 * designs[:, 0])
   f2_error = f2_error + 0.1 * np.cos(5 * designs[:, 1])
   
   return f1_error, f2_error

class BifocalOptimization(Problem):
   """
   定义双焦点镜片优化问题
   [^31, ^32]
   """
   def __init__(self, n_vars=2):
       # 我们用2个变量来模拟一个简化的设计空间
       super().__init__(n_var=n_vars,
                        n_obj=2,          # 2个目标: f1, f2
                        n_ieq_constr=0,   # 无约束
                        xl=-2.0,          # 设计变量下限
                        xu=2.0)           # 设计变量上限

   def _evaluate(self, x, out, *args, **kwargs):
       """
       评估函数，pymoo会批量调用此函数 [^31]
       x: (N_POP, N_VARS) 形状的NumPy数组，代表当前种群
       """
       
       # 1. 调用（模拟的）光学引擎计算性能
       f1_values, f2_values = simulate_lens_performance(x)
       
       # 2. 将结果打包
       # out["F"] 必须是 (N_POP, N_OBJ) 形状
       out["F"] = np.column_stack([f1_values, f2_values])

# 初始化问题
problem = BifocalOptimization(n_vars=2)
```

### 9.6.2 步骤2：初始化并运行 NSGA-II 算法

现在我们配置并运行 NSGA-II 算法[^33]。

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# 1. 初始化算法 [^28, ^33]
algorithm = NSGA2(
   pop_size=100,      # 种群大小
   n_offsprings=20,   # 每代产生的后代数量
   sampling=FloatRandomSampling(), # 初始种群采样器
   crossover=SBX(prob=0.9, eta=15),  # 模拟二进制交叉
   mutation=PM(eta=20),              # 多项式变异
   eliminate_duplicates=True
)

# 2. 定义终止条件 (运行50代)
termination = get_termination("n_gen", 50)

# 3. 运行优化！
print("--- 开始多目标优化 ---")
res = minimize(problem,
              algorithm,
              termination,
              seed=1,
              save_history=True,
              verbose=True)

print("--- 优化完成 ---")

# 4. 提取结果
# res.X 包含了帕累托最优解的 *设计变量* (在设计空间)
# res.F 包含了帕累托最优解的 *目标函数值* (在目标空间)
pareto_solutions = res.X
pareto_front = res.F

print(f"找到了 {len(pareto_front)} 个帕累托最优解")
```

### 9.6.3 步骤3：生成设计决策支持图表

最后，我们将计算得到的 `res.F` 绘制出来，这就是我们要交付的“决策支持图表”[^1]。

```python
import matplotlib.pyplot as plt

print("--- 绘制帕累托前沿 ---")

# 获取所有被评估过的点 (用于对比)
# 注意：这在真实运行中可能会非常多
all_evaluated_solutions_list = [h.pop.get("F") for h in res.history if h is not None]
all_evaluated_solutions = np.vstack(all_evaluated_solutions_list)


plt.figure(figsize=(12, 8))

# 绘制所有被评估过的点（灰色）
plt.scatter(all_evaluated_solutions[:, 0], all_evaluated_solutions[:, 1], 
           s=5, color='0.8', label='所有评估过的设计')

# 绘制最终的帕累托前沿（红色）
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
           s=40, facecolors='none', edgecolors='r', 
           linewidth=1.5, label='帕累托最优前沿')

# 标记两个极端点
idx_best_clarity = np.argmin(pareto_front[:, 0])
idx_best_stability = np.argmin(pareto_front[:, 1])

plt.scatter(pareto_front[idx_best_clarity, 0], pareto_front[idx_best_clarity, 1],
           s=100, c='b', marker='*', label='最佳清晰度 (A点)')
plt.scatter(pareto_front[idx_best_stability, 0], pareto_front[idx_best_stability, 1],
           s=100, c='g', marker='*', label='最佳稳定性 (B点)')

plt.title('设计决策支持图表：清晰度 vs. 稳定性')
plt.xlabel('f1: 清晰度误差 (越低越好)')
plt.ylabel('f2: 稳定性误差 (越低越好)')
plt.legend()
plt.grid(True)
plt.show()
```（预期输出：一张散点图，显示所有灰色的评估点，以及一条清晰的、由红色空心圆构成的帕累托前沿曲线，类似于图9.1。）

### 9.6.4 附录：用于数据分析的快速帕累托工具函数

`pymoo` 用于寻找最优解。但在实际工作中，我们有时已经有了一大批仿真数据（例如，蒙特卡洛分析的10000个设计），我们只想筛选出其中的帕累托解。此时，使用一个高效的 numpy 函数会更方便[^33]。

```python
import numpy as np

def find_pareto_efficient(costs):
   """
   在一个(N, M)的成本数组中快速找到帕累托有效点 (假设所有目标都是最小化)
   
   :param costs: 一个 (n_points, n_costs) 的NumPy数组
   :return: 一个 (n_points,) 的布尔数组，True代表该点是帕累托最优的
   """
   is_efficient = np.ones(costs.shape[0], dtype = bool)
   for i, c in enumerate(costs):
       if is_efficient[i]:
           # 找出所有被 c 支配的点
           is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1) | np.all(costs[is_efficient]==c, axis=1)
           is_efficient[i] = True  # And keep self
   return is_efficient

# --- 使用示例 ---
# 假设我们有一批仿真数据 (10000个设计, 2个目标)
# all_simulation_results = np.random.rand(10000, 2)
# pareto_mask = find_pareto_efficient(all_simulation_results)
# analysis_pareto_front = all_simulation_results[pareto_mask]

# import matplotlib.pyplot as plt
# plt.scatter(all_simulation_results[:, 0], all_simulation_results[:, 1], s=1, c='gray')
# plt.scatter(analysis_pareto_front[:, 0], analysis_pareto_front[:, 1], s=10, c='red')
# plt.show()
```

## 9.7 Manim 可视化演示

为辅助理解本章的三个核心概念，我们提供以下三个Manim动画演示[^1]：
*   **Manim演示 9.1：帕累托前沿的动态生成**
    *   内容：在一个2D目标空间（$f_1$ vs $f_2$）中，动画展示NSGA-II的优化过程。
    *   视觉效果：
        1.  （第0代）大量蓝色圆点（初始种群）随机分布在空间中。
        2.  （第1-10代）圆点开始向左下角（最优方向）“漂移”和“收缩”。
        3.  （第10-30代）圆点逐渐“贴合”到一条不可见的帕累托前沿曲线上。
        4.  （第30-50代）算法的“拥挤度”机制开始发挥作用，圆点在曲线上均匀散开，填满整个前沿。
    *   教学目标：直观理解演化算法是如何“同时”优化一个解集，而不是一个单点的。
*   **Manim演示 9.2：弱收敛 vs. 强收敛对比**
    *   内容：并排展示两个序列的收敛行为（参见 9.4 节的分析）。
    *   视觉效果：
        1.  左侧（弱收敛）：展示 $u_n(x) = \sin(nx)$。随着 $n$ 增大，函数图像的振荡频率越来越高。下方同时显示其范数 $\|u_n\| = \sqrt{\pi}$ 始终不变。再下方显示一个“测量值” $f(u_n) = \int \sin(nx)\phi(x)dx$，该值收敛到0。
        2.  右侧（强收敛）：展示 $v_n(x) = \frac{1}{n}\sin(x)$。随着 $n$ 增大，函数图像的振幅衰减到0。下方显示其范数 $\|v_n\|$ 收敛到0。
    *   教学目标：建立“范数不收敛，但测量值收敛”的弱收敛直观印象。
*   **Manim演示 9.3：权重调整的实时效果（加权和法的陷阱）**
    *   内容：展示加权和法在非凸问题上的失败（参见 9.5 节的分析）。
    *   视觉效果：
        1.  场景中有一条非凸（向内凹）的帕累托前沿曲线。
        2.  一条直线（代表 $L_{\text{total}}$）“靠”在曲线上。
        3.  一个Manim滑块控制权重 $w_1$。当观众拖动滑块时，直线的斜率随之改变。
        4.  观众会清晰地看到，当直线“滚”过凹陷区域时，接触点会从凹陷的一端（例如A点）瞬间跳跃到另一端（例如B点），而凹陷内部的所有解都无法通过改变权重被触及。
    *   教学目标：视觉上“证明”为什么加权和法是一种有缺陷的策略，以及为什么我们需要 $\epsilon$-约束法 或 NSGA-II。

## 9.8 本章总结

本章是连接泛函分析理论与眼视光学工程实践的关键桥梁。
1.  **临床问题**：我们从渐进镜片佩戴者的核心痛点——“游泳效应”[^6]出发，并将其科学地确认为由镜片周边像散引起的“失真光学流”[^7]。
2.  **数学建模**：我们将“减少晃动感”和“提升清晰度”这一临床权衡[^2]建模为一个多目标优化（MOO）问题[^11]。
3.  **解决方案定义**：我们明确了MOO的解不是一个点，而是一个帕累托前沿[^11]。它为决策者提供了所有最优权衡的“菜单”。
4.  **理论存在性**：我们深入探讨了为什么需要弱收敛[^1]。在处理镜片曲面（函数）这类无穷维变量时，经典的强收敛无法保证最小解的存在性。我们必须依赖“强制性 $\to$ 有界序列 $\to$ 弱紧致性 $\to$ 弱收敛子序列”[^15]这一泛函分析的“黄金链条”来证明解的存在。
5.  **算法策略**：我们对比了三种寻找帕累托前沿的策略。我们揭示了加权和法在非凸问题上的致命缺陷[^12]，并转向了更稳健的**$\epsilon$-约束法**[^26]和高效的演化算法（NSGA-II）[^28]。
6.  **Python实践**：我们使用 `pymoo` 库[^28]和 NSGA-II 算法，从零开始构建了一个简化的双焦点镜片优化器，并成功生成了可视化的**“设计决策支持图表”**[^1]，完成了从临床需求到可执行代码的完整闭环。

通过本章学习，学员应掌握将一个复杂的、主观的工程权衡问题，转化为一个定义明确、理论上存在解、并且在实践中（使用Python）可解的多目标优化问题的完整流程。

---
### 引用的著作

[^1]: 《实用泛函分析：Python驱动的眼视光学镜片设计优化》课程大纲
[^2]: [Continuing Education Course - OptiCampus.com](https://opticampus.opti.vision/popcourse.php?url=progressive_lenses/)
[^3]: [Multifocal (Progressive) Lens: Most Common problems and solutions - Rx optometry](https://rxoptometry.com/multifocal-progressive-lens-most-common-problems-and-solutions/)
[^4]: [Progressive Lenses: How They Work and Which Lens Options Might Fit Your Life - Tayani Institute](https://tayani.com/progressive-lenses-how-they-work-and-which-lens-options-might-fit-your-life)
[^5]: [Are Your Progressive Lenses Putting You at Risk? Here's What You Need to Know - My Vision Expert](https://myvisionexpert.com/2025/06/13/are-your-progressive-lenses-putting-you-at-risk-heres-what-you-need-to-know/)
[^6]: [IOT Free-Form Insights Part 16 - 20/20 Magazine](https://www.2020mag.com/article/iot-freeform-insights-part-16)
[^7]: [Self-motion illusions from distorted optic flow in multifocal glasses ... - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC8693457/)
[^8]: [Understanding Today's Progressives - Review of Optometry](https://www.reviewofoptometry.com/article/understanding-todays-progressives)
[^9]: [Lesson: Unleash the Power of the Corridor - 20/20 Magazine](https://www.2020mag.com/nysso/ce/lesson/unleash-the-power-of-the-1FC01)
[^10]: [A NUMERICAL METHOD FOR PROGRESSIVE LENS DESIGN - ResearchGate](https://www.researchgate.net/publication/263988734_A_NUMERICAL_METHOD_FOR_PROGRESSIVE_LENS_DESIGN)
[^11]: [Multi-objective optimization - Wikipedia](https://en.wikipedia.org/wiki/Multi-objective_optimization)
[^12]: [Lecture 9: Multi-Objective Optimization - Purdue Engineering](https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf)
[^13]: [EXISTENCE OF GLOBAL MINIMUM FOR ... - UCI Mathematics](https://www.math.uci.edu/~chenlong/290C/3_Existence.pdf)
[^14]: [Calculus of variations - Wikipedia](https://en.wikipedia.org/wiki/Calculus_of_variations)
[^15]: [Lectures Notes Calculus of Variations - University of Hamburg](https://www.math.uni-hamburg.de/home/schmidt/lectures/CalcVar.pdf)
[^16]: [Lp space - Wikipedia](https://en.wikipedia.org/wiki/Lp_space)
[^17]: [Weak convergence in $L^p$ spaces their dual - Mathematics Stack Exchange](https://math.stackexchange.com/questions/4913396/weak-convergence-in-lp-spaces-their-dual)
[^18]: [Weak Convergence of Integrands and the Young Measure Representation - SIAM.org](https://epubs.siam.org/doi/10.1137/0523001)
[^19]: [Existence of minimisers for variational problems: Relaxing the coercivity condition (following the L. Evens book) - Math Stack Exchange](https://math.stackexchange.com/questions/4120707/existence-of-minimisers-for-variational-problems-relaxing-the-coercivity-condit)
[^20]: [functional analysis - What is the difference between weak and strong ... - Math Stack Exchange](https://math.stackexchange.com/questions/1769960/what-is-the-difference-between-weak-and-strong-convergence)
[^21]: [functional analysis lecture notes: weak and weak* convergence - Christopher Heil](https://heil.math.gatech.edu/handouts/weak.pdf)
[^22]: [11.4 Weak and strong convergence - Numerical Analysis II - Fiveable](https://fiveable.me/numerical-analysis-ii/unit-11/weak-strong-convergence/study-guide/E8MG9rOc5tF0zPJC)
[^23]: [Weak convergence in Banach Spaces - Joel H. Shapiro](https://joelshapiro.org/Pubvit/Downloads/joel_wktop.pdf)
[^24]: [Multi-Objective Optimization for Deep Learning : A Guide ... - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/multi-objective-optimization-for-deep-learning-a-guide/)
[^25]: [Strategies for Balancing Multiple Loss Functions in Deep Learning | by Baicen Xiao - Medium](https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0)
[^26]: [Multiobjective optimization - OpenMDAO](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/multiobjective.html)
[^27]: [A Weighted and Epsilon-Constraint Biased-Randomized Algorithm for the Biobjective TOP with Prioritized Nodes - MDPI](https://www.mdpi.com/2079-3197/12/4/84)
[^28]: [pymoo: Multi-objective Optimization in Python](https://pymoo.org/)
[^29]: [COIN Report Number 2020001 pymoo: Multi-objective Optimization in Python - MSU College of Engineering](https://www.egr.msu.edu/~kdeb/papers/c2020001.pdf)
[^30]: [pymoo - Getting Started - Michigan State University](https://www.egr.msu.edu/coinlab/blankjul/pymoo-rc/getting_started.html)
[^31]: [Preface: Basics and Challenges - pymoo](https://pymoo.org/getting_started/preface.html)
[^32]: [Part II: Find a Solution Set using Multi-objective Optimization - pymoo](https://pymoo.org/getting_started/part_2.html)
[^33]: [numpy - Fast calculation of Pareto front in Python - Stack Overflow](https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python)