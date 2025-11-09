# 第7章：非线性算子与个性化设计

**表 7.1：第7周 学习目标与实践**

| 主题 | Manim动画演示内容 | Python实践项目 | 课时分配 |
| :--- | :--- | :--- | :--- |
| 非线性算子与个性化设计 | Manim演示7：不动点迭代动画<br>• 固定点定理在逆向设计中的应用<br>• 固定点定理的几何解释<br>• 迭代收敛过程可视化<br>• 收敛性保障策略<br>• 发散情况的对比演示<br>• 扩展：用Manim展示设计流程 | 项目7：个性化渐进镜片原型<br>• 基于模拟患者数据<br>• 患者数据驱动设计流程<br>• 实现逆向设计算法 | 理论: 20% (Manim动画)<br>实践: 80% |

## 7.1 引言：从标准化到个性化

在前面的模块中，我们使用线性算子（第5周）和积分方程（第6周）成功地对光学系统中的像差和涂层进行了建模与优化。这些工具在处理定义明确、约束良好的“标准化”问题时非常强大。

然而，眼视光学设计的未来在于“个性化”[^1]。一名60岁的钢琴演奏家对中距离（乐谱）和远距离（指挥）的视觉要求，与一名30岁的软件工程师对近距离（屏幕）和中距离（会议室）的要求截然不同。他们无法用同一款“渐进多焦点”镜片满足需求。

这种个性化需求（例如，特定的“目标离焦曲线”）向我们提出了一个全新的挑战[^1]。我们不再是简单地“最小化”某个像差，而是要求解一个复杂的目标：“是否存在一个镜片曲面，使其恰好产生患者所需的目标视觉效果？”

这个问题将我们从线性优化的世界，带入了非线性算子的领域。本章将介绍如何使用“不动点迭代”这一强大的数学工具，结合Python（特别是SciPy库），来解决这类个性化的“逆向设计”问题。

## 7.2 临床驱动：为什么线性模型会失效？

线性模型（如我们在第5周探讨的）假设系统的响应与输入成正比。这在小扰动（如轻微的离焦或像散）分析中是成立的。然而，渐进镜片的设计从根本上就是非线性的。

1.  复杂的曲面与光线追迹：渐进镜片的曲面是复杂的自由曲面。光线通过镜片不同位置的折射率和曲率变化是剧烈的。如研究所示，即使是主曲率从一个光学表面到下一个表面的传递，也“显然是一个非线性操作”[^2]。
2.  “精确”而非“近似”：传统的三阶理论（一种线性近似）已不足以描述现代镜片性能[^3]。现代设计依赖于“精确光线追迹”和“多参数优化”[^3]。
3.  高度耦合的方程：当我们试图同时优化远、中、近三个区域的视觉质量，并最小化过渡区的“晃动感”（像散）时，这些目标是相互冲突和高度耦合的。描述这个问题的变分法，最终会导出一个“四阶非线性椭圆偏微分方程”[^4]。

简而言之，当临床需求从“修正单一缺陷”变为“定制完整视觉体验”时，我们必须面对系统固有的非线性。

## 7.3 核心问题：光学“逆向设计”

我们面临的问题在数学上被称为“逆向问题”（Inverse Problem）[^5]。

*   正向问题（我们熟悉的）：给定一个镜片设计 $x$（曲面参数、材料），通过复杂的物理模拟（如光线追迹），计算出其光学性能 $y$（如MTF、离焦曲线）。
    *   在数学上，我们写作：$y = T(x)$
    *   这里，$x$ 是我们设计的函数（如镜片矢高函数），$y$ 是结果（如视觉质量分布），而 $T$ 是一个复杂的非线性算子（它代表了光线追迹的整个物理过程）。
*   逆向问题（我们的新挑战）：给定一个期望的光学性能 $y_{target}$（来自患者的个性化需求，如“在40cm处有+2.0D的清晰视力”）[^1]，反向求解能产生这种性能的镜片设计 $x$。
    *   在数学上，我们要解方程：$T(x) = y_{target}$

这个挑战正如研究中指出的：在设计光学组件时，“如果电磁场（$y_{target}$）是已知的，但周围的结构（$x$）是未知的”，那么逆向求解器将是“更擅长此类设计和优化问题”的工具[^5]。

问题在于，非线性算子 $T$ 几乎不可能（也没有解析方法）去求它的逆 $T^{-1}$。我们不能通过 $x = T^{-1}(y_{target})$ 来直接计算答案。我们需要一种更巧妙的迭代方法。

## 7.4 解决方案：不动点定理

这就是泛函分析展现其威力的时刻。我们将使用一个强大的数学“诡计”，将这个无法求解的逆向问题，转化为一个可以迭代求解的“不动点”问题。

不动点（Fixed Point） 的定义是：对于一个函数（或算子）$G$，如果一个点 $x_{fp}$ 满足 $G(x_{fp}) = x_{fp}$，那么 $x_{fp}$ 就是 $G$ 的一个不动点[^6]。

转化的过程如下：
1.  我们的目标是求解：
    $$T(x) = y_{target}$$
2.  我们将它改写为“残差”形式，目标是让残差为零：
    $$T(x) - y_{target} = 0$$
3.  现在，我们定义一个新的迭代函数 $G(x)$。一个简单的（但后面会改进的）定义是：
    $$G(x) = x - (T(x) - y_{target})$$
    (注意：这里的减法是一个简化的表示法。在实际工程中，我们会使用一个映射算子 $L$ 将“视觉误差” $T(x) - y_{target}$ 转换回“参数校正量”。为清晰起见，本章暂用简化形式。)
4.  请注意这个新函数 $G(x)$ 的神奇之处：
    *   假设我们找到了一个解 $x_{solution}$，它使得 $T(x_{solution}) = y_{target}$。
    *   那么 $T(x_{solution}) - y_{target} = 0$。
    *   把 $x_{solution}$ 代入 $G(x)$：
        $$G(x_{solution}) = x_{solution} - (T(x_{solution}) - y_{target}) = x_{solution} - 0$$
    *   因此：
        $$G(x_{solution}) = x_{solution}$$

结论：
我们成功地将“求解 $T(x) = y_{target}$”这个困难的逆向问题，等价转换为了“寻找 $G(x)$ 的不动点”[^8]。

为什么这样做？因为寻找不动点有现成的、强大的数值算法，即不动点迭代（Fixed-Point Iteration）[^6]。

这个算法非常简单：
1.  从一个初始猜测 $x_0$ 开始（例如一个标准的镜片设计）。
2.  反复应用 $G$ 算子：$x_{n+1} = G(x_n)$。
3.  如果这个迭代序列 $x_0, x_1, x_2, \dots$ 收敛，它的极限 $x_{fp}$ 必定满足 $x_{fp} = G(x_{fp})$[^6]。

这个 $x_{fp}$ 就是我们梦寐以求的、满足患者需求的个性化镜片设计 $x_{solution}$。

## 7.5 Manim 可视化：“看见”不动点

不动点迭代的收敛性在视觉上有一个非常直观的解释。本课程的核心教学工具 Manim 将为我们“点亮”这个抽象概念[^10]。

### 7.5.1 不动点的几何意义

在一个一维系统（$x$ 轴代表输入， $y$ 轴代表输出）中，不动点的几何意义是什么？

*   一个点 $x_{fp}$ 是 $G(x)$ 的不动点，意味着 $G(x_{fp}) = x_{fp}$。
*   在图上，这对应着函数 $y = G(x)$ 的曲线，与 $y = x$ 这条对角线相交的点[^11]。

寻找不动点，就是寻找 $y = G(x)$ 和 $y = x$ 的交点。

### 7.5.2 迭代法与“蛛网图”（Cobweb Plot）

不动点迭代 $x_{n+1} = G(x_n)$ 的过程，可以在这张图上被可视化为“蛛网图”（Cobweb Plot）或“阶梯图”[^13]。

其绘制过程如下[^14]：
1.  起始：从任意一个初始猜测 $x_0$ 开始。
2.  步骤 1（垂直线）：从 $x$ 轴上的 $(x_0, 0)$ 点出发，垂直向上（或下）画线，直到碰到 $y = G(x)$ 的曲线，交点为 $(x_0, G(x_0))$。
3.  步骤 2（水平线）：$y$ 值 $G(x_0)$ 将成为我们的下一个 $x$ 值，即 $x_1 = G(x_0)$。为了在图上表示这个“传递”，我们从 $(x_0, G(x_0))$ 点出发，水平画线，直到碰到 $y = x$ 的对角线，交点为 $(G(x_0), G(x_0))$，即 $(x_1, x_1)$。
4.  重复：从 $(x_1, x_1)$ 点出发，垂直画线到 $y = G(x)$ 曲线（点 $(x_1, G(x_1))$），再水平画线到 $y = x$ 对角线（点 $(x_2, x_2)$）。

这个蛛网的轨迹，就是不动点迭代 $x_0, x_1, x_2, \dots$ 的完整视觉记录。

### 7.5.3 Manim 演示7：收敛 vs 发散

现在，我们将使用 Manim 制作本周的核心动画：不动点迭代[^10]。我们将演示三种关键情况：

**演示 1：收敛（压缩映射）**

*   条件：当 $y = G(x)$ 曲线在不动点 $x_{fp}$ 附近的斜率（导数）的绝对值小于1时，即 $|G'(x_{fp})| < 1$。
*   数学术语：这个 $G(x)$ 被称为“压缩映射”（Contraction Mapping）。著名的**巴拿赫不动点定理（Banach Fixed-Point Theorem）**保证，对于一个完备度量空间上的压缩映射，不动点存在、唯一，并且任何初始猜测 $x_0$ 出发的迭代必定收敛到该不动点[^6]。
*   Manim 动画：我们将看到“蛛网”不断向内盘旋，最终螺旋收敛到 $y = G(x)$ 和 $y = x$ 的交点。

**演示 2：发散（扩张映射）**

*   条件：当不动点附近的斜率绝对值大于1时，即 $|G'(x_{fp})| > 1$。
*   Manim 动画：我们将看到“蛛网”不断向外扩散，离交点越来越远。迭代序列发散，我们永远无法找到解。

**演示 3：震荡发散**

*   条件：当不动点附近的斜率为负数且绝对值大于1时，例如 $G'(x_{fp}) = -2$。
*   Manim 动画：我们将看到“蛛网”在一个“盒子”里向外震荡，同样无法收敛[^15]。

这个 Manim 演示清晰地告诉我们一个工程结论：算法的收敛性，完全取决于我们构造的 $G(x)$ 算子在解附近的“斜率”（导数）[^17]。

## 7.6 Manim 实践：制作不动点迭代动画

作为本周的实践项目，学员将利用 Manim 复现 7.5.3 节中的蛛网图动画[^10]。

### 7.6.1 场景设置

1.  坐标系与函数：使用 `Axes` 类创建坐标系[^18]。
2.  函数图像：使用 `axes.plot(lambda x: G(x))` 来绘制 $y = G(x)$ 的曲线。
3.  对角线：使用 `axes.plot(lambda x: x, line_style="dashed")` 绘制 $y = x$ 的对角线。

### 7.6.2 绘制“蛛网”路径

我们将使用 Manim 的 `.animate` 语法[^19] 和一个循环来动态生成蛛网。
```python
from manim import *

class CobwebPlot(Scene):
   def construct(self):
       # 1. 设置坐标系和函数
       axes = Axes(
           x_range=[-1, 1], y_range=[-1, 1],
           x_length=6, y_length=6,
           axis_config={"color": BLUE},
       )
       # 定义一个收敛的 G(x)
       def G(x):
           return 0.5 * x + 0.25 # 斜率 |G'| = 0.5 < 1

       g_graph = axes.plot(G, color=YELLOW)
       diag_line = axes.plot(lambda x: x, color=WHITE, line_style="dashed")
       
       self.add(axes, g_graph, diag_line)

       # 2. 迭代绘制蛛网
       x_val = 0.1  # 初始猜测 x0
       x_tracker = ValueTracker(x_val)
       
       path = VMobject(color=RED) # 用于存储路径
       path.set_points_as_corners([axes.c2p(x_val, 0)])

       self.add(path)

       for _ in range(10): # 迭代10次
           x_old = x_tracker.get_value()
           y_new = G(x_old)
           
           # (x0, 0) -> (x0, y1)
           p1 = axes.c2p(x_old, y_new)
           # (x0, y1) -> (y1, y1)
           p2 = axes.c2p(y_new, y_new)

           # 播放动画
           #.animate 语法用于平滑地修改对象 
           self.play(path.animate.add_points_as_corners([p1, p2]),
                     run_time=0.5)
           
           x_tracker.set_value(y_new) # 更新 x 值为 y_new
           
       self.wait()

```
（注：上述代码为 Manim 社区版 v0.18.0 语法）

### 7.6.3 (高级) 使用 add_updater 动态跟踪

一个更高级的 Manim 技术是使用 `add_updater`[^18]。学员可以尝试定义一个 `Dot`（点），并让 `path`（路径）Mobject 自动跟踪这个点的位置，从而动态绘制出蛛网图。这在 manim_utils/animation_templates.py 中会有模板提供[^10]。

## 7.7 Python 实践：用 SciPy 实现个性化设计

理论和可视化已经就绪，现在我们回到真正的工程问题：为模拟患者数据（simulated_patient_data.csv）[^10] 设计一个个性化渐进镜片。

### 7.7.1 构建迭代函数 G(x)

我们的目标是求解 $T(x) = y_{target}$。如 7.4 节所述，我们需要构建迭代函数 $G(x) = x - L(T(x) - y_{target})$，其中 $L$ 是一个将“视觉误差”映射回“参数校正”的算子。

在实践中，$T$ 是一个复杂的光学模拟函数（可能来自 Optiland[^20] 或其他光追库），而 $L$ 可能是 $T$ 的（简化的）逆。

伪代码（Pythonic Pseudocode）:
```python
import numpy as np
import pandas as pd
from scipy.optimize import fixed_point
from utils.optical_simulators import calculate_power_profile # T(x)
from utils.inverse_mappings import map_error_to_params     # L(...)

# --- 1. 加载患者数据 (y_target) ---
# 加载模拟的患者数据，例如目标离焦曲线[^25]
try:
   patient_data = pd.read_csv('simulated_patient_data.csv')
   y_target = patient_data['target_defocus_curve'].values
except FileNotFoundError:
   print("模拟数据文件未找到，使用标准数据。")
   y_target = np.linspace(0, 2.0, 100) # 示例目标：从0到+2.0D的平滑渐进

# --- 2. 定义正向算子 T(x) ---
# x 是一组描述镜片曲面的 Zernike 系数或样条点
# T(x) 是一个昂贵的计算，它模拟光线追迹
def T_operator(lens_params_x):
   # 这是我们光学设计的核心
   # T(x) -> y_current
   #  这里的实现细节很复杂，我们先将其作为黑盒
   power_profile_y = calculate_power_profile(lens_params_x)
   return power_profile_y

# --- 3. 定义 G(x) ---
# 这是逆向设计的核心[^5][^21]
def G_iteration_function(x):
   # (a) 计算当前设计的光学性能
   y_current = T_operator(x)
   
   # (b) 计算性能误差
   error_in_vision = y_current - y_target
   
   # (c) 将视觉误差 "映射" 回参数空间的校正量
   # 这是 G(x) = x - L(T(x) - y_target) 中的 L(...) 部分
   parameter_correction = map_error_to_params(error_in_vision)
   
   # (d) 返回新的、“更好”的参数估计
   # G(x) = x_new
   return x - parameter_correction
```

### 7.7.2 高效实现：使用 scipy.optimize.fixed_point

有了 `G_iteration_function(x)`，在 SciPy 中求解不动点（即我们的镜片设计）变得异常简单。我们不需要自己编写迭代循环，而是直接调用 `scipy.optimize.fixed_point`[^8]。

这个函数非常强大。它不仅仅是一个简单的 while 循环；如 SciPy 文档所述，它默认使用 `method='del2'`，即带Aitken's Del^2 加速的Steffensen方法[^22]。这是一种高阶的迭代加速技术，比我们手写的循环收敛得快得多。

Python 实践代码 (项目7):
```python
# --- 4. 求解不动点 ---
# 初始猜测 (x0): 可以是一个标准镜片或零向量
x0 = np.zeros_like(y_target) 

print("开始不动点迭代求解 (使用 SciPy)...")

#  调用 SciPy 的不动点求解器
# func: 迭代函数 G(x)
# x0: 初始猜测
# xtol: 收敛容差
# maxiter: 最大迭代次数
solution_params_x = fixed_point(
   G_iteration_function, 
   x0, 
   xtol=1e-8, 
   maxiter=500
)

print("--- 求解完成 ---")
print(f"找到的镜片设计参数 (x_fp): {solution_params_x[:5]}...")

# 5. (验证)
final_profile_y = T_operator(solution_params_x)
final_error = np.linalg.norm(final_profile_y - y_target)
print(f"最终设计与目标的误差: {final_error}")
```

### 7.7.3 (选学) 手动实现：Picard 迭代循环

为了深入理解 `fixed_point` 内部在做什么，我们可以手动实现最简单的不动点迭代，即Picard 迭代[^23]。
```python
def manual_picard_iteration(G_func, x0, xtol=1e-8, maxiter=500):
   x_n = x0
   for i in range(maxiter):
       #  Picard 迭代: x_n+1 = G(x_n)
       x_n_plus_1 = G_func(x_n)
       
       # 计算步长（误差）
       step_size = np.linalg.norm(x_n_plus_1 - x_n)
       
       if step_size < xtol:
           print(f"Picard 迭代在 {i+1} 次后收敛。")
           return x_n_plus_1
       
       x_n = x_n_plus_1
       
   print(f"Picard 迭代在 {maxiter} 次后 *未收敛*。")
   return x_n
```
将这个手动实现与 `scipy.optimize.fixed_point` 对比，学员会发现，对于“良好”的 `G(x)`，两者都能收敛，但 SciPy 的版本通常快得多。

## 7.8 关键挑战：迭代不收敛怎么办？

在 7.7.3 节的实践中，学员很快会遇到一个灾难性的问题：对于大多数真实且复杂的 `T(x)` 算子，`manual_picard_iteration` 甚至 `scipy.optimize.fixed_point` 都会发散！

程序将陷入无限循环，或者 `step_size` 越来越大，最终导致 NaN（非数）溢出。

为什么？

这正是 7.5.3 节的 Manim 演示（演示2和3）中预言的情况。

我们的迭代失败，不是因为 Python 代码有 bug，而是因为我们构建的 $G(x)$ 算子在数学上是一个扩张映射（$|G'(x)| > 1$）[^15]。

正如研究中警告的，许多迭代方法在设计时“没有考虑收敛特性”[^24]。而在工程实践中，“只有少数函数具有自然度量下的压缩特性”[^16]。

我们的 $G(x)$ “太积极”了。当它计算出一个小的 `error_in_vision` 时，它生成的 `parameter_correction` 可能“矫枉过正”，导致下一步的误差比上一步更大，从而引发雪崩式的发散。

## 7.9 收敛性保障策略：松弛（Relaxation）方法

面对 7.8 节的“发散”挑战，我们是否必须放弃不动点法？

完全不必。 我们只需要对算法进行一个简单的“工程改造”：松弛（Relaxation）。

松弛法的核心思想是：既然 $G(x)$ 算子给出的“下一步” $x_{n+1}$ 太激进，我们就“不完全相信它”。我们只朝着它建议的方向“挪动一小步”。

具体来说，我们引入一个松弛因子 $\alpha$（alpha），其中 $0 < \alpha < 1$。

标准的 Picard 迭代是[^23]：
$$x_{n+1} = G(x_n)$$
松弛的 Picard 迭代 变为[^23]：
$$x_{n+1} = \alpha \cdot G(x_n) + (1 - \alpha) \cdot x_n$$

*   当 $\alpha = 1$ 时，它就是标准 Picard 迭代。
*   当 $\alpha = 0.1$ 时，我们是说：“新的 $x_{n+1}$ 应该是 $10\%$ 的‘建议步’（$G(x_n)$）和 $90\%$ 的‘当前位置’（$x_n$）的加权平均。”

这是一种“欠松弛”（Under-relaxation）[^17]，它极大地提高了迭代的稳定性。

为什么松弛法在数学上是有效的？

这是本章最关键的工程洞察。松弛法不仅仅是“减慢速度”；它改变了迭代算子。

我们定义了一个新的迭代算子 $G_{new}(x)$：
$$G_{new}(x) = \alpha \cdot G(x) + (1 - \alpha) \cdot x$$

1.  它是否有相同的不动点？
    *   是的。假设 $x_{fp}$ 是 $G(x)$ 的不动点（即 $G(x_{fp}) = x_{fp}$）。
    *   那么 $G_{new}(x_{fp}) = \alpha \cdot G(x_{fp}) + (1 - \alpha) \cdot x_{fp} = \alpha \cdot x_{fp} + x_{fp} - \alpha \cdot x_{fp} = x_{fp}$。
    *   $G_{new}(x)$ 的不动点与 $G(x)$ 完全相同。我们仍然在求解同一个镜片设计问题。
2.  它的收敛性（导数）呢？
    *   根据链式法则，新算子的导数是：
        $$G'_{new}(x) = \alpha \cdot G'(x) + (1 - \alpha)$$
    *   现在，假设我们原来的算子 $G(x)$ 是震荡发散的，比如在不动点附近 $G'(x_{fp}) = -3$（绝对值大于1，导致发散）。
    *   我们选择 $\alpha = 0.1$。
    *   新算子的导数变为：
        $$G'_{new}(x_{fp}) = (0.1) \cdot (-3) + (1 - 0.1) = -0.3 + 0.9 = 0.6$$
    *   奇迹发生了！ 新算子的导数 $|G'_{new}(x_{fp})| = |0.6| < 1$。
    *   通过引入 $\alpha = 0.1$ 的松弛，我们把一个发散的迭代算子，转换成了一个必定收敛的压缩映射[^6]。

Python 实践代码 (最终版)[^17][^23]:
```python
def robust_manual_iteration(G_func, x0, alpha=0.1, xtol=1e-8, maxiter=2000):
   """
   使用松弛（Relaxation）的手动不动点迭代
   """
   x_n = x0
   for i in range(maxiter):
       # 1. 计算 G(x) 的“目标步骤”
       G_x_n = G_func(x_n)
       
       # 2.  应用松弛：x_n+1 = alpha*G(x_n) + (1-alpha)*x_n
       x_n_plus_1 = alpha * G_x_n + (1 - alpha) * x_n
       
       step_size = np.linalg.norm(x_n_plus_1 - x_n)
       
       if step_size < xtol:
           print(f"带松弛 (alpha={alpha}) 的迭代在 {i+1} 次后收敛。")
           return x_n_plus_1
       
       x_n = x_n_plus_1
       
   print(f"带松弛 (alpha={alpha}) 的迭代在 {maxiter} 次后 *未收敛*。")
   return x_n

# --- 5. 使用稳健的求解器 ---
# 当 7.7.2 中的 `fixed_point` 失败时，
# 工程师会转而使用这种更可控的松弛迭代。
solution_params_x = robust_manual_iteration(G_iteration_function, x0, alpha=0.05)
```

## 7.10 本章小结：连接临床、数学与代码的桥梁

本章（第7周）是本课程（《实用泛函分析》）的转折点。我们完成了从“分析”到“设计”的飞跃。

我们从一个纯粹的临床需求（“个性化渐进镜片”）出发[^1]，认识到已有的线性工具不足以解决问题[^2]。

我们应用泛函分析，将这个复杂的非线性逆向设计问题（$T(x) = y_{target}$）[^5]，通过数学变换，重构为一个不动点问题（$G(x) = x$）[^6]。

我们使用 Manim 这一可视化工具[^10]，通过“蛛网图”[^13]，直观地理解了迭代收敛的几何本质——压缩映射（$|G'| < 1$）。

最后，我们回到 Python，使用 `scipy.optimize.fixed_point`[^22] 实现了高效求解，并掌握了当迭代发散时，使用松弛法（Relaxation）[^23] 作为“工程安全网”来保障收敛。

学员现在已经掌握了从临床需求出发，建立非线性数学模型，并编写稳健的 Python 代码来自动生成全新设计原型的完整工作流。这是连接数学、代码与临床眼视光学的核心桥梁，也是现代光学设计师的必备技能。

---
### 引用的著作

[^1]: [A Data-Driven Application for Personalized IOL Selection - CRSToday](https://crstoday.com/articles/nov-dec-2024/a-data-driven-application-for-personalized-iol-selection)
[^2]: [Generalized Coddington equations in ophthalmic lens design - Optica Publishing Group](https://opg.optica.org/abstract.cfm?uri=josaa-13-8-1637)
[^3]: [Improved Analytical Theory of Ophthalmic Lens Design - MDPI](https://www.mdpi.com/2076-3417/11/12/5696)
[^4]: [Analysis of a Variational Approach to Progressive Lens Design - SIAM.org](https://epubs.siam.org/doi/10.1137/S0036139902408941)
[^5]: [Inverse design of nanophotonic structures using complementary convex optimization - Optica Publishing Group](https://opg.optica.org/abstract.cfm?uri=oe-18-4-3793)
[^6]: [Fixed-point iteration - Wikipedia](https://en.wikipedia.org/wiki/Fixed-point_iteration)
[^7]: [Brouwer fixed-point theorem - Wikipedia](https://en.wikipedia.org/wiki/Brouwer_fixed-point_theorem)
[^8]: [SciPy - Optimize - Tutorials Point](https://www.tutorialspoint.com/scipy/scipy_optimize.htm)
[^9]: [Efficiency of a New Iterative Algorithm Using Fixed-Point Approach in the Settings of Uniformly Convex Banach Spaces - MDPI](https://www.mdpi.com/2075-1680/13/8/502)
[^10]: 《实用泛函分析：Python驱动的眼视光学镜片设计优化》课程大纲
[^11]: [Fixed Point Theorems in Topology and Geometry A Senior Thesis Submitted to the Department of Mathematics In Partial Fulfillment - Millersville University](https://sites.millersville.edu/rumble/StudentProjects/Schreffler/schreffler.pdf)
[^12]: [What is the geometrical meaning of the existence of a fixed point for the complex case - Math Stack Exchange](https://math.stackexchange.com/questions/3308498/what-is-the-geometrical-meaning-of-the-existence-of-a-fixed-point-for-the-comple)
[^13]: [Cobweb plots - Applied Mathematics Consulting](https://www.johndcook.com/blog/2020/01/19/cobweb-plots/)
[^14]: [5.3: 5.3 Cobweb Plots for One-Dimensional Iterative Maps - Mathematics LibreTexts](https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Introduction_to_the_Modeling_and_Analysis_of_Complex_Systems_(Sayama)/05%3A_DiscreteTime_Models_II__Analysis/5.03%3A_5.3_Cobweb_Plots_for_One-Dimensional_Iterative_Maps)
[^15]: [Math Fun! Cobweb Plots Explained! (Draw by Hand After Graphing Calculator Function Iteration) - YouTube](https://www.youtube.com/watch?v=hdJjrCQ0lMg)
[^16]: [Contraction maps and applications to the analysis of iterative algorithms - DSpace@MIT](https://dspace.mit.edu/handle/1721.1/108973)
[^17]: [Relaxation Method - Adam Djellouli](https://adamdjellouli.com/articles/numerical_methods/1_root_and_extrema_finding/relaxation_method)
[^18]: [Example Gallery - Manim Community v0.19.0](https://docs.manim.community/en/stable/examples.html)
[^19]: [Manim's building blocks](https://docs.manim.community/en/stable/tutorials/building_blocks.html)
[^20]: [Welcome to Optiland's documentation! — Optiland 0.5.7 documentation](https://optiland.readthedocs.io/)
[^21]: [scipy.optimize.fixed_point — SciPy v1.16.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fixed_point.html)
[^22]: [Fixed point iteration algorithms - Cardinal](https://cardinal.cels.anl.gov/syntax/Executioner/FixedPointAlgorithms/)
[^23]: [Factors affecting convergence in the design of diffractive optics by iterative vector-space methods - Optica Publishing Group](https://opg.optica.org/abstract.cfm?uri=josaa-16-1-149)
[^24]: [Relaxation (iterative method) - Wikipedia](https://en.wikipedia.org/wiki/Relaxation_(iterative_method))
[^25]: [Data-Driven Ophthalmology - CRST Global](https://crstodayeurope.com/articles/nov-dec-24/data-driven-ophthalmology/)