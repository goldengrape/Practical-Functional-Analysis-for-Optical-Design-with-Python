# 第13章 前沿：AI、函数空间与未来光学设计

## 13.1 引言：光学设计的第三次浪潮

课程回顾： 在本课程的整个旅程中，我们致力于将光学设计从传统的、基于离散参数的搜索，重塑为在无穷维函数空间中寻找一个最优函数的连续优化问题。我们掌握了泛函分析的经典工具，特别是变分法（第2-3周）和算子理论（第5-7周），以驾驭这些复杂的空间[^1]。

定义第三次浪潮：
*   第一次浪潮： 经典设计。依赖于光学工程师的直觉、手动计算和简单的光线追迹模型。
*   第二次浪潮： 计算优化。这是本课程（第1-3模块）的核心重点，涉及基于梯度的算法、变分原理和有限元方法，以数值方式求解设计问题[^1]。
*   第三次浪潮： AI驱动的设计。这代表了一次根本性的范式转变。在第二次浪潮中，我们使用一个明确定义的优化算法来寻找一个解；而在第三次浪潮中，我们训练一个通用函数逼近器——即神经网络——来学习那个从设计空间映射到性能空间（Design Space $\rightarrow$ Performance Space）的函数本身。

本章目标： 本章将论证，人工智能（AI）并非泛函分析的替代品，而是其迄今为止最强大的应用。AI是解决我们在前几章中定义的无穷维问题的计算框架。我们将一同探索定义光学领域未来的新数学（PINN与Sobolev空间）、新挑战（AR/VR）和新工作流（AI代理模型）。

## 13.2 数学收敛：Sobolev空间中的神经网络

本课程大纲的核心概念之一是“神经网络近似Sobolev规范”[^1]。本节将深入解构这一关键联系。

### 13.2.1 神经网络作为通用函数逼近器

正如本课程Manim教学动画的核心启发者3Blue1Brown所直观展示的那样，一个神经网络本质上只是一个极其复杂的复合函数[^2]。它是一个计算机器，理论上，只要有足够多的参数（权重和偏置），它就能够逼近任何其他连续函数[^3]。

在本课程中，我们已经深入讨论了“函数空间”（如第4周的Hilbert空间和第8周的Banach空间）[^1]。从这个角度来看，神经网络为我们提供了一个强大的、可计算的工具，用以在这些抽象的数学空间中，实际地构造和表示一个向量（即一个函数）[^4]。

### 13.2.2 $L^2$范数的问题：为何标准机器学习在物理学中会失败

在标准的机器学习任务中，最常见的损失函数是均方误差（MSE），它在数学上是对 $L^2$ 范数的一种逼近。

问题在于，$L^2$ 范数只测量函数值的误差（即预测值与真值之间的距离）。它对于函数导数的误差是完全盲视的。

这一局限性在物理和工程领域是致命的。其内在的因果链如下：
1.  所有物理学，包括光学设计，都是由偏微分方程（PDEs）支配的。例如，光线路径由Eikonal方程决定，而复杂自由曲面设计则可能涉及Monge-Ampère方程[^5]。
2.  PDE 本质上就是对一个函数导数（如梯度、曲率）的根本约束。
3.  因此，一个仅仅被训练来最小化 $L^2$ 损失的网络，可能会产生一个在数据点上看起来“很接近”的解，但这个解在物理上却是不可能的，因为它在数据点之间的导数（例如，曲面的平滑度）是无意义的。它可能具有很低的数据误差，但却具有极高的“物理误差”。

### 13.2.3 解决方案：物理信息神经网络 (PINN)

物理信息神经网络（Physics-Informed Neural Networks, PINNs）是一类新型的神经网络架构，它将控制物理定律（即PDEs）直接编码到损失函数中[^7]。

PINN的总损失函数 $L_{total}$ 通常由两个（或更多）部分组成[^9]：
$L_{total} = L_{data} + L_{physics}$

1.  $L_{data}$（数据损失）： 这是传统的 $L^2$ 损失。它确保网络的解在已知点（如边界条件、测量点）上与真实数据拟合。
2.  $L_{physics}$（物理损失）： 这是“残差损失”。我们将神经网络的输出（及其通过自动微分计算出的导数）代入PDE方程，如果结果不为零（即存在残差），网络就会受到惩罚[^9]。

通过最小化这个混合损失函数，网络被迫使去寻找一个既拟合数据又遵守物理定律的解。这种方法已被证明在解决纳米光学、超材料等领域的复杂逆问题上非常成功[^10]。

### 13.2.4 “正确的”度量：作为终极损失函数的Sobolev范数

现在，我们将PINN与本课程的核心主题——泛函分析——进行显式连接。

我们首先回忆一个关键定义：什么是Sobolev范数？如[^13]中所述，Sobolev范数 $W^{s,p}$ 是一种“更强”的范数，它不仅量化函数本身的 $L^p$ 范数（其值），还同时量化其直到 $s$ 阶（弱）导数的 $L^p$ 范数。

在这里，我们得出一个深刻的结论：
1.  PINN的损失函数 ($L_{total} = L_{data} + L_{physics}$) 是对一个Sobolev范数的计算近似。
2.  $L_{data}$ 负责最小化函数本身的 $L^p$ 范数（通常是 $L^2$）。
3.  $L_{physics}$ 负责最小化函数导数的 $L^p$ 范数（由PDE定义）。
4.  因此，"训练一个PINN" 的数学本质，等同于 "在函数空间中寻找一个解，使其与真解之间的Sobolev范数最小化"。

这就是PINN有效的深层数学原理。通过采用“Sobolev训练”（无论是显式使用Sobolev范数作为损失函数，还是通过PINN隐式实现），我们迫使网络学习到一个平滑的、物理上合理的解，从而极大地改善了收敛性、降低了对大量训练数据的依赖，并提高了泛化能力[^14]。这正是本课程大纲[^1]中连接AI与Sobolev规范的承诺的直接实现。

### 13.2.5 Manim 可视化：“看见”函数空间中的逼近

为使“在Sobolev空间中收敛”这一抽象概念变得直观，本章的配套Manim动画（见代码库 `manim_demos/ai_function_space.py`）演示了以下过程：

*   场景1：$L^2$ 范数的失败。 动画展示了一系列离散的采样点（代表一个期望的镜片轮廓）。一个仅在 $L^2$ 损失下训练的神经网络试图拟合这些点。我们将看到网络曲线精确地“击中”了每个点，但在点与点之间却发生了剧烈的“扭动”（即高阶导数失控）。
*   场景2：Sobolev 范数的成功。 相同的动画再次运行，但这次使用的是Sobolev（或PINN）损失函数。我们将看到网络曲线平滑地弯曲到位，它不仅逼近了数据点，同时还尊重了整体的曲率和平滑度约束。

这种可视化教学方法，是本课程利用Manim将抽象数学转化为工程直觉的核心教学理念[^1]。

## 13.3 前沿设计挑战：AR/VR光学

如果说PINN和Sobolev空间是新兴的数学工具，那么增强现实（AR）和虚拟现实（VR）光学系统就是这些工具最迫切的应用场景[^1]。

### 13.3.1 下一代光学的“不可能三角”

光学技术被公认为是AR/VR设备能否被主流市场接受的关键瓶颈[^16]。市场对下一代设备同时提出了三个在物理上相互冲突的核心要求：

1.  小尺寸（Form Factor）： 设备必须像普通眼镜一样轻薄[^16]。
2.  大视场角（FoV）： 必须提供沉浸式的宽广视野[^18]。
3.  大出瞳量（Eyebox）与高光效： 即使用户的眼睛（或设备）轻微移动，图像依然清晰可见，并且图像必须足够明亮以适应户外使用。

在光学设计中，这三者构成了一个“不可能三角”。当前的架构选择，本质上是在权衡你愿意牺牲哪一个。我们将分析两种最主流的架构：用于VR的折反射式（Pancake）和用于AR的衍射光波导（Waveguide）。

### 13.3.2 架构一：折反射式“Pancake”透镜 (VR)

“Pancake”（薄饼）透镜是一种折反射式（Catadioptric）系统，它同时利用了光的折射和反射。这是当前高端VR头显（如Meta Quest Pro, Apple Vision Pro）的主流选择[^19]。

*   工作原理（折叠光路）：
    Pancake方案的核心是利用偏振光学元件来折叠光路，从而极大地缩短显示屏和人眼之间的物理距离[^21]。一个典型的光路（综合[^24]）如下：
    1.  显示屏发出的非偏振光首先通过一个线性偏振片（LP）。（第一次光损失：$\approx$ 50%）
    2.  线偏振光入射到偏振分束器（PBS）上，被反射。
    3.  光线穿过一个1/4波片（QWP），变为圆偏振光。
    4.  光线在透镜组的外部镜片（镀反射膜）上反射。
    5.  光线再次穿过QWP，偏振态被翻转90度（例如，从X向线偏振变为Y向线偏振）。
    6.  由于偏振态被翻转，光线现在将穿过PBS（而不是被反射），最终进入人眼。

*   关键挑战1：灾难性的光效率
    如[^26]和[^26]的分析，Pancake架构的理论光效率上限，对于偏振光输入仅为25%，而对于最常见的非偏振光输入，其上限仅为 12.5%。

    这一规格（12.5%的光效率）对整个头显设计产生了一系列毁灭性的连锁反应：
    *   它迫使工程师必须使用一块极其明亮的显示屏（例如，亮度是传统显示屏的8-10倍）。
    *   超亮的显示屏会 (a) 产生巨量热量；(b) 极大地消耗电池寿命[^17]；(c) 需要一块更大的电池来维持运行。
    *   更大的电池和散热系统显著增加了头显的总重量，这又反过来违背了采用Pancake方案以追求“轻薄”的初衷。

*   关键挑战2：杂散光与鬼影
    该系统依赖于完美的偏振态控制。任何偏振“泄漏”（例如，QWP不完美、光线非正轴入射）都会导致光线走错路径（例如，本应透射的光被反射），从而在用户视野中产生明显的“鬼影”和低对比度[^26]。

### 13.3.3 架构二：衍射光波导 (AR)

衍射光波导是目前“透视型”AR眼镜（如Microsoft HoloLens）的主流方案[^16]。

*   工作原理：
    其工作原理[^16]如下：
    1.  一个微型投影仪（光引擎）将图像光线射入一块薄玻璃（或树脂）基板的边缘。
    2.  一个“入耦合”光栅（一种衍射光学元件，DOE）以特定角度衍射光线，使其被“困”在基板内部，并通过全内反射（TIR）向前传播。
    3.  光线在基板内部“反弹”前进。
    4.  在人眼区域，光线每次反弹时都会遇到“出耦合”光栅（另一种DOE），该光栅会将一小部分受控的光“泄漏”出基板，射向人眼。这一过程也极大地扩展了“出瞳量”（Eyebox）。

*   关键挑战1：仿真与设计的复杂性
    这是一个严重的多物理场、多尺度问题[^16]。
    *   纳米尺度： 耦合光栅本身是纳米结构（例如，周期几百纳米），其衍射效率高度依赖于波长和角度。它们受物理光学（麦克斯韦方程组）支配，必须使用RCWA（严格耦合波分析）等工具进行仿真[^29]。
    *   系统尺度： 光线在厘米级基板中的传播则遵循几何光学（光线追迹）原理。
    *   工作流瓶颈：[^16]设计师必须采用混合工作流：(1) 在物理光学工具（如RSoft, Lumerical）中仿真光栅，生成其BSDF（双向散射分布函数）数据。(2) 将这个（通常很大的）BSDF数据文件导入到几何光学工具（如Zemax, Speos, LightTools）中，作为一个表面属性[^30]。(3) 运行完整的系统仿真。这个过程极其缓慢、繁琐，是优化迭代的最大瓶颈。

*   关键挑战2：色散。 光栅对不同颜色的光有不同的衍射角度，导致严重的色散（彩虹条纹），需要复杂的RGB三通道系统来补偿[^30]。

### 13.3.4 AI驱动的解决方案

无论是Pancake还是光波导，它们产生的像差（特别是离轴像差，如像散）都极其复杂。传统球面/非球面透镜无法校正。这推动了两个领域的交叉：

1.  自由曲面光学：[^31]自由曲面（Freeform）是非对称的、无旋转对称性的光学表面，它们是校正AR/VR中复杂像差的必要组件[^33]。
2.  AI模型： 解决这些问题的唯一实用方法是使用AI。一方面，AI可以作为代理模型（Surrogate）打破仿真瓶颈（见13.4）；另一方面，AI（特别是PINN）甚至可以用来直接求解设计自由曲面所需的Monge-Ampère方程[^6]。

**表 13.1：AR/VR 关键光学架构对比**

| 架构类型 | 主要应用 | 尺寸形态 | 光学效率 | 视场角 (FoV) / 出瞳量 (Eyebox) | 关键设计挑战 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 传统折射 | 低端VR | 笨重（长焦距） | 高 ( > 90% ) | 受限 | 重量、体积、“潜水镜”效应 |
| 折反射 (Pancake) | 高端VR | 紧凑（折叠光路） | 极低 ( $\approx$ 12.5% )[^26] | 宽FoV，小Eyebox | 光效低、杂散光/鬼影[^27]、电池与散热压力 |
| 衍射光波导 | AR（透视） | 极薄（镜片式） | 低 ( < 10% ) | 小FoV，大Eyebox | 仿真设计复杂性（多尺度）[^16]、色散[^30] |

## 13.4 未来工作流：从代理到副驾驶

面对13.3中提出的巨大挑战，传统的设计-仿真-迭代循环已经不堪重负。AI正通过重构整个光学设计工作流来提供解决方案[^1]。

### 13.4.1 阶段一：AI作为加速器（代理模型）

*   问题： 我们的设计循环受限于昂贵的仿真（无论是光波导的混合仿真[^30]还是超材料的FDTD仿真[^12]）。一次高精度仿真可能需要数小时甚至数天。
*   解决方案： “代理模型”（Surrogate Model）。我们不再依赖仿真器进行每次迭代，而是训练一个机器学习模型来预测仿真的输出（例如，RMS光斑尺寸、MTF）。这个模型的输入是设计参数（例如，曲率、厚度），输出是预测的性能[^34]。
*   实例： LensAI GitHub代码库[^37]提供了完美的例证。它展示了如何构建：
    *   一个 `scikit-learn` 中的 `RandomForestRegressor` （随机森林回归器）来预测最优的透镜半径[^37]。
    *   一个神经网络代理模型，用于替代传统光线追迹，实现了高达 10,000倍 的有效速度提升[^37]。
*   高级技术： 我们可以更进一步，使用“深度主动学习”（Deep Active Learning）[^34]。在这种模式下，AI模型会智能地选择它最不确定的设计点，主动请求对这些点运行昂贵的仿真，从而以最少的仿真次数达到最高的模型精度。

### 13.4.2 阶段二：AI作为合作者（生成式设计）

*   范式转变： 代理模型（阶段一）的核心是加速对已有参数的优化。生成式设计（阶段二）的核心是发现全新的、未知的拓扑结构。
*   定义： 这就是“逆向设计”（Inverse Design）[^38]。工程师不再提供设计参数，而是只指定性能需求（输出）。一个生成式网络（如VAE或GAN）会反向工作，生成一个全新的、通常是反直觉的光学结构（输入）来满足这些需求[^40]。
*   应用： 这项技术正被用于设计具有特定光谱响应的超材料[^10]、复杂的光子结构[^34]甚至优化的仪器支架[^41]。如果说代理模型是帮助我们优化一个已知的“Cooke三片式”镜头，那么生成式设计则是在我们只给出像差目标的情况下，从零开始发明“Cooke三片式”镜头[^43]。

### 13.4.3 路线图：AI辅助的工作流

未来最高效的工作流，既不是纯粹的人类设计，也不是完全的“黑盒”AI设计，而是一个“AI副驾驶”（AI Co-pilot）模型[^44]。我们可以预见一个渐进式的集成路线图[^45]：

*   阶段1：AI辅助规格定义。 （AI帮助工程师将临床需求转化为数学约束）
*   阶段2：AI代理模型用于快速迭代。 （本章的最终项目）
*   阶段3：与CAD/CAE集成。 （AI直接在Zemax或CODE V中建议参数修改）
*   阶段4：完整的生成式设计流程。 （AI为优化器生成全新的、高质量的初始设计点）

**表 13.2：光学设计中的AI范式演进**

| 范式 | 目标 | 核心方法 | 典型应用场景 |
| :--- | :--- | :--- | :--- |
| 传统优化 | 在已知参数空间中寻找最优解 | 梯度下降、L-M算法 | 优化现有镜头的曲率、厚度 |
| 代理模型 (阶段一) | 加速昂贵的性能仿真 | 神经网络回归、随机森林[^37] | 实时公差分析、快速迭代自由曲面 |
| 生成式设计 (阶段二) | 发现满足性能需求的全新拓扑 | GANs, VAEs, PINNs[^40] | 逆向设计超材料、发明新颖的光学构型 |

## 13.5 最终项目：AI辅助设计原型（Python实战）

作为本课程的顶点项目，我们将构建一个“阶段一”的AI辅助设计原型[^1]。这个项目将综合运用我们在课程中学到的Python工具：使用 `scikit-learn` 构建AI代理模型，使用 `Streamlit` 构建交互式Web UI，并（概念性地）集成 `Manim` 动画来解释设计原理。

### 13.5.1 项目概览：构建“阶段一”代理模型Web应用

*   目标： 创建一个Web应用。用户在侧边栏通过滑块调整镜片参数（如曲率、厚度）。应用主界面会立即显示AI代理模型预测的光学性能（如RMS光斑尺寸）。
*   工具链： `scikit-learn` 用于训练模型[^37]，`Streamlit` 用于构建前端App[^47]。

### 13.5.2 第1部分：训练代理模型 (scikit-learn)

在这一步，我们模拟一个昂贵的仿真过程。我们假设已经运行了1000次Zemax仿真，并将结果保存到 `lens_data.csv` 中。

文件名: `train_surrogate.py`:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. 数据收集 (假设已存在)
# 我们假设 lens_data.csv 包含:
# 'curvature_1', 'thickness', 'curvature_2', 'rms_spot_size'
try:
   data = pd.read_csv('lens_data.csv')
except FileNotFoundError:
   print("错误: 'lens_data.csv' 未找到。请先生成仿真数据。")
   exit()

# 2. 数据准备
X = data[['curvature_1', 'thickness', 'curvature_2']]
y = data['rms_spot_size']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型训练 (使用随机森林, 它对于此类表格数据非常鲁棒) 
# 我们选择一个 n_estimators=100 的随机森林回归器
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("正在训练代理模型...")
model.fit(X_train, y_train)

# (可选) 评估模型性能
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"代理模型训练完毕。RMSE: {rmse:.4f} (微米)")

# 4. 序列化 (保存) 模型，以便Streamlit应用调用
joblib.dump(model, 'lens_surrogate.joblib')
print("模型已保存为 'lens_surrogate.joblib'")
```

### 13.5.3 第2部分：构建交互式UI (Streamlit)

现在我们创建 `app.py`。这是我们的Web应用本身。在终端中运行 `streamlit run app.py` 即可启动它。

文件名: `app.py`:
```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. 标题和设置
st.set_page_config(layout="wide")
st.title('🤖 AI 辅助光学设计原型')
st.markdown("这是一个演示AI代理模型（阶段一）如何加速设计迭代的工具。")

# 2. 加载训练好的代理模型
try:
   model = joblib.load('lens_surrogate.joblib')
except FileNotFoundError:
   st.error("错误：未找到 'lens_surrogate.joblib' 模型文件。请先运行 `train_surrogate.py`。")
   st.stop()

# 3. 创建 UI 控件 (放置在侧边栏)
st.sidebar.header('镜片设计参数 (Inputs)')
c1 = st.sidebar.slider('曲率 1 (1/R)', -0.05, 0.05, 0.02, step=0.001, format="%.3f")
t = st.sidebar.slider('厚度 (mm)', 1.0, 10.0, 4.5, step=0.1, format="%.1f")
c2 = st.sidebar.slider('曲率 2 (1/R)', -0.05, 0.05, -0.01, step=0.001, format="%.3f")

# 4. 实时预测与显示结果
st.header('AI 代理模型性能预测 (Outputs)')

# 准备模型输入：Streamlit的滑块值 -> DataFrame
# scikit-learn的 predict 方法期望一个2D数组或DataFrame
input_data = pd.DataFrame(
   [[c1, t, c2]],
   columns=['curvature_1', 'thickness', 'curvature_2']
)

# 调用模型进行预测
try:
   prediction = model.predict(input_data)[0]
   
   # 使用 st.metric 显示预测结果，提供直观的性能反馈
   st.metric(label="预测的 RMS 光斑尺寸 (微米)", value=f"{prediction:.3f}")
   st.caption("AI模型正在实时预测该参数组合的光学性能，无需运行昂贵的仿真。")

except Exception as e:
   st.error(f"模型预测时发生错误: {e}")

# 可视化输入参数
st.subheader("当前输入参数")
st.write(input_data)
```

### 13.5.4 第3部分：集成Manim可视化

在真实的工程应用中，在Streamlit中实时渲染Manim动画计算量巨大且不切实际。更稳健、更常见的做法是嵌入预先渲染好的Manim视频，用以解释工作原理。这符合Streamlit作为数据科学和洞察分享工具的定位[^51]。

我们将添加一个 `st.video` 标签，播放一个（假设已渲染好的）Manim动画，该动画演示了AR/VR中的复杂光路，例如Pancake的折叠光路或光波导中的TIR[^30]。

继续在 `app.py` 中添加:
```python
# 5. 集成Manim原理演示 
st.header('Manim 原理可视化')
st.markdown("以下动画（预渲染）展示了本设计试图优化的复杂光学系统（例如AR/VR光路）的原理。")

# 假设我们已有名为 'ar_vr_raytrace.mp4' 的动画文件
try:
   video_file = open('ar_vr_raytrace.mp4', 'rb')
   video_bytes = video_file.read()
   st.video(video_bytes)
   st.caption("Manim动画演示：AR/VR 系统中的折叠光路或波导光线追迹。")
except FileNotFoundError:
   st.warning("未找到 'ar_vr_raytrace.mp4' 演示视频。")
```

这个项目完美地总结了本课程的理念：从一个具体的工程问题（AR/VR设计瓶颈）出发，利用泛函分析的思维（PINN/Sobolev）建立数学框架，并最终通过Python工具链（scikit-learn, Streamlit, Manim）实现了一个可交互的工程原型。

## 13.6 本章小结与课程总结

本章带领我们跨越了从经典理论到AI驱动设计前沿的桥梁。我们建立了三个核心支柱：

1.  新的数学： 我们揭示了PINN与Sobolev训练的深层数学联系。这表明，AI不再只是数据科学的模式识别工具，而是正在成为一个强大的、能够求解物理PDE的泛函分析工具。
2.  新的问题： 我们分析了AR/VR光学的“不可能三角”。Pancake透镜灾难性的光效率[^26]和光波导极度的仿真复杂性[^16]，为AI驱动的设计工具提供了最迫切的商业和工程需求。
3.  新的工作流： 我们演示了AI在工程实践中的落地路径。从“阶段一”的代理模型（如我们的 `scikit-learn` 项目）开始，逐步过渡到“阶段二”的生成式AI副驾驶。

本课程始于一个核心观点：将光学设计视为无穷维函数空间中的优化问题。我们现在已经看到，神经网络是我们迄今为止拥有的、用以逼近、搜索和掌握这个函数空间的最强工具。未来的光学设计师，不仅是物理学家和工程师，更将是AI系统的架构师，他们将能够以史无前例的速度和创造力，在这个广阔的设计空间中航行。

[^1]: 《实用泛函分析：Python驱动的眼视光学镜片设计优化》课程大纲
[^2]: [But what is a Neural Network? - 3Blue1Brown](https://www.3blue1brown.com/lessons/neural-networks)
[^3]: [Gradient descent, how neural networks learn - 3Blue1Brown](https://www.3blue1brown.com/lessons/gradient-descent)
[^4]: [Development of a machine learning-based automated system for the detection of rub in gas turbines - Webthesis - Politecnico di Torino](https://webthesis.biblio.polito.it/15765/1/tesi.pdf)
[^5]: [Neural Network Method for Solving Partial Differential Equations](https://www.researchgate.net/publication/220577840_Neural_Network_Method_for_Solving_Partial_Differential_Equations)
[^6]: [A neural network approach for solving the Monge-Amp\`ere equation with transport boundary condition - ResearchGate](https://www.researchgate.net/publication/385292041_A_neural_network_approach_for_solving_the_Monge-Ampere_equation_with_transport_boundary_condition)
[^7]: [Daily Papers - Hugging Face](https://huggingface.co/papers?q=inverse%20Fourier%20transform)
[^8]: [Data-Driven Elasticity Imaging Using Cartesian Neural Network Constitutive Models and the Autoprogressive Method | Request PDF - ResearchGate](https://www.researchgate.net/publication/328755439_Data-Driven_Elasticity_Imaging_Using_Cartesian_Neural_Network_Constitutive_Models_and_the_Autoprogressive_Method)
[^9]: [Physics-Informed Neural Networks for the Structural Analysis and Monitoring of Railway Bridges: A Systematic Review - MDPI](https://www.mdpi.com/2227-7390/13/10/1571)
[^10]: [[1912.01085] Physics-informed neural networks for inverse problems in nano-optics and metamaterials - arXiv](https://arxiv.org/abs/1912.01085)
[^11]: [[2109.12754] Physics-informed neural networks for imaging and parameter retrieval of photonic nanostructures from near-field data - arXiv](https://arxiv.org/abs/2109.12754)
[^12]: [Large-scale photonic inverse design: computational challenges and breakthroughs - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11465988/)
[^13]: [Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning - arXiv](https://arxiv.org/pdf/2402.10926)
[^14]: [A Fast Spectral Method for Active 3D Shape Reconstruction](https://www.researchgate.net/publication/277389125_A_Fast_Spectral_Method_for_Active_3D_Shape_Reconstruction)
[^15]: [Sobolev neural network with residual weighting as a surrogate in linear and non-linear mechanics - ResearchGate](https://www.researchgate.net/publication/382492091_Sobolev_neural_network_with_residual_weighting_as_a_surrogate_in_linear_and_non-linear_mechanics)
[^16]: [Designing Augmented/Virtual Reality Devices using ... - Synopsys](https://www.synopsys.com/content/dam/synopsys/photonic-solutions/documents/pdf-demos/october-2020-arvr-presentation-slides.pdf)
[^17]: [AR and VR Devices Lens Market Dynamics: Drivers and Barriers to Growth 2025-2033](https://www.datainsightsmarket.com/reports/ar-and-vr-devices-lens-1653109)
[^18]: [Meta-Optics for Optical Engineering of Next-Generation AR/VR Near-Eye Displays - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12471599/)
[^19]: [The Optical System Design of AR/VR Headsets - The Past, Present, and Future](https://opticaorg-dev-cac7d2csctagc8bm.z01.azurefd.net/$web/optica/media/files/getinvolved/tech-groups/webinars/2024/pdfs/the_optical_system_design_of_arvr_headsets_-_the_past_present_and_future.pdf)
[^20]: [Double‐path pancake optics. (A) Operation mechanism of double‐path... | Download Scientific Diagram - ResearchGate](https://www.researchgate.net/figure/Double-path-pancake-optics-A-Operation-mechanism-of-double-path-pancake-optics-system_fig2_380310597)
[^21]: [System configuration of our proposed folded optics. - ResearchGate](https://www.researchgate.net/figure/System-configuration-of-our-proposed-folded-optics_fig3_370097990)
[^22]: [US20180101020A1 - Head mounted display including pancake lens block - Google Patents](https://patents.google.com/patent/US20180101020A1/en)
[^23]: [Apple Vision Pro Optics insights - HyperVision](https://www.hypervision.ai/tech-research/apple-vp-optics-insights)
[^24]: [(a) Simplified schematic of folded optics (pancake eyepiece) VR... - ResearchGate](https://www.researchgate.net/figure/a-Simplified-schematic-of-folded-optics-pancake-eyepiece-VR-displays-The-ray-path-of_fig27_363592166)
[^25]: [Thin and lightweight camera based on Pancake lens and deep learning](https://opg.optica.org/ol/abstract.cfm?uri=ol-49-17-4851)
[^26]: [(PDF) Augmented reality and virtual reality displays: emerging ...](https://www.researchgate.net/publication/355587665_Augmented_reality_and_virtual_reality_displays_emerging_technologies_and_future_perspectives)
[^27]: [Display and Optics Architecture For Metas AR VR Development ...](https://www.scribd.com/document/736557306/Display-and-Optics-Architecture-for-Metas-AR-VR-Development)
[^28]: [Understanding Waveguide: the Key Technology for Augmented Reality Near-eye Display (Part II) | by Rokid](https://arvrjourney.com/understanding-waveguide-the-key-technology-for-augmented-reality-near-eye-display-part-ii-fe4bf3490fa)
[^29]: [A comprehensive approach to diffractive waveguide optimization in mixed reality near-eye displays - SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13129/131290O/A-comprehensive-approach-to-diffractive-waveguide-optimization-in-mixed-reality/10.1117/12.3028508.full)
[^30]: [RGB Augmented Reality Optical System – Ansys Optics](https://optics.ansys.com/hc/en-us/articles/33794233218579-RGB-Augmented-Reality-Optical-System)
[^31]: [Augmented Reality and Virtual Reality Lens Navigating Dynamics Comprehensive Analysis and Forecasts 2025-2033](https://www.archivemarketresearch.com/reports/augmented-reality-and-virtual-reality-lens-52984)
[^32]: [Advancing freeform gradient index (GRIN) optics for vision correction - SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13466/134660F/Advancing-freeform-gradient-index-GRIN-optics-for-vision-correction/10.1117/12.3057169.full)
[^33]: [Light field displays with computational vision correction for astigmatism and high-order aberrations with real-time implementation - Optica Publishing Group](https://opg.optica.org/abstract.cfm?uri=oe-31-4-6262)
[^34]: [Data-driven design of thin-film optical systems using deep active ...](https://opg.optica.org/abstract.cfm?uri=oe-30-13-22901)
[^35]: [(PDF) Deep neural network (DNN) surrogate models for the accelerated design of optical devices and systems - ResearchGate](https://www.researchgate.net/publication/335708969_Deep_neural_network_DNN_surrogate_models_for_the_accelerated_design_of_optical_devices_and_systems)
[^36]: [AI-Based Metamaterial Design - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181287/)
[^37]: [HarrisonKramer/LensAI: Integrated Machine and Deep ... - GitHub](https://github.com/HarrisonKramer/LensAI)
[^38]: [[2505.03354] Physics-Informed Neural Networks in Electromagnetic and Nanophotonic Design - arXiv](https://arxiv.org/abs/2505.03354)
[^39]: [Physics-informed neural networks with hard constraints for inverse design - arXiv](https://arxiv.org/abs/2102.04626)
[^40]: [Deep neural networks for the evaluation and design of photonic devices - UMBC](https://dil.umbc.edu/wp-content/uploads/sites/629/2022/09/REV-Jiang_Nat_Revews_2020_DL_for_Photonics.pdf)
[^41]: [Generative design and digital manufacturing: using AI and robots to build lightweight instrument structures - SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12217/122170O/Generative-design-and-digital-manufacturing--using-AI-and-robots/10.1117/12.2646476.short)
[^42]: [Making Physical Objects with Generative AI and Robotic Assembly: Considering Fabrication Constraints, Sustainability, Time, Functionality and Accessibility - arXiv](https://arxiv.org/html/2504.19131v2)
[^43]: [Deep Learning, AI, and Generative Design - ATM, Optics and DIY Forum - Cloudy Nights](https://www.cloudynights.com/topic/854199-deep-learning-ai-and-generative-design/)
[^44]: [Workflow for generative AI-assisted CAD modeling (GAD ...](https://www.researchgate.net/figure/Workflow-for-generative-AI-assisted-CAD-modeling-GAD_fig1_392440353)
[^45]: [Using Conversational AI for Design Inspiration with ChatGPT and Autodesk Fusion](https://static.au-uw2-prd.autodesk.com/CP2101_Handout-2101-Conversational_AI_for_Design_Inspiration_1756658892858001x6fc.pdf)
[^46]: [(PDF) Nanophotonic structure inverse design for switching application using deep learning](https://www.researchgate.net/publication/383917897_Nanophotonic_structure_inverse_design_for_switching_application_using_deep_learning)
[^47]: [Top 25 Data Science Projects Ideas for Beginners to Advanced](https://brollyacademy.com/data-science-projects-ideas/)
[^48]: [How to Quickly Deploy Machine Learning Models with Streamlit - MachineLearningMastery.com](https://machinelearningmastery.com/how-to-quickly-deploy-machine-learning-models-streamlit/)
[^49]: [Affiliated Projects | AiiDA, bqplot, Conda, + more - NumFOCUS](https://numfocus.org/sponsored-projects/affiliated-projects)
[^50]: [EuCAP2025 - Comparing Differentiable and Dynamic Ray Tracing: Introducing the Multipath Lifetime Map | Jérome Eertmans](https://eertmans.be/posts/eucap2025/)
[^51]: <https://www.streamlit.io/>