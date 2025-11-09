# 第12章：不确定性量化实战：从泛函分析到临床风险

## 12.1 引言：为什么不确定性是新的优化目标

在前面的章节中，我们的重点是为理想化的患者模型构建性能最优的镜片设计。然而，在实际的眼视光学工程中，一个在模拟中“完美”的设计，当应用于真实、多变且数据不完整的患者时，往往会表现不佳。

本课程的核心理念是，现代个性化镜片设计的真正目标已从追求单一的“最佳”设计转变为交付在真实世界不确定性下表现稳健的设计。不确定性不再是需要避免的麻烦，而是设计过程中必须主动量化和管理的核心参数。

在个性化眼科设计中，不确定性主要来源于两个根本不同的方面，它们需要两套不同的数学和计算工具来应对。

1.  认知不确定性 (Epistemic Uncertainty) - 源于系统/模型： 这类不确定性源于我们自身知识的缺乏。在镜片设计中，这主要表现为制造公差（我们无法完美制造一个曲面）[^2]和模型简化（我们的数学模型可能未捕捉到所有物理效应）。这类不确定性可以通过敏感性分析来评估和降低。
2.  偶然不确定性 (Aleatoric Uncertainty) - 源于患者数据： 这类不确定性是系统固有的、不可简化的变异，源于患者的多样性、测量噪声和临床数据的不完整性[^1]。我们无法消除这种不确定性，因此必须使用概率方法（如贝叶斯统计）来对其进行建模。

本章将介绍一个“总体不确定性工作流” (Total Uncertainty Workflow)，它不是三个独立主题的集合，而是一个连贯的工程实践过程。我们将学习如何追踪不确定性，从它在工厂和患者数据中的源头，到它在光学模型中的传播，直至其最终的工程后果——临床风险的量化。

本章的逻辑流程如下：

1.  分析系统不确定性： 使用泛函敏感性分析 (FSA) 来识别设计的“脆弱点”。
2.  建模患者不确定性： 使用贝叶斯方法处理不完整和含噪声的临床数据。
3.  整合与决策： 将两者结合在一个正式的风险评估框架 (ISO 14971) 中，以制定可辩护的临床和工程决策。

## 12.2 主题一：泛函敏感性分析 (FSA)

### 12.2.1 回顾：参数敏感性分析 (PSA)

在进入泛函领域之前，我们先回顾一下标准工程实践：参数敏感性分析 (PSA)。PSA 研究模型的输出不确定性如何归因于其输入参数的不确定性[^4]。

在光学设计中，这通常表现为一个简单的偏导数，例如：

$$
S = \frac{\Delta(\text{MTF})}{\Delta(\text{曲率半径})}
$$

计算结果 $S$ 是一个标量，告诉我们调制传递函数 (MTF) 对镜片曲率半径变化的敏感程度。这很有用，但其局限性也显而易见：当我们的“参数”不是一个单一的标量，而是整个镜片表面的连续函数时，我们该怎么办？

### 12.2.2 飞跃：泛函敏感性分析 (FSA)

泛函敏感性分析 (FSA) 是一种更强大的技术，它测量的不是系统对标量参数的响应，而是对一个函数（例如镜片矢高轮廓）的任意微小变化的响应[^6]。

其核心数学工具是泛函导数。如果一个性能指标（标量）$J$（例如MTF）是镜片表面函数 $s(x, y)$ 的泛函，记作 $J[s(x, y)]$，那么其泛函导数为：

$$
\frac{\delta J}{\delta s(x, y)}
$$

这个表达式的结果不是一个数字，而是一个新的函数。这个函数定义在与 $s(x, y)$ 相同的域（即镜片光瞳）上，可以被视为一张“敏感性地图”[^6]。

### 12.2.3 解读：作为公差工具的FSA地图

FSA最有价值的工程产出是一张可视化的“工程焦点图”。

*   FSA地图的构建： 我们可以计算 $\frac{\delta J}{\delta s(x, y)}$ 并在镜片光瞳 $(x, y)$ 上将其绘制为热力图。
*   地图的含义：
    *   热力图上的高值区域（例如亮红色）是设计的“高风险区”。在这些区域，即使是微小的制造误差（公差）也会导致性能指标 $J$ 的急剧下降。
    *   热力图上的低值区域（例如蓝色）是“稳健区”。在这些区域，设计对制造误差不敏感，允许更宽松（因此更便宜）的公差。
*   工程应用： 这张地图完美地回答了第8周关于制造公差的问题。它不再是给整个镜片一个单一的公差值，而是允许设计师根据FSA地图进行“公差预算分配”。高敏感性区域[^6]必须使用更严格（更昂贵）的制造工艺来控制，而低敏感性区域则可以放宽要求。这种方法是实现“公差退化”[^2]受控设计的关键。

### 12.2.4 计算方法：伴随法 (Adjoint Methods)

直接计算FSA地图是困难的。采用蛮力法（在镜片表面数千个点上分别施加微扰，并为每个微扰运行一次完整的射线追踪）在计算上是不可行的。

解决方案是伴随法 (Adjoint Methods)[^9]。伴随法是一种强大的计算技巧，它允许我们在一次逆向计算中，同时获得性能指标 $J$ 相对于所有输入参数（或曲面上的所有点）的梯度。

这与现代“可微分光学” (Differentiable Optics)[^2]和深度学习中的反向传播在概念上是相同的。通过构建一个完全可微分的光学系统模型，我们可以利用GPU进行“一步梯度反向传播” (one-step gradient backpropagation)，高效地计算出完整的敏感性地图[^2]，从而实现对公差敏感性的快速优化。

## 12.3 主题二：用于患者数据不确定性的贝叶斯方法

处理完来自制造（系统）的不确定性后，我们转向第二个更棘手的来源：患者数据。临床数据本质上是“凌乱的”——它充满了测量噪声、数据缺失和个体差异。

### 12.3.1 问题：为什么 `mean()` 远不足够

面对不完整的患者数据（例如，验光单上缺少角膜Q值），最天真的做法是用群体的平均值（例如 $Q = -0.26$）来填充。

这种“均值插补”方法极其危险。它不仅忽略了该数值本身的不确定性（即我们对这个填充值的信心有多大），而且还人为地降低了数据的方差，扭曲了变量间的相关性，最终导致对设计性能的过度自信和错误评估。

### 12.3.2 贝叶斯哲学：将缺失数据视为参数

贝叶斯统计提供了一个优雅的解决方案。其核心思想是：在贝叶斯框架下，一个我们想要估计的参数（如 $\theta$）和一个缺失的数据点（如 $Q_{\text{missing}}$）之间没有本质区别[^11]。它们都是我们希望了解其概率分布的未知量。

我们可以利用贝叶斯定理，使用患者的观测数据（例如年龄、屈光度、K值）作为证据，来更新我们对未观测数据（缺失的Q值）的信念（即概率分布）[^3]。

$$
P(\text{Q}_{\text{missing}} | \text{Data}_{\text{observed}}) \propto P(\text{Data}_{\text{observed}} | \text{Q}_{\text{missing}}) \times P(\text{Q}_{\text{missing}})
$$

其中：

*   $P(\text{Q}_{\text{missing}})$ 是我们的先验 (Prior)：在看到该患者数据前，我们对Q值的信念（例如，来自人群统计的分布）。
*   $P(\text{Data}_{\text{observed}} | \text{Q}_{\text{missing}})$ 是似然 (Likelihood)：假设一个Q值，观测到该患者其他数据的可能性。
*   $P(\text{Q}_{\text{missing}} | \text{Data}_{\text{observed}})$ 是后验 (Posterior)：我们的最终答案，即在综合所有证据后，我们对缺失Q值的更新后的、更精确的概率分布。

### 12.3.3 Manim 概念 12.1：贝叶斯更新动画

为了在Manim中直观展示贝叶斯更新过程，我们将创建一个动画来演示信念（概率分布）是如何根据数据而演变的。

1.  场景 1 (Prior)： 使用 `Axes`[^13]绘制一个宽泛的概率分布（例如高斯分布，使用 `pm.Normal` 定义[^14]），标记为“先验信念 (Prior Belief)”。这代表了我们最初的不确定性[^15]。
2.  场景 2 (Likelihood)： 在场景中引入一个新的、通常更窄的分布，标记为“观测数据 (Likelihood)”。
3.  场景 3 (Posterior)： 使用 `Transform` 动画，将“先验”分布变形为“后验”分布。这个新的后验分布[^16]将位于先验和似然之间，并且其宽度（代表不确定性）会减小。

在技术上，这可以通过一个 `ValueTracker`[^17]来驱动，使其平滑地插值分布的均值和标准差，从而动态地从先验过渡到后验。

### 12.3.4 Python 教程 12.1：使用 PyMC 进行贝叶斯插补

我们将使用强大的概率编程库 PyMC[^19]来实现贝叶斯插补。

PyMC 中处理缺失数据的关键机制是使用 `numpy.ma.masked_array`（掩码数组）[^21]。

当 PyMC 的观测变量（`observed`）接收到一个掩码数组时，它会自动执行以下操作：

1.  对于 `mask=False` 的数据点（即观测到的值），它会将其用于计算模型的似然。
2.  对于 `mask=True` 的数据点（即缺失值），PyMC 会自动将其提升为模型中的一个自由随机变量（即一个待估计的参数）[^19]。

示例代码：插补缺失的角膜Q值：

```python
import pymc as pm
import numpy as np

# 1. 患者数据，其中Q值为 np.nan (缺失)
# 假设我们有5个数据点，第5个缺失
q_data = np.array([-0.2, -0.25, -0.18, -0.22, np.nan])

# 2. 创建一个掩码数组
# np.isnan(q_data) 会返回 [F, F, F, F, T]
masked_q = np.ma.masked_array(q_data, mask=np.isnan(q_data))

with pm.Model() as imputation_model:
   # 3. 定义Q值总体的先验分布（基于人群统计）
   pop_mean = pm.Normal('pop_mean', mu=-0.26, sigma=0.1)
   pop_std = pm.HalfNormal('pop_std', sigma=0.1)
   
   # 4. 建模数据（包括缺失值）
   # PyMC 会自动为 masked_q 创建一个待估计的变量
   q_observed = pm.Normal('q_obs', mu=pop_mean, sigma=pop_std, 
                          observed=masked_q)
   
   # 5. 运行 MCMC 采样
   trace = pm.sample(2000)

# 6. 结果分析
# trace['q_obs_missing'] (或类似名称) 将包含第5个数据点的后验分布
# 我们可以提取这个分布的均值和标准差，作为插补结果
imputed_mean = trace.posterior['q_obs_missing'].mean().item()
imputed_std = trace.posterior['q_obs_missing'].std().item()

print(f"插补的Q值: {imputed_mean:.3f} ± {imputed_std:.3f}")
```

这种方法的结果不是一个单一的插补值，而是一个完整的概率分布，精确地量化了我们对该缺失值的不确定性。

### 12.3.5 高级说明：使用高斯过程 (GP) 插补相关数据

上述教程适用于插补独立的标量值。但如果缺失的数据是相关的（例如角膜地形图的一部分缺失），情况会怎样？独立插补每个像素显然是错误的。

对于这类空间或时间序列数据，高斯过程 (GP) 是理想的插补工具。GP 是一种贝叶斯非参数模型，非常适合对连续函数进行建模并“填补”数据中的空白[^23]。

然而，需要注意的是，GP模型虽然功能强大，但实现复杂且“对核函数（Kernel）的选择高度敏感”。在某些情况下，简单的插值方法（如三次样条插值）可能同样有效甚至更优[^23]。工程决策需要权衡模型的复杂性和实际效果。

## 12.4 主题三：定量风险评估框架

现在我们有了两组不确定性：

1.  系统不确定性： 来自FSA的公差敏感性地图。
2.  数据不确定性： 来自贝叶斯插补的患者参数后验分布。

最后一步是将这些信息整合到一个正式的框架中，以量化临床风险并据此作出设计决策。

### 12.4.1 在眼科设计中定义风险

医疗器械风险管理的黄金标准是 ISO 14971[^26]。该标准主要关注的是患者安全，即避免物理伤害。

在眼科镜片设计中，主要的“风险”通常不是物理伤害，而是设计失败或临床失败。对于渐进多焦点镜片 (PAL)，最主要的临床失败模式是“不适应” (non-adaptation)[^29]。

因此，我们的核心任务是将 ISO 14971 的形式化风险管理流程，应用于“不适应”这个看似“软”的临床问题上。我们将把“不适应”视为一种“危险情况” (hazardous situation)[^28]，它可能导致“伤害” (harm)，例如患者感到眩晕、恶心，甚至增加跌倒的风险（特别是老年人）[^33]，或是镜片被退回导致的商业和声誉损害。

### 12.4.2 风险控制层级 (ISO 14971)

ISO 14971 最核心的原则之一是其风险控制的层级结构。降低风险的措施不是可以随意选择的，它们必须遵循一个严格的优先顺序[^26]。

我们将这个监管层级结构转化为眼科镜片设计的具体工程决策：

**表 12.1：ISO 14971 风险控制层级在眼科镜片设计中的应用**

| 优先级 | 风险控制措施 (ISO 14971) | 眼科设计实例 |
| :--- | :--- | :--- |
| 1 (最高) | 本质安全设计 (Inherently Safe Design)[^35] | 通过优化算法，设计一个在数学上对患者数据不确定性和制造公差都不敏感（鲁棒）的镜片光学曲面。从源头上消除像差或晃动感。 |
| 2 (中等) | 防护措施 (Protective Measures)[^26] | 在镜片上添加功能性涂层（例如，抗反射涂层以减少鬼影，防刮涂层）。这不改变核心光学设计，但能防护特定风险。 |
| 3 (最低) | 安全信息 (Information for Safety)[^26] | 在临床决策支持报告中提供明确的警告。例如：“警告：检测到不适应风险 > 80%。不推荐用于夜间驾驶。” |

这个框架迫使工程师优先选择第一级（通过泛函分析和稳健优化改进设计本身），而不是依赖第三级（将风险转移给临床医生）。

### 12.4.3 Manim 概念 12.2：不确定性传播的可视化

为了将患者不确定性（来自12.3）与最终的性能风险联系起来，我们需要可视化不确定性传播 (Uncertainty Propagation)。

1.  左侧： 绘制 Manim 概念 12.1 中的“后验信念”高斯分布（例如，插补的Q值）。
2.  中间： 一个“黑匣子” `Rectangle`，标记为“光学模型 $f(\text{params})$”。
3.  右侧： 一个 `BarChart`（条形图）[^38]，标记为“性能分布 (MTF)”。
4.  动画：
    *   大量 `Dot`[^17]从左侧的后验分布中“采样”飞出。
    *   `Dot` 穿过中间的黑匣子。
    *   `Dot` 降落在右侧的条形图中，动态地构建起一个性能直方图[^40]。

这个动画在视觉上复现了一个蒙特卡洛模拟 (Monte Carlo simulation)[^41]。在Python实践中，我们可以使用 `uncertainties` 库更高效地完成这一步[^43]。通过将贝叶斯插补的结果（均值和标准差）封装为 `ufloat` 对象（例如 `q_value = ufloat(-0.25, 0.05)`），然后将其输入到我们的Python光学函数中，`uncertainties` 库会自动计算和传播误差，输出的MTF也将是一个 `ufloat` 对象（例如 `mtf = ufloat(0.45, 0.08)`）。

### 12.4.4 Python 教程 12.2：使用 scikit-learn 构建预测性风险模型

现在我们需要量化“不适应”这一临床风险。我们将使用历史数据训练一个机器学习模型来实现。

工具： `scikit-learn` 库及其 `DecisionTreeClassifier`（决策树分类器）[^45]。

方法：

1.  准备数据： 加载一个历史患者数据库 (`historical_patients.csv`)。
    *   `X`（特征）：包含患者的人口统计学数据、屈光度、插补的Q值、设计的像差指标等。
    *   `y`（标签）：一个二元标志，1 代表该患者最终“不适应”，0 代表“适应”。
2.  训练模型：
    ```python
    from sklearn.tree import DecisionTreeClassifier
    #... 加载 X_train, y_train...
    clf = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    clf.fit(X_train, y_train)
    ```
3.  量化新患者的风险： 对于一个新患者，我们提取其特征 `new_patient_features`：
    ```python
    # predict_proba 返回每个类别的概率 [P(adapt), P(non-adapt)]
    risk_probabilities = clf.predict_proba([new_patient_features])[0]
    
    # 提取“不适应”的概率
    non_adaptation_risk = risk_probabilities[1]
    
    print(f"量化的临床不适应风险: {non_adaptation_risk * 100:.1f}%")
    ```

这个 `non_adaptation_risk`（例如 85%）就是我们寻求的“量化的设计方案风险”[^1]。

### 12.4.5 Manim 概念 12.3：风险评估决策树的可视化

上述 `DecisionTreeClassifier` 模型的一个主要优势是其可解释性。它不像神经网络那样是一个“黑匣子”。我们可以将其决策逻辑可视化。

我们将使用 Manim 的 `Graph` Mobject[^46]来绘制这个决策树。幸运的是，已有现成的 Manim 扩展（如 ManimML）提供了执行此操作的类（例如 `DecisionTreeDiagram`）[^47]。

这个动画将逐步 `Create` 整个树形结构：

*   分裂节点 (SplitNode)： 使用 `VGroup`[^47]包含一个 `SurroundingRectangle` 和一个 `Text`，显示决策规则（例如“Q值 <= -0.3?”）。
*   叶节点 (LeafNode)： 使用 `Group`[^47]包含一个 `Rectangle` 和一个 `Text`，显示最终的风险分类（例如“风险：高”）。
*   边 (Edges)： 使用 `Line`[^47]连接各个节点。

这个可视化使临床医生和工程师都能准确理解模型是如何得出其风险评分的。

## 12.5 整合项目：项目12 - 个性化老花镜设计与临床决策支持

本周的最终项目是将上述所有三个主题合成为一个完整的、自动化的工作流，以解决一个真实的临床场景。

场景： 收到一个新患者的数据文件 `patient_123.json`，用于设计个性化老花镜片。该文件包含一些 `null` 缺失字段。

### 12.5.1 步骤 1：数据摄取与贝叶斯插补 (应用 12.3)

1.  加载 `patient_123.json`。
2.  识别缺失的字段（例如 `corneal_q_value: null`）。
3.  使用 `numpy.ma.masked_array`[^22]准备数据。
4.  运行教程 12.1 中的 PyMC 模型，为缺失的 `corneal_q_value` 生成后验分布。
5.  输出： 将结果（均值和标准差）存储为 `uncertainties` 库的 `ufloat` 对象[^43]。

    `patient_imputed = {'q_value': ufloat(-0.25, 0.05),...}`

### 12.5.2 步骤 2：随机设计与不确定性传播 (应用 12.2 & 12.4.3)

1.  将包含 `ufloat` 对象的 `patient_imputed` 字典传递给在前面章节中构建的（假设已完全用Python编写的）光学设计函数。
2.  由于 `uncertainties` 库的魔力[^43]，所有数学运算（加、减、乘、sin、cos等）都会自动传播不确定性。
3.  输出： 设计的最终性能指标（MTF、畸变、晃动感指数）也将是 `ufloat` 对象。

    `design_performance = {'mtf_50lpmm': ufloat(0.45, 0.08), 'sway_index': ufloat(1.5, 0.4)}`

这量化了由于患者数据不确定性而导致的性能不确定性。

### 12.5.3 步骤 3：临床风险量化 (应用 12.4.4)

1.  从 `patient_imputed` 和 `design_performance` 中提取特征的标称值（即均值，例如 `patient_imputed['q_value'].nominal_value`）。
2.  将这些特征向量输入到教程 12.2 中训练好的 `DecisionTreeClassifier` 中。
3.  输出： 获得一个明确的、量化的临床风险评分。

    `risk_score = clf.predict_proba(features)`
    `# 结果: risk_score = 0.85 (85% 的不适应风险)`

### 12.5.4 步骤 4：生成临床决策支持报告 (应用 12.4.2)

最后一步是生成一份专业的PDF报告。这一步至关重要，因为它本身就是 ISO 14971 层级 3：“安全信息” 的正式实施[^26]。

这份报告是我们作为工程师的审计追踪 (audit trail)。它证明我们已经：

1.  识别了风险（步骤 3：85% 的风险）。
2.  传达了风险（步骤 4：生成PDF）。
3.  从而将最终的接受/拒绝决策权（以及责任）转移给掌握了充分信息的临床医生。

技术实现：

*   虽然 Streamlit 适用于快速原型制作，但生成可存档的PDF需要更稳健的方法。
*   我们将使用 Jinja2 库来填充一个预先设计好的 HTML 模板，然后使用 `pdfkit`[^48]或 `fpdf`[^49]库将该 HTML 转换为 PDF。

**表 12.2：临床决策支持报告 (CDSR) 内容**

| 报告字段 | 示例值 |
| :--- | :--- |
| 患者 ID | `patient_123` |
| 设计类型 | 个性化渐进镜片（老花） |
| 数据质量 | 警告： 检测到不完整数据。 |
| 插补值 (95% CI) | 角膜 Q 值 (估计): -0.25 ± 0.05 |
| 性能不确定性 (95% CI) | 估计 MTF@50lp/mm (中央): 0.45 ± 0.08<br>晃动感指数 (估计): 1.5 ± 0.4 |
| 量化临床风险 | 不适应风险: 85% (高) |
| 临床指南 (ISO 14971) | 警告： 基于模型预测，该患者有极高的“不适应”风险[^29]。患者可能会经历眩晕、周边视物变形或“晃动感”[^32]。在制造镜片前，强烈建议进行临床会诊讨论这些风险。 |

## 12.6 本章小结

本章完成了个性化设计从理论到实践的最后一步。我们建立了一个强大的工作流来应对设计过程中不可避免的敌人：不确定性。

我们采用了三管齐下的方法：

1.  泛函敏感性分析 (FSA)：使我们能够“看到”设计的脆弱性，指导我们进行公差分配和稳健性优化。
2.  贝叶斯插补 (Bayesian Imputation)：使我们能够以数学上严谨和诚实的方式处理不完整、含噪声的患者数据，将“未知”转化为可量化的“概率分布”。
3.  风险评估框架 (ISO 14971)：提供了一个符合行业标准的、可辩护的流程，用于将所有量化的不确定性转化为一个单一的、可操作的临床风险评分。

最终，成功的个性化设计不在于盲目地消除所有不确定性（这是不可能的），而在于精确地量化它，并智能地管理它。本章已经为您提供了实现这一目标的 Python 工具和数学框架。

[^1]: 《实用泛函分析：Python驱动的眼视光学镜片设计优化》课程大纲
[^2]: [Fast sensitivity control method with differentiable optics](https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-6-14404)
[^3]: [Bayesian personalized treatment selection strategies that integrate predictive with prognostic determinants - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341533/)
[^4]: [What Is Sensitivity Analysis? - MATLAB & Simulink - MathWorks](https://www.mathworks.com/help/sldo/ug/what-is-sensitivity-analysis.html)
[^5]: [Sensitivity analysis - Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_analysis)
[^6]: [contrast sensitivity functions: Topics by Science.gov](https://www.science.gov/topicpages/c/contrast+sensitivity+functions.html)
[^7]: [derive sensitivity equations: Topics by Science.gov](https://www.science.gov/topicpages/d/derive+sensitivity+equations)
[^8]: [Applied Science Division - eScholarship](https://escholarship.org/content/qt8rh781hr/qt8rh781hr_noSplash_674ec1027cccb444a44bb788909d74d8.pdf)
[^9]: [How to Perform a Sensitivity Analysis in COMSOL Multiphysics](https://www.comsol.com/blogs/how-to-perform-a-sensitivity-analysis-in-comsol-multiphysics)
[^10]: [Gradient Calculation and Sensitivity Analysis - MIT OpenCourseWare](https://ocw.mit.edu/courses/ids-338j-multidisciplinary-system-design-optimization-spring-2010/ce1a087c39ebe629ff93ed57686ada0a_MITESD_77S10_lec09.pdf)
[^11]: [Missing Data — PyMC example gallery](https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/18-Missing_Data.html)
[^12]: [Applications of Bayesian Statistics in Healthcare for Improving Predictive Modeling, Decision-Making, and Adaptive Personalized Medicine | International Journal of Applied Health Care Analytics](https://norislab.com/index.php/IJAHA/article/view/99)
[^13]: [2D Graphs | Tutorial 2, Manim Explained - YouTube](https://www.youtube.com/watch?v=jFqYq9quBds)
[^14]: [FULL Manim Code | Normal Distribution - YouTube](https://www.youtube.com/watch?v=uHd1ZIJfSPo)
[^15]: [Bayesian Updating - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/bayesian-updating/)
[^16]: [02 Data Science Interactive: Bayesian Updating - YouTube](https://www.youtube.com/watch?v=FSf_zYiU5Yw)
[^17]: [Animations - FlyingFrames v0.11.0](https://flyingframes.readthedocs.io/en/v0.11.0/ch4.html)
[^18]: [Manim Tutorial Series E04: Updater Functions | Mathematical Animations WITH EASE](https://www.youtube.com/watch?v=vUIfNN6Bs_4)
[^19]: [Introductory Overview of PyMC — PyMC 5.26.1 documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html)
[^20]: [PyMC Example Gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html)
[^21]: [3. Tutorial — PyMC 2.3.6 documentation](https://pymcmc.readthedocs.io/en/latest/tutorial.html)
[^22]: [Missing Data Imputation With Bayesian Networks in Pymc - DS lore](http://nadbordrozd.github.io/blog/2017/03/05/missing-data-imputation-with-bayesian-networks/)
[^23]: [Comparison of Gaussian Processes Methods to Linear methods for Imputation of Sparse Physiological Time Series - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC6561479/)
[^24]: [Sparse multi-output Gaussian processes for online medical time series prediction - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341595/)
[^25]: [Analyzing Healthcare Trends with Gaussian Process Regression - DEV Community](https://dev.to/aionlinecourse/analyzing-healthcare-trends-with-gaussian-process-regression-e9f)
[^26]: [The illustrated guide to risk management for medical devices and ISO 14971](https://medicaldevicehq.com/articles/the-illustrated-guide-to-risk-management-for-medical-devices-and-iso-14971/)
[^27]: [ISO 14971:2019 Risk Management for Medical Devices: 2025 Implementation Guide](https://www.complizen.ai/post/iso-14971-medical-device-risk-management)
[^28]: [Risk management for medical devices and the new BS EN ISO ... - BSI](https://www.bsigroup.com/globalassets/localfiles/en-us/images/wp_risk_management_web.pdf)
[^29]: [HOW TO MANAGE A NON-ADAPTATION - Horizons Optical](https://horizonsoptical.com/us/blog/how-to-manage-a-non-adaptation/)
[^30]: [Tips for better wearing comfort of progressive lenses - ZEISS](https://www.zeiss.com/vision-care/us/eye-health-and-care/driving-mobility/tips-for-better-wearing-comfort-of-progressive-lenses.html)
[^31]: [Pros and Cons of Progressive Lenses - American Academy of Ophthalmology](https://www.aao.org/eye-health/glasses-contacts/pros-cons-progressive-lenses-computer-glasses)
[^32]: [Unlocking Clarity: An Optician's Guide to Mastering Progressive Non-Adapts](https://www.acculab.net/post/unlocking-clarity-optician-s-guide-to-mastering-progressive-non-adapts)
[^33]: [Prismatic displacement effect of progressive multifocal glasses on reaction time and accuracy in elderly people - PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC4025939/)
[^34]: [The Ultimate Guide to ISO 14971 Risk Management - Atlassian Community](https://community.atlassian.com/forums/App-Central-articles/The-Ultimate-Guide-to-ISO-14971-Risk-Management/ba-p/2999670)
[^35]: [Risk control in the third edition of ISO 14971 - BSI Compliance Navigator](https://compliancenavigatorppd.bsigroup.com/en/medicaldeviceblog/risk-control-in-the-third-edition-of-iso-14971/)
[^36]: [Risk management for medical devices and the new BS EN ISO 14971](https://www.medical-device-regulation.eu/wp-content/uploads/2020/09/WP_Risk_management_web.pdf)
[^37]: [ISO 14971 Medical Devices Risk Management](https://qualitation.co.uk/iso-14971)
[^38]: [BarChart - Manim Community v0.19.0](https://docs.manim.community/en/stable/reference/manim.mobject.graphing.probability.BarChart.html)
[^39]: [Data Visualization: Bar Chart Animations With Manim — Andres ...](https://medium.com/@andresberejnoi/data-visualization-bar-chart-animations-with-manim-andres-berejnoi-75ece91a2da4)
[^40]: [Probability Simulators | Tutorial 6, Manim Explained - YouTube](https://www.youtube.com/watch?v=tqc35g2hPng)
[^41]: [Uncertainty Propagation: Monte Carlo - YouTube](https://www.youtube.com/watch?v=Wdob95zfqe8)
[^42]: [Monte Carlo approach to calculating uncertainty using Python | by Thombson Chungkham](https://medium.com/@ch.thombson18/monte-carlo-approach-to-calculating-uncertainty-using-python-7d4298a307a8)
[^43]: [Uncertainties — uncertainties](https://uncertainties.readthedocs.io/)
[^44]: [Uncertainty propagation - risk-engineering.org](https://risk-engineering.org/notebook/uncertainty-propagation.html)
[^45]: [1.10. Decision Trees — scikit-learn 1.7.2 documentation](https://scikit-learn.org/stable/modules/tree.html)
[^46]: [Graph - Manim Community v0.19.0](https://docs.manim.community/en/stable/reference/manim.mobject.graph.Graph.html)
[^47]: [CoCalc -- decision_tree.py](https://cocalc.com/github/helblazer811/ManimML/blob/main/manim_ml/decision_tree/decision_tree.py)
[^48]: [Generate pdf report from dataframe and charts using Jinja2 and ...](https://discuss.streamlit.io/t/generate-pdf-report-from-dataframe-and-charts-using-jinja2-and-pdfkit-in-streamlit/42891)
[^49]: [Creating a PDF file generator - Using Streamlit](https://discuss.streamlit.io/t/creating-a-pdf-file-generator/7613)