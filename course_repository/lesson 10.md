# 第10章：分布理论与散射建模

## 10.1 引言：从理想成像到现实散射

在模块3的前几周，我们集中探讨了设计鲁棒性，即如何在存在制造公差和不确定性的情况下优化镜片性能。本周，我们将处理一个对视觉质量有直接负面影响的物理现象：光散射 (Light Scattering)。

在眼视光学中，散射表现为“雾霾感”、光晕或“眩光”。这种现象可能源于镜片材料本身（例如，材料内部的微小杂质）、表面粗糙度，甚至是眼球内部（如白内障）。一个（在数学上）完美的镜片设计如果忽略了散射效应，那么它在真实世界中的表现将远低于预期。

为了在设计阶段就“预测”并“补偿”这种雾霾感，我们需要一个能够描述它的数学模型。然而，描述一个理想的、无限清晰的点光源，需要一种超越传统函数概念的语言。

本章的目标是建立一个从抽象数学到实际工程应用的完整链条：
1.  理论： 引入“分布理论”(Distribution Theory) 和狄拉克 $delta$ 函数 (Dirac $delta$ function)，将其作为描述理想光学系统——即点扩散函数 (PSF)——的数学语言[^2]。
2.  建模： 展示如何使用卷积算子 (Convolution Operator) 将理想PSF与散射效应结合，从而建立一个数学模型来模拟散射[^5]。
3.  加速： 解决卷积带来的巨大计算量问题，利用傅里叶变换 (FFT) 和卷积定理实现高效模拟[^7]。

最终，学员将能够使用Python构建一个散射效应模拟器，将一个抽象的数学概念（分布）转化为一个具体的、可计算的工程工具。

## 10.2 概念：点扩散函数 (PSF) 与狄拉克 $\delta$ 函数

### 10.2.1 什么是点扩散函数 (PSF)？

在光学系统中，点扩散函数 (Point Spread Function, PSF) 是一个基础且至关重要的概念。它的定义非常简单：一个理想的点光源（例如，一个无限远的、无限小的星星）在成像平面（如视网膜或相机传感器）上形成的图像[^6]。

在一个完美的光学系统中，一个点光源应该成像为一个完美的点。然而，由于衍射（光波的本性）和像差（系统的缺陷），实际的PSF总是一个模糊的光斑（例如艾里斑）[^10]。PSF是衡量光学系统质量的有效指标；它描述了系统将一个“点”模糊（blur）或“扩散”（spread）的程度[^6]。

### 10.2.2 理想PSF的数学语言：狄拉克 $\delta$ 函数

我们如何用数学来描述那个“完美的、未被模糊的”理想点光源？

这个“函数”必须具备以下特性：它在原点处无限高（无限集中），在其他任何地方都为零，并且其总积分（总能量）为1[^3]。然而，没有一个“普通”函数能满足所有这些条件[^3]。

为了解决这个问题，数学家（特别是 Paul Dirac 和 Laurent Schwartz）发展了分布理论 (Distribution Theory)，也称为广义函数 (Generalized Functions)[^2]。

定义：分布 (Distribution)
一个分布（或广义函数）不是通过它在每个点 $x$ 的值来定义的，而是通过它如何作用于一个“测试函数” $\phi(x)$ （通常是一个无限平滑且在有限区间外为零的函数）来定义的。它是一个作用于函数空间的连续线性泛函 (continuous linear functional)[^12]。

在这个框架下，我们可以严格定义那个理想的“点”：

狄拉克 $\delta$ 函数 (Dirac $delta$ Function)
$\delta$ 函数是一个分布，它作用于一个测试函数 $\phi(x)$ 时，会“筛选”出该函数在原点的值[^3]。

$$
\langle \delta, \phi \rangle = \int_{-\infty}^{\infty} \delta(x) \phi(x) \, dx = \phi(0)
$$

这个定义完美地捕捉了“点”的特性。

### 10.2.3 $\delta$ 函数与光学的联系

PSF 和 $\delta$ 函数之间的联系是本章的核心。

1.  理想系统： 一个没有像差、没有衍射、没有散射的（假设的）完美光学系统，其点扩散函数 (PSF) 就是一个狄拉克 $\delta$ 函数[^4]。输入一个 $\delta$ (点)，输出也是一个 $\delta$ (点)。
2.  现实系统： 现实世界中的PSF是模糊的。
3.  散射与 $\delta$ 函数： 散射可以被看作是PSF形态演变的过程。在一个散射介质中（如雾气或毛玻璃），当光源与探测器之间的光学距离 $\tau$ 趋近于0时（即没有散射），PSF 接近于一个 $\delta$ 函数。随着 $\tau$ 增加，光子被多次散射，PSF的形状从一个尖锐的峰值（$\delta$ 状）演变为一个更宽、更平坦的分布[^15]。

因此，$\delta$ 函数为我们提供了描述“清晰度”的绝对基准。

## 10.3 建模：用卷积算子模拟散射

我们已经定义了单个点的成像 (PSF)。那么，一个复杂的物体（可以看作无数个点的集合）是如何成像的呢？

在光学中，图像形成过程被建模为一个线性系统[^6]。这个线性特性意味着，一个物体的总图像，等于该物体上每个点独立成像（即，每个点产生一个PSF）后的总和（叠加）[^6]。

这个“在每个点上应用PSF并求和”的过程，在数学上被精确地定义为卷积 (Convolution)[^6]。

图像形成 = 物体 $\circledast$ PSF

其中 $\circledast$ 代表卷积操作。在这个模型中，PSF充当了卷积核 (Convolution Kernel)[^6]。

这个模型为我们模拟散射提供了一个强大的工具：

1.  理想图像： 我们可以将一个清晰的、无散射的图像视为“理想物体”。
2.  散射核： 我们可以将散射效应（例如，由材料特性引起的特定“雾霾”模式）建模为一个单独的PSF，我们称之为“散射核”。这个核通常是一个中心亮、向外衰减的函数（例如，高斯函数或更复杂的BSDF模型）[^17]。
3.  模拟图像： 将“理想图像”与“散射核”进行卷积，我们就能得到模拟的、包含散射效应的最终图像[^5]。

理想图像 $\circledast$ 散射核 = 模拟的模糊图像

## 10.4 Manim 演示 10：(A) “看见” $\delta$ 函数

$\delta$ 函数本身是无法直接绘制的[^3]。正如 Manim 动画的核心理念是“看见数学”，我们将通过动画展示 $\delta$ 函数的“本质”——作为一个极限过程。

演示目标： 将 $\delta(x)$ 动画化，展示其作为高斯函数 $g_\sigma(x)$ 在 $\sigma \to 0$ 时的极限[^19]。

高斯函数（归一化）为：

$$
g_\sigma(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}}
$$

在 $\sigma \to 0$ 的极限下，这个函数序列就（在分布意义上）收敛于 $\delta(x)$[^21]。

Manim 动画逻辑 (伪代码):

```python
from manim import *

class DiracDeltaAnimation(Scene):
   def construct(self):
       # 1. 创建坐标轴
       axes = Axes(x_range=[-3, 3], y_range=[0, 4])
       
       # 2. 定义高斯函数 [35]
       def gaussian_func(x, sigma):
           # )
           return (1 / (sigma * np.sqrt(2 * PI))) * \
                  np.exp(-0.5 * (x / sigma)**2)

       # 3. 创建一个 ValueTracker 来控制 sigma [36]
       sigma = ValueTracker(1.0) # 初始 sigma

       # 4. 创建一个始终更新的图像
       graph = always_redraw(lambda:
           axes.plot(lambda x: gaussian_func(x, sigma.get_value()),
                     color=BLUE)
       )
       
       # 5. 添加标题，显示 sigma 的值
       label = always_redraw(lambda:
           MathTex(f"\\sigma = {sigma.get_value():.2f}") \
          .next_to(axes, UP)
       )
       
       self.add(axes, graph, label)
       self.wait()
       
       # 6. 核心动画：让 sigma 趋近于 0
       self.play(sigma.animate.set_value(0.1), run_time=5)
       self.wait()
```

动画效果： 学员将“看见”一个蓝色曲线。随着 $\sigma$ 变小，曲线变得越来越窄、越来越高，但其下方的总面积（通过Manim的 axes.get_area 可视化）始终保持为1。这个过程直观地展示了 $\delta$ 函数的“无限高、无限窄、积分为1”的特性[^19]。

## 10.5 挑战：卷积的计算效率

我们建立的模型（图像 $\circledast$ 散射核）在计算上是昂贵的。

直接的2D卷积（如项目10中手动实现的版本）涉及一个“滑动窗口”操作[^23]。对于 $N \times N$ 的图像和 $M \times M$ 的核：
1.  将 $M \times M$ 的核覆盖在图像的 $N \times N$ 个像素上。
2.  在每个位置，执行 $M^2$ 次乘法和 $M^2-1$ 次加法。

总计算复杂度约为 $O(N^2 \cdot M^2)$[^7]。

如果我们在处理一个 $1000 \times 1000$ 像素的图像，并使用一个 $50 \times 50$ 像素的大型散射核，计算量将达到 $1000^2 \times 50^2 = 25$ 亿次操作。这对于需要实时反馈的设计迭代来说太慢了[^24]。

## 10.6 解决方案：利用快速傅里叶变换 (FFT) 加速

幸运的是，数学和物理学为我们提供了一个优雅且极其高效的捷径：卷积定理 (Convolution Theorem)。

卷积定理：
空间域中的卷积，等价于频率域中的逐点乘法[^7]。

$\text{FFT}(\text{Image} \circledast \text{Kernel}) = \text{FFT}(\text{Image}) \times \text{FFT}(\text{Kernel})$

这个定理在光学中有着深刻的物理意义：
*   $\text{FFT}(\text{Image})$ 是图像的频谱。
*   $\text{FFT}(\text{Kernel})$（即 $\text{FFT}(\text{PSF})$）正是我们在第8周和第9周讨论过的 光学传递函数 (Optical Transfer Function, OTF)[^16]。

因此，这个“数学技巧”实际上就是切换到频率空间（OTF空间）中去工作，这在光学上是完全自然的。

FFT加速算法：
1.  计算图像的FFT：$\mathcal{F}(\text{Image})$
2.  计算核的FFT：$\mathcal{F}(\text{Kernel})$ (即OTF)
3.  在频率域中相乘：$R = \mathcal{F}(\text{Image}) \cdot \mathcal{F}(\text{Kernel})$
4.  计算逆FFT：$\text{Result} = \mathcal{F}^{-1}(R)$

计算效率：
快速傅里叶变换 (FFT) 是一种计算 $N \times N$ 图像傅里叶变换的革命性算法，其复杂度仅为 $O(N^2 \log N)$[^7]。

使用FFT进行卷积的总复杂度（包括两次FFT和一次IFFT）约为 $O(N^2 \log N)$。

对比：
*   直接卷积： $O(N^2 M^2)$ (依赖于核 $M$ 的大小)
*   FFT卷积： $O(N^2 \log N)$ (几乎不依赖于核 $M$ 的大小)[^24]

对于大型散射核（$M$ 很大），FFT方法的速度提升是指数级的，可能快上百倍甚至上千倍[^28]。在Python的 scipy.signal 库中，这两种方法分别对应：
*   convolve2d：(通常是)直接卷积
*   fftconvolve：FFT卷积[^30]

基准测试（如[^24]所示）明确证实，对于大图像和中等到大型核，fftconvolve 的性能远超 convolve2d。

## 10.7 Manim 演示 10：(B) 卷积过程与效率对比

演示目标：
1.  可视化2D卷积的“滑动窗口”过程。
2.  对比直接卷积与FFT卷积的计算速度。

Manim 动画逻辑 (伪代码):

```python
from manim import *
# manim-ml 是一个有用的Manim插件 [31]
from manim_ml.neural_network import Convolutional2DLayer 

class ConvolutionVisualizer(Scene):
   def construct(self):
       # 1. (A) 滑动窗口动画
       # 创建两个矩阵 [37]
       image_matrix = Matrix([[...]]) # 10x10
       kernel_matrix = Matrix([[...]]) # 3x3
       
       # 创建一个高亮的“窗口” [37]
       window = SurroundingRectangle(image_matrix.get_entries()[:3][:3])
       
       # 动画：[31, 38]
       # self.play(window.animate.shift(RIGHT))
       #... 循环展示窗口滑动、计算、填充结果...
       # (这部分动画可以使用 manim-ml 库 [31] 
       #  或手动实现 [32])

       # 2. (B) 效率对比 
       title_direct = Text("Direct: convolve2d")
       title_fft = Text("FFT: fftconvolve").next_to(title_direct, RIGHT)
       
       # 模拟长时间运行
       timer_direct = ValueTracker(0)
       timer_fft = ValueTracker(0)
       
       # 播放滑动窗口动画 (很慢)
       self.play(window.animate... , 
                 timer_direct.animate.set_value(11.2), 
                 run_time=10) # 模拟  中的 11.2 秒

       # 播放FFT动画 (很快)
       # 
       fft_anim_group = VGroup(
           # (显示图像 -> FFT(图像) -> 乘以 OTF -> IFFT(结果))
       )
       self.play(FadeIn(fft_anim_group), 
                 timer_fft.animate.set_value(0.58), 
                 run_time=1) # 模拟  中的 0.58 秒
```

动画效果： 学员首先会看到一个 3x3 的小窗口在图像矩阵上缓慢“滑动”，并逐个计算输出值[^23]。然后，场景切换到效率对比：一个“直接卷积”的计时器缓慢增加到 11.2 秒，而“FFT卷积”的计时器几乎瞬间完成（0.58秒）[^24]。这种视觉冲击力极强地证明了为什么FFT是工程实践中的唯一选择。

## 10.8 Python 实践项目 10：散射效应模拟器

目标： 利用本章所学知识，构建一个功能性的Python工具，用于模拟光学散射对输入图像的影响。

核心工具链： NumPy, SciPy, Matplotlib, Pillow (PIL)

### 10.8.1 任务1：生成2D高斯散射核

散射通常可以用一个中心强、四周弱的函数来近似，高斯函数是最好的起点。你需要编写一个函数，根据给定的size（核大小，如 $31 \times 31$）和sigma（散射宽度）生成一个2D高斯核。
*   使用 np.linspace 和 np.meshgrid 来创建坐标网格[^18]。
*   应用2D高斯公式 $K(x, y) = A \cdot e^{-\frac{x^2 + y^2}{2\sigma^2}}$[^18]。
*   关键： 确保核被归一化（即所有元素的总和为1），以保证卷积过程不改变图像的整体亮度。

下面是一个示例函数:[^18]

```python
import numpy as np

def create_gaussian_kernel(size, sigma=1.0):
   """
   使用 NumPy 创建一个 2D 高斯核 
   """
   # 创建坐标网格
   x = np.linspace(-(size // 2), size // 2, size)
   y = np.linspace(-(size // 2), size // 2, size)
   xx, yy = np.meshgrid(x, y)
   
   # 计算高斯函数
   kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
   
   # 归一化，使其总和为 1
   return kernel / np.sum(kernel)
```

### 10.8.2 任务2：实现（并对比）卷积

你将使用 SciPy 库中两个不同的函数来实现卷积，并亲身体验它们的性能差异。

```python
from scipy.signal import convolve2d, fftconvolve
from PIL import Image
import matplotlib.pyplot as plt
import time

# --- 设置 ---
KERNEL_SIZE = 51  # 一个大的散射核
SIGMA = 15.0
IMAGE_PATH = 'your_test_image.png' # 准备一张测试图片

# --- 1. 加载图像并转换为灰度图 ---
image = np.array(Image.open(IMAGE_PATH).convert('L'))

# --- 2. 创建散射核 ---
kernel = create_gaussian_kernel(KERNEL_SIZE, SIGMA)

# --- 3. (A) 使用 fftconvolve (FFT) ---
# [24, 30]
start_fft = time.time()
# 使用 fftconvolve，因为它对大型核更高效 
scattered_fft = fftconvolve(image, kernel, mode='same')
end_fft = time.time()
print(f"FFT Convolution (fftconvolve) 耗时: {end_fft - start_fft:.4f} 秒")

# --- 4. (B) 对比：使用 convolve2d (Direct) ---
# [39, 40]
# 警告：对于大核 (51x51)，这可能会非常慢！
start_direct = time.time()
scattered_direct = convolve2d(image, kernel, mode='same', boundary='symm')
end_direct = time.time()
print(f"Direct Convolution (convolve2d) 耗时: {end_direct - start_direct:.4f} 秒")

# --- 5. 可视化结果 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("原始图像")

axes[1].imshow(scattered_fft, cmap='gray')
axes[1].set_title(f"FFT 散射模拟 (Sigma={SIGMA})")

axes[2].imshow(kernel, cmap='gray')
axes[2].set_title(f"散射核 ({KERNEL_SIZE}x{KERNEL_SIZE})")

plt.tight_layout()
plt.show()
```

### 10.8.3 任务3（可选）：使用 OpenCV

OpenCV 库提供了一个高度优化的2D卷积函数 cv2.filter2D，它在底层也可能使用了FFT或其他加速技术。

```python
import cv2

# (创建 kernel 同上)
# (加载 image 同上, 确保为 np.float32)

# OpenCV 的 filter2D [33, 34]
# ddepth = -1 表示输出图像将具有与输入相同的深度
scattered_cv = cv2.filter2D(src=image.astype(np.float32), 
                           ddepth=-1, 
                           kernel=kernel)
```

学员应比较 fftconvolve 和 cv2.filter2D 的性能。

## 10.9 总结与展望

在本章中，我们完成了一次从高度抽象的数学理论到高性能计算实践的飞跃。

1.  理论基石： 我们引入了分布理论，将狄拉克 $\delta$ 函数[^3]确立为描述理想光学系统 (PSF) 的“通用语言”[^4]。
2.  物理建模： 我们证明了图像形成和散射过程在数学上是一个卷积算子[^6]。
3.  计算突破： 我们直面了直接卷积 $O(N^2 M^2)$ 的效率瓶颈[^24]，并利用卷积定理[^26]找到了解决方案。
4.  工程实践： 通过使用 FFT，我们将计算复杂度降低到 $O(N^2 \log N)$，使得在Python中（使用 scipy.signal.fftconvolve[^30]）进行实时散射模拟成为可能。

通过Manim动画，我们“看见”了 $\delta$ 函数的极限过程[^19]，并“感受”到了FFT加速的绝对必要性[^24]。

通往模块4的桥梁： 在第10周，我们已经构建了一个强大的、物理上精确的模拟器。但是，在实际的镜片设计中，我们如何将这个模拟器（一个Python脚本）与我们的CAD软件、优化循环和制造公差分析集成起来？我们如何将它从一个“脚本”变成一个“工具”？

在接下来的第11周“优化算法工程化”中，我们将学习如何将本项目中的 fftconvolve 逻辑封装到一个健壮的、可调用的 API 中，使其成为我们自动化设计工作流中一个可靠的组成部分。

[^1]: <https://mathworld.wolfram.com/GeneralizedFunction.html#:~:text=A%20generalized%20function%2C%20also%20called,the%20concept%20of%20a%20function.>
[^2]: [Distributions | Generalized functions - Applied Mathematics Consulting](https://www.johndcook.com/blog/2015/12/21/distributions/)
[^3]: [Three-dimensional point spread function and generalized amplitude transfer function of near-field flat lenses - Optica Publishing Group](https://opg.optica.org/abstract.cfm?uri=ao-49-30-5870)
[^4]: [Memory-less scattering imaging with ultrafast convolutional optical neural networks - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11177939/)
[^5]: [Point Spread Function (PSF) | Scientific Volume Imaging](https://svi.nl/Point-Spread-Function-%28PSF%29)
[^6]: [Convolution for optical broad-beam responses in scattering media - Wikipedia](https://en.wikipedia.org/wiki/Convolution_for_optical_broad-beam_responses_in_scattering_media)
[^7]: [Convolution and FFT - cs.Princeton](https://www.cs.princeton.edu/courses/archive/spr05/cos423/lectures/05fft.pdf)
[^8]: [The Point Spread Function - YouTube](https://www.youtube.com/watch?v=Tkc_GOCjx7E)
[^9]: [What is a Point Spread Function? - Ansys Optics](https://optics.ansys.com/hc/en-us/articles/42661723066515-What-is-a-Point-Spread-Function)
[^10]: [[2312.14356] Interactive simulation and visualization of point spread functions in single molecule imaging - arXiv](https://arxiv.org/abs/2312.14356)
[^11]: [Generalized Functions and Distribution - ijsri](https://ijsri.amberpublishers.in/resources/papers/p1.pdf)
[^12]: [Distribution (mathematical analysis) - Wikipedia](https://en.wikipedia.org/wiki/Distribution_(mathematical_analysis))
[^13]: [Generalized Functions — A Primer - Math ∩ Programming](https://www.jeremykun.com/2012/06/06/generalized-functions/)
[^14]: [Beam and Point Spread Functions :: Ocean Optics Web Book](https://www.oceanopticsbook.info/view/radiative-transfer-theory/level-2/beam-and-point-spread-functions)
[^15]: [What is the point spread function and optical transfer function and what uses are they in image processing - Signal Processing Stack Exchange](https://dsp.stackexchange.com/questions/59778/what-is-the-point-spread-function-and-optical-transfer-function-and-what-uses-ar)
[^16]: [Modelling and simulation of light scattering in optical systems - Digitale Bibliothek Thüringen](https://www.db-thueringen.de/receive/dbt_mods_00050550)
[^17]: [How to generate 2-D Gaussian array using NumPy? - GeeksforGeeks](https://www.geeksforgeeks.org/python/how-to-generate-2-d-gaussian-array-using-numpy/)
[^18]: [Show that Dirac Delta function is a limit of Gaussian function - YouTube](https://www.youtube.com/watch?v=BOuS1I5VBKc)
[^19]: [Representation of Dirac delta function as a limit of Gaussian Function. - YouTube](https://www.youtube.com/watch?v=b6NsoKCbUGE)
[^20]: [Stat 992: Lecture 03 Gaussian kernel smoothing.](https://pages.stat.wisc.edu/~mchung/teaching/stat992/ima03.pdf)
[^21]: [Reference for Dirac Delta function as gaussian - Mathematics Stack Exchange](https://math.stackexchange.com/questions/2833912/reference-for-dirac-delta-function-as-gaussian)
[^22]: [Convolutions with OpenCV and Python - PyImageSearch](https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/)
[^23]: [optimization - Fastest 2D convolution or image filter in Python - Stack ...](https://stackoverflow.com/questions/5710842/fastest-2d-convolution-or-image-filter-in-python)
[^24]: [Fast Fourier Convolution - NIPS papers](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)
[^25]: [Optical transfer function - Wikipedia](https://en.wikipedia.org/wiki/Optical_transfer_function)
[^26]: [Can FFT convolution be faster than direct convolution for signals of large sizes?](https://dsp.stackexchange.com/questions/71278/can-fft-convolution-be-faster-than-direct-convolution-for-signals-of-large-sizes)
[^27]: [FFT versus Direct Convolution | Spectral Audio Signal Processing - DSPRelated.com](https://www.dsprelated.com/freebooks/sasp/FFT_versus_Direct_Convolution.html)
[^28]: [The Scientist and Engineer's Guide to Digital Signal Processing FFT Convolution](https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_ch18.pdf)
[^29]: [2D Convolution in Python similar to Matlab's conv2 - Stack Overflow](https://stackoverflow.com/questions/16121269/2d-convolution-in-python-similar-to-matlabs-conv2)
[^30]: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>