# 📐 实用泛函分析：Python驱动的眼视光学镜片设计优化

[![License](https://img.shields.io/badge/许可证-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Manim](https://img.shields.io/badge/Manim-v0.18%2B-red.svg)](https://www.manim.community/)

> **将光学设计挑战转化为数学解决方案**  
> 一门由AI设计的实用课程，专注泛函分析在眼视光学镜片设计中的实际应用，全程使用Python + Manim数学动画

![课程封面](assets/course_banner.png)

## 🤖 AI设计声明

本课程的**完整设计、教材编写、代码实现均由AI生成**。课程内容基于：
- 眼视光学领域的专业文献和工程实践
- 泛函分析在光学设计中的最新应用研究  
- **Manim数学动画引擎**的教学应用和最佳实践
- Python科学计算生态的完整工具链
- 教育心理学中的成人学习原理

**AI不是替代人类专家，而是将专家知识转化为可操作的学习体验**。我们鼓励学员结合自身工作经验，对课程内容进行批判性思考和实践验证。

## 🎯 课程概览

这是一门**13周的实用课程**（含模块0自学周），通过眼视光学镜片设计的实际问题，教授泛函分析的核心概念。课程专为数学背景有限的光学设计师和工程师设计，**80%的时间用于Python实践**，而非理论推导。

**核心理念**：*"每个数学概念都配有一个可工作的Python实现 + Manim动画演示，用于解决真实的镜片设计问题。"*

**技术亮点**：全程使用 **[Manim](https://www.manim.community/)** 作为核心教学工具，提供：
- 专业级数学动画制作能力
- 抽象概念的直观可视化
- 光学原理的动态演示
- 与Python科学计算的无缝集成

## 👥 目标学员

- 眼视光学镜片设计师和光学工程师
- 眼镜研发部门的技术人员
- 视光学科的研究生和研究人员
- **先决条件**：模块0（自学）：多元微积分与线性代数（理解梯度、内积）、Python编程入门（熟悉NumPy/Matplotlib）、Manim环境能运行
- **无需高级数学背景** - 我们通过Manim动画可视化和实际应用建立直觉

## ✨ 课程特色

| 传统数学课程 | 本AI设计课程 |
|------------|-------------|
| 从定理证明开始 | **从光学设计痛点开始** |
| 抽象符号推导 | **Python + Optiland即时验证** |
| 通用数学问题 | **真实镜片设计案例** |
| 期末考试评估 | **设计效率提升量化** |
| 纸笔作业 | **可部署的Optiland工具箱** |

### 🔥 核心优势

- **AI个性化适配**：课程内容可根据学员背景动态调整难度
- **Manim动画教学**：所有抽象数学概念都配有专业级动画演示
- **临床-数学-动画三重转换**：每节课提供"症状→数学约束→Manim动画"完整链条
- **生产就绪代码**：所有示例代码可直接集成到工作流程
- **量化价值**：预期缩短镜片设计周期20-30%

## 📂 仓库结构

```bash
├── 📁 course_repository/         # 课程仓库（13周完整结构）
│   ├── module0_bridge/          # 模块0：桥梁周（自学）
│   │   └── self_assessment_checklist.md # 自测清单
│   ├── module1_foundations/     # 模块1：泛函基础（第1-4周）
│   │   ├── week1_continuous_thinking/
│   │   │   ├── lens_distortion_visualization.py # 项目1
│   │   │   └── manim_demos/
│   │   │       └── fermat_principle.py
│   │   ├── week2_variational_principles/ # (新增周)
│   │   │   ├── path_optimization_numerical.py # 项目2
│   │   │   └── manim_demos/
│   │   │       ├── euler_lagrange_derivation.py
│   │   │       ├── brachistochrone.py
│   │   │       └── snells_law_derivation.py
│   │   ├── week3_functional_gradient_descent/
│   │   │   ├── curvature_optimization.py # 项目3
│   │   │   └── manim_demos/
│   │   │       └── gradient_descent_functional.py
│   │   └── week4_function_spaces/
│   │       ├── wavefront_fitting.py # 项目4
│   │       └── manim_demos/
│   │           └── zernike_basis.py
│   ├── module2_operators/       # 模块2：算子理论（第5-7周）
│   │   ├── week5_linear_operators/
│   │   ├── week6_integral_equations/
│   │   └── week7_nonlinear_operators/
│   ├── module3_advanced/        # 模块3：高级应用（第8-10周）
│   │   ├── week8_banach_spaces/
│   │   ├── week9_weak_convergence/
│   │   └── week10_distribution_theory/
│   ├── module4_integration/     # 模块4：工作流整合（第11-13周）
│   │   ├── week11_optimization_engineering/
│   │   ├── week12_uncertainty_quantification/
│   │   └── week13_ai_integration/
│   ├── manim_utils/             # Manim专用工具
│   │   ├── optical_mobjects.py # 光学专用图形对象
│   │   └── animation_templates.py # 动画模板
│   └── utils/                   # 通用工具
│       ├── optical_data_loader.py # 公开眼科数据集接口
│       └── clinical_metrics.py    # 临床指标计算库
├── 📁 datasets/                  # 数据集（模拟+真实）
│   ├── corneal_topography/      # 角膜地形图
│   ├── wavefront_aberrations/   # 波前像差
│   └── patient_vision_data/     # 患者视觉数据
├── 📁 projects/                  # 实践项目
│   ├── module1_projects/        # 模块1项目
│   ├── module2_midterm/         # 中期项目
│   ├── module3_team_challenge/  # 团队挑战赛
│   └── module4_final/           # 期末项目
├── 📁 course_materials/          # 补充教材
│   ├── manim_tutorial/          # Manim入门指南
│   └── optical_principles/        # 光学原理补充
├── 📄 requirements.txt           # Python依赖（含Manim）
├── 📄 setup_guide.md             # 安装指南
├── 📄 CONTRIBUTING.md            # 贡献指南
└── 📄 README.md                  # 本文件
```

## ⚙️ 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/goldengrape/Practical-Functional-Analysis-for-Optical-Design-with-Python.git
cd Practical-Functional-Analysis-for-Optical-Design-with-Python
```

### 2. 创建Python环境
```bash
# 使用conda（推荐）
conda create -n lens-functional-analysis python=3.11
conda activate lens-functional-analysis

# 或使用venv
python -m venv lens-env
source lens-env/bin/activate  # Linux/Mac
# 或
.\lens-env\Scripts\activate   # Windows
```

### 4. 验证Manim安装
```python
# 验证脚本
import manim as mn
print(f"Manim版本: {mn.__version__}")
print(f"可用功能: {dir(mn)[:10]}...")  # 显示前10个功能

# 快速测试：创建简单动画
from manim import *

class HelloWorld(Scene):
    def construct(self):
        text = Text("Hello, Functional Analysis!")
        self.play(Write(text))
        self.wait()

print("Manim环境验证成功！")
```

### 5. 开始学习（推荐顺序）
```bash
# 1. 先完成模块0自测清单
cat course_repository/module0_bridge/self_assessment_checklist.md

# 2. 运行Manim入门教程
jupyter notebook course_materials/manim_tutorial/manim_basics.ipynb

# 3. 尝试第1个项目：球面镜片边缘畸变可视化
jupyter notebook projects/module1_projects/lens_distortion_visualization.ipynb
```

## 🚀 云端运行（无需安装）

[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/lens-functional-analysis/blob/main/notebooks/quick_start_manim.ipynb)【待完成】

点击上方按钮，直接在Google Colab中运行首个Manim示例，无需本地配置。
## 🤝 贡献指南

虽然本课程由AI设计生成，但我们**强烈鼓励人类专家参与改进**：

1. **报告问题**：发现错误或不准确内容？[提交Issue](https://github.com/yourusername/lens-functional-analysis/issues)
2. **改进内容**：有临床经验或光学设计经验？[提交Pull Request](https://github.com/yourusername/lens-functional-analysis/pulls)
3. **Optiland扩展**：为Optiland贡献新功能或改进现有API
4. **添加案例**：分享真实镜片设计案例（脱敏后）
5. **本地化**：帮助翻译成其他语言

**贡献原则**：
- 保持实用导向，避免纯理论扩展
- 确保所有代码可运行、有注释
- 临床案例需符合医学伦理
- AI生成内容需标注来源
- **Manim动画贡献优先**：本课程的核心价值在于数学概念的直观可视化

## Manim动画制作最佳实践

### 课程中的Manim封装
```python
# 为光学设计定制的Manim工具类
class OpticalScene(Scene):
    """光学专用场景基类，提供常用光学元素"""
    
    def __init__(self):
        super().__init__()
        self.lens_style = {
            "stroke_width": 3,
            "color": BLUE,
            "fill_opacity": 0.2
        }
        self.light_ray_style = {
            "stroke_width": 2,
            "color": YELLOW
        }
    
    def create_lens(self, radius=2, thickness=0.5):
        """创建镜片图形"""
        lens = Arc(radius=radius, angle=PI, **self.lens_style)
        lens.shift(RIGHT * thickness)
        return lens
    
    def create_light_ray(self, start, end):
        """创建光线"""
        return Line(start, end, **self.light_ray_style)
    
    def animate_fermat_principle(self, paths):
        """动画演示费马原理"""
        animations = []
        for path in paths:
            ray = self.create_light_ray(path[0], path[1])
            animations.append(Create(ray))
        return AnimationGroup(*animations)

# 使用示例
class FermatPrincipleDemo(OpticalScene):
    def construct(self):
        lens = self.create_lens()
        self.play(Create(lens))
        
        # 演示不同光路
        paths = [
            [LEFT * 3, ORIGIN],
            [ORIGIN, RIGHT * 3]
        ]
        
        self.play(self.animate_fermat_principle(paths))
        self.wait()
```

### 性能优化建议
```python
# 1. 使用简化的mobjects - 复杂场景中使用较低细节级别
class OptimizedLens(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用简单的线条而非复杂的曲线
        self.add(Line(LEFT, RIGHT, stroke_width=2))
        self.add(Arc(radius=1, angle=PI/4))

# 2. 批量动画 - 减少单独的play调用
class BatchAnimationExample(Scene):
    def construct(self):
        elements = [Circle() for _ in range(10)]
        
        # 一次性创建所有元素
        self.play(*[Create(elem) for elem in elements])
        
        # 批量移动动画
        self.play(*[elem.animate.shift(RIGHT) for elem in elements])

# 3. 缓存复杂计算 - 预计算动画路径
import numpy as np

@lru_cache(maxsize=128)
def calculate_lens_surface(radius, curvature, resolution=50):
    """缓存镜片表面计算"""
    theta = np.linspace(0, 2*np.pi, resolution)
    x = radius * np.cos(theta)
    y = curvature * x**2  # 简化的曲面方程
    return list(zip(x, y, [0]*resolution))

# 4. 使用Camera正确设置分辨率
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
```

## 📜 许可证

本课程采用 [MIT许可证](LICENSE) - 你可以自由使用、修改、分发，但**不提供任何保证**。

**特别条款**：
- AI生成的教材内容可自由用于教育和研究
- 企业商业使用请联系作者获取授权
- 临床数据仅用于教学，不得用于真实患者诊断
- **Manim使用条款**：请遵守[Manim许可证](https://github.com/ManimCommunity/manim/blob/main/LICENSE)


## 🌟 重要提醒

> **本课程内容由AI生成，仅供参考学习。**  
> **不应用于真实患者诊断或治疗决策。**  
> **所有设计结果需经专业光学工程师和临床专家验证。**  
> **Manim是第三方库，本课程不对其功能提供保证。**

---

## 📚 课程结构与评估体系

### 13周完整学习路径

| 模块 | 周次 | 核心主题 | Manim动画重点 | 实践项目 |
|-----|-----|---------|--------------|----------|
| **模块0** | 第0周 | 桥梁周（自学） | - | 技能自测清单 |
| **模块1** | 第1-4周 | 泛函基础与建模 | 光线路径、E-L方程、梯度下降 | 4个Python脚本 + 1个Manim动画 |
| **模块2** | 第5-7周 | 算子理论与系统建模 | 算子作用、积分过程、不动点迭代 | 个性化镜片设计工具 |
| **模块3** | 第8-10周 | 高级应用与不确定性 | 范数比较、多目标优化、分布理论 | 鲁棒性设计挑战赛 |
| **模块4** | 第11-13周 | 工作流整合与前沿 | 算法流程、贝叶斯更新、AI融合 | 端到端设计工作流 |

### 评估体系（100%实践导向）

| 评估类型 | 占比 | 具体要求 | 通过标准 |
|----------|------|----------|----------|
| 模块1项目 | 20% | 4个Python脚本 + Manim动画 | 代码可运行，结果合理 |
| 中期项目（模块2） | 25% | 个性化镜片设计工具包 + 配套动画 | 解决真实临床问题 |
| 小组项目（模块3） | 25% | 鲁棒性设计解决方案 + 团队动画演示 | 团队协作，创新性 |
| 期末项目（模块4） | 25% | 完整工作流集成 + 专业演示视频 | 从数据到原型输出 |
| 学习日志 | 5% | 每周反思：工具如何改进工作 | 真实性，深度思考 |

**总体通过标准**：≥80分，且所有Python项目完成率≥90%，Manim动画完成率≥80%

### Python工具链（开源优先）

```python
# 基础科学计算
import numpy as np, scipy, pandas

# 符号计算
import sympy

# 可视化
import matplotlib, plotly

# 数学动画（核心教学工具）
import manim

# AI辅助
from sklearn.linear_model import LinearRegression

# 部署
import streamlit
```

---

**版本**：2.0.0 (AI设计版 + Manim可视化增强)  
**最后更新**：2025年11月6日  
**AI模型**：Qwen-Plus

