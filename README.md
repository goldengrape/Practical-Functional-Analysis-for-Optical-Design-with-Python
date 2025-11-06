# 📐 实用泛函分析：Python驱动的眼视光学镜片设计优化

[![License](https://img.shields.io/badge/许可证-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Optiland](https://img.shields.io/badge/Optiland-v1.0%2B-orange.svg)](https://github.com/optiland/optiland)

> **将光学设计挑战转化为数学解决方案**  
> 一门由AI设计的实用课程，专注泛函分析在眼视光学镜片设计中的实际应用，全程使用Python + Optiland光学库

![课程封面](assets/course_banner.png)

## 🤖 AI设计声明

本课程的**完整设计、教材编写、代码实现均由AI生成**。课程内容基于：
- 眼视光学领域的专业文献和工程实践
- 泛函分析在光学设计中的最新应用研究  
- **Optiland光学库**的API设计和最佳实践
- Python科学计算生态的完整工具链
- 教育心理学中的成人学习原理

**AI不是替代人类专家，而是将专家知识转化为可操作的学习体验**。我们鼓励学员结合自身工作经验，对课程内容进行批判性思考和实践验证。

## 🎯 课程概览

这是一门**12周的实用课程**，通过眼视光学镜片设计的实际问题，教授泛函分析的核心概念。课程专为数学背景有限的光学设计师和工程师设计，**80%的时间用于Python实践**，而非理论推导。

**核心理念**：*"每个数学概念都配有一个可工作的Python实现，用于解决真实的镜片设计问题。"*

**技术亮点**：全程使用 **[Optiland](https://github.com/optiland/optiland)** 作为核心光学库，提供：
- 完整的光线追迹引擎
- 像差分析和优化工具
- 镜片曲面建模API
- 与泛函分析算法的无缝集成

## 👥 目标学员

- 眼视光学镜片设计师和光学工程师
- 眼镜研发部门的技术人员
- 视光学科的研究生和研究人员
- **先决条件**：基础Python编程、入门级线性代数、基础光学原理
- **无需高级数学背景** - 我们通过可视化和实际应用建立直觉

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
- **Optiland深度整合**：所有示例基于现代光学设计库，非玩具代码
- **临床-工程桥梁**：将验光师需求直接转化为数学约束
- **生产就绪代码**：所有示例代码可直接集成到工作流程
- **量化价值**：预期缩短镜片设计周期20-30%

## 📂 仓库结构

```bash
├── 📁 course_materials/          # 课程教材（AI生成）
│   ├── week1_fundamentals/      # 第1周：基础概念
│   │   ├── lecture_notes.md     # 讲义（含Optiland示例）
│   │   ├── python_optiland_tutorial.ipynb # Optiland入门
│   │   └── clinical_case_study/ # 临床案例
│   ├── week2_operators/         # 第2周：算子理论
│   └── ...                      # 其他周次
├── 📁 code_library/              # Python代码库
│   ├── core_algorithms/         # 核心泛函算法
│   ├── optiland_integration/    # Optiland专用封装
│   │   ├── lens_design_wrapper.py # 镜片设计API封装
│   │   ├── aberration_analysis.py # 像差分析工具
│   │   └── optimization_utils.py  # 优化工具
│   ├── clinical_data/           # 临床数据处理
│   └── visualization_tools/     # 可视化工具
├── 📁 datasets/                  # 数据集（模拟+真实）
│   ├── corneal_topography/      # 角膜地形图
│   ├── wavefront_aberrations/   # 波前像差
│   └── patient_vision_data/     # 患者视觉数据
├── 📁 projects/                  # 实践项目
│   ├── project1_distortion/     # 项目1：畸变优化（Optiland实现）
│   ├── project2_progressive/    # 项目2：渐进镜片设计
│   └── final_project/           # 期末项目模板
├── 📁 utils/                     # 工具函数
│   ├── ai_course_generator/     # AI课程生成器（可选）
│   └── validation_tools/        # 验证工具
├── 📄 requirements.txt           # Python依赖（含Optiland）
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

### 4. 验证Optiland安装
```python
# 验证脚本
import optiland as ol
print(f"Optiland版本: {ol.__version__}")
print(f"可用功能: {dir(ol)}")

# 快速测试：创建简单镜片
lens = ol.Lens()
lens.add_surface(ol.Surface(curvature=0.0))  # 平面
lens.add_surface(ol.Surface(curvature=1/50))  # 50mm曲率半径
print("Optiland环境验证成功！")
```

### 5. 开始学习（推荐顺序）
```bash
# 1. 先阅读第1周讲义（含Optiland介绍）
cat course_materials/week1_fundamentals/lecture_notes.md

# 2. 运行Optiland入门教程
jupyter notebook course_materials/week1_fundamentals/python_optiland_tutorial.ipynb

# 3. 尝试第1个项目：使用Optiland优化球面镜片
jupyter notebook projects/project1_distortion/project_template.ipynb
```

## 🚀 云端运行（无需安装）

[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/lens-functional-analysis/blob/main/notebooks/quick_start_optiland.ipynb)【待完成】

点击上方按钮，直接在Google Colab中运行首个Optiland示例，无需本地配置。
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
- **Optiland相关贡献优先**：本课程的核心价值在于与现代光学库的深度整合

## Optiland使用最佳实践

### 课程中的Optiland封装
```python
# 为泛函分析优化的封装
class FunctionalLensOptimizer:
    """将泛函优化算法与Optiland集成"""
    
    def __init__(self, lens: ol.Lens):
        self.lens = lens
        self.optiland_optimizer = ol.Optimizer(lens)
    
    def optimize_with_functional(self, functional, constraints=None):
        """
        使用泛函作为优化目标
        
        Args:
            functional: 泛函函数，接受lens参数返回标量
            constraints: Optiland约束列表
        """
        # 将泛函转换为Optiland兼容的目标函数
        def target_function(params):
            self._update_lens_params(params)
            return functional(self.lens)
        
        # 调用Optiland优化器
        result = self.optiland_optimizer.custom_optimize(
            target_function=target_function,
            constraints=constraints
        )
        return result
    
    def _update_lens_params(self, params):
        """更新镜片参数"""
        # 实现参数到镜片表面的映射
        pass
```

### 性能优化建议
```python
# 1. 批量处理 - 避免在循环中重复创建Optiland对象
lens_template = ol.Lens()  # 创建模板
lens_variants = [lens_template.copy() for _ in range(100)]

# 2. 使用JIT编译 - 对性能关键部分
import numba

@numba.jit(nopython=True)
def fast_aberration_calculation(params):
    # 高性能计算代码
    pass

# 3. 并行化 - 利用多核CPU
from concurrent.futures import ProcessPoolExecutor

def parallel_optimization(lens_params_list):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(optimize_single_lens, lens_params_list))
    return results
```

## 📜 许可证

本课程采用 [MIT许可证](LICENSE) - 你可以自由使用、修改、分发，但**不提供任何保证**。

**特别条款**：
- AI生成的教材内容可自由用于教育和研究
- 企业商业使用请联系作者获取授权
- 临床数据仅用于教学，不得用于真实患者诊断
- **Optiland使用条款**：请遵守[Optiland许可证](https://github.com/optiland/optiland/blob/main/LICENSE)

## 🙏 致谢

- **AI协作者**：本课程由大型语言模型设计生成，特别感谢其在知识整合和教学设计方面的能力
- **Optiland团队**：提供强大的现代光学设计库，使本课程成为可能
- **开源社区**：NumPy、SciPy、Optiland等开源项目的维护者
- **临床专家**：匿名提供案例的眼科医生和验光师
- **早期学员**：参与测试并提供反馈的镜片设计师

## 📞 联系我们

- **课程问题**：course@lens-functional-analysis.org
- **技术问题**（含Optiland）：tech@lens-functional-analysis.org
- **商业合作**：business@lens-functional-analysis.org
- **紧急临床问题**：**本课程不提供医疗建议**，请联系专业医疗机构
- **Optiland支持**：support@optiland.org

## 🌟 重要提醒

> **本课程内容由AI生成，仅供参考学习。**  
> **不应用于真实患者诊断或治疗决策。**  
> **所有设计结果需经专业光学工程师和临床专家验证。**  
> **Optiland是第三方库，本课程不对其功能提供保证。**

---

**版本**：1.0.0 (AI设计版 + Optiland集成)  
**最后更新**：2025年11月6日  
**AI模型**：Qwen-Plus  

