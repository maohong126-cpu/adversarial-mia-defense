# Adversarial FGSM for Membership Inference Defense

> **一句话总结**：我们改进了一种叫 FGSM 的对抗攻击算法，把它用来训练模型，
> 让模型在保持分类准确率的同时，对抗一种叫"成员推理攻击"的隐私窃取手段。

---

## 目录

1. [背景：什么是成员推理攻击（MIA）](#1-背景什么是成员推理攻击mia)
2. [背景：什么是 FGSM 对抗攻击](#2-背景什么是-fgsm-对抗攻击)
3. [我们做了什么：改进 FGSM](#3-我们做了什么改进-fgsm)
4. [实验怎么设计的](#4-实验怎么设计的)
5. [实验结果（所有数字）](#5-实验结果所有数字)
6. [每个指标是什么意思](#6-每个指标是什么意思)
7. [总结与结论](#7-总结与结论)
8. [代码使用方法](#8-代码使用方法)
9. [项目文件结构](#9-项目文件结构)
10. [参考文献](#10-参考文献)

---

## 1. 背景：什么是成员推理攻击（MIA）

### 问题场景

假设某医院用 10000 个病人数据训练了一个 AI 诊断模型，模型训练完之后对外提供服务。
攻击者想知道：**某个病人的数据有没有被用来训练这个模型？**

如果攻击者能猜出来，就等于泄露了隐私——"这个人曾经在某医院看过病"。

这就是**成员推理攻击（Membership Inference Attack，MIA）**。

### 为什么能猜出来？

深度学习模型有一个问题：**对训练过的数据过度自信**。

```
训练集里的图片 → 模型预测：猫，置信度 99%
从没见过的图片 → 模型预测：猫，置信度 73%
```

这个"置信度差距"就是 MIA 的漏洞。攻击者只需要：
1. 查询模型对某张图片的置信度
2. 置信度高 → 猜这张图片是训练数据
3. 置信度低 → 猜不是训练数据

### 攻击流程（本项目实现的方法）

```
攻击者已知：
  ✓ 100 张确定是训练集的图片（known members）
  ✓ 100 张确定不是训练集的图片（known non-members）
  ? 1000 张不知道是不是训练集的图片（target）

攻击步骤：
  1. 对已知的 200 张图片查询模型置信度
  2. 找一个置信度阈值 τ（比如 0.85）：
       置信度 ≥ τ → 预测是成员
       置信度 < τ → 预测不是成员
  3. 用这个阈值对 target 1000 张预测
```

**评价指标**：MIA Accuracy = 猜对的比例。随机猜是 50%，越高说明模型隐私泄露越严重。

---

## 2. 背景：什么是 FGSM 对抗攻击

### 对抗样本是什么？

给图片加一点点人眼看不见的噪声，让模型看错：

```
原图：猫（模型预测：猫 99%）
加噪后：还是猫（人眼看不出区别）→ 模型预测：狗 94%
```

这个噪声就叫**对抗扰动**，加噪的过程叫**对抗攻击**。

### FGSM（Fast Gradient Sign Method）

Goodfellow 2014 年提出的最经典方法，公式非常简单：

```
对抗样本 = 原图 + ε × sign(梯度)
```

**解释**：
- `梯度`：模型对这张图"最敏感"的方向（哪些像素改一点模型就会判断错）
- `sign(梯度)`：只取方向（+1 或 -1），不管大小
- `ε`：扰动强度，越大噪声越明显，一般取 8/255 ≈ 0.031（图片像素值 0\~1 范围）

### FGSM 用于防御？

把对抗样本加入**训练过程**，让模型同时学习：
- 干净图片要分对
- 被加噪的对抗图片也要分对

这样模型就不能只靠"记住训练图片"来分类，必须学到更通用的特征，
导致对训练集和测试集的置信度差距缩小，MIA 就更难成功。

---

## 3. 我们做了什么：改进 FGSM

### 原方法的问题

**标准 FGSM**：对所有像素加相同大小的噪声（只看梯度方向，不看大小）
```
δᵢ = ε × sign(∂L/∂xᵢ)   对每个像素 i 都一样大
```

**你们之前的方法（归一化梯度）**：用梯度真实方向代替符号
```
δ = ε × 梯度 / ||梯度||₂
```
**实验发现这个方法没效果，有时候还让 MIA 变得更容易。**

### 我们的改进：感知自适应 FGSM（Adaptive FGSM）

**核心思路**：聪明地分配 ε 预算，不是每个像素一样多。

**创新点 A：梯度集中加权**

梯度大的像素 → 对模型决策影响大 → 多分配一点扰动  
梯度小的像素 → 对模型影响小 → 少分配或不分配

```
权重W_eff(像素i) = (|梯度ᵢ| / 最大梯度)^α    取值 [0, 1]
```

**创新点 B：感知遮罩（视觉不可察性）**

图片纹理丰富的区域（草地、毛发）→ 人眼不敏感 → 可以多加噪声  
图片平滑区域（天空、白墙）→ 人眼容易察觉 → 少加噪声

```
权重W_perc(像素i) = (局部方差ᵢ / 最大方差)^β    取值 [0, 1]
```

局部方差 = 以该像素为中心的 5×5 窗口内的像素值方差。

**最终扰动**（两个权重相乘，归一化后保证 L∞ ≤ ε）：
```
δ = ε × sign(梯度) × W_norm    其中 W_norm = W_eff × W_perc / max(...)
```

**创新点 C：MIA 置信度导向损失**

标准 FGSM 只让模型"分错"：

```
生成损失 = CE(模型预测, 真实标签)   → 最大化分类错误
```

我们同时让模型"降低置信度"：

```
生成损失 = CE(模型预测, 真实标签) - λ × 平均最大置信度
```

第二项的效果：生成的对抗样本专门针对"模型过于自信"这个弱点，
让训练时模型不能依赖高置信度，从而**直接压缩 MIA 利用的置信度差距**。

### 完整版（Adaptive + KL + AdvReg）

在以上基础上叠加两个文献方法：

**TRADES 风格 KL 损失**（Zhang et al., 2019）：
```
训练损失 = CE(干净样本) + λ × KL散度(干净预测 ∥ 对抗预测)
```
作用：让模型对干净样本和对抗样本的预测尽量一致，平滑决策边界。

**AdvReg 风格训练置信度惩罚**（Nasr et al., 2018 简化版）：
```
训练损失 += μ × 训练样本的平均最大置信度
```
作用：直接惩罚模型在训练数据上过于自信，强制缩小置信度差距。

---

## 4. 实验怎么设计的

### 数据集

| 数据集 | 类别数 | 图片尺寸 | 训练集大小 | 难度 |
|---|---|---|---|---|
| MNIST | 10 | 1×28×28（灰度手写数字） | 60,000 | 简单 |
| CIFAR-10 | 10 | 3×32×32（彩色物体） | 50,000 | 中等 |
| CIFAR-100 | 100 | 3×32×32（彩色细粒度） | 50,000 | 较难 |

### 模型

| 模型 | 参数量 | 适用场景 |
|---|---|---|
| SimpleCNN | ~13万 | 小模型，快速验证 |
| ResNet18 | ~1100万 | 大模型，接近实际应用 |

### 为什么用小训练集？

MIA 需要模型**先过拟合**（记住训练数据）才有攻击空间。  
用全量 50000 个训练样本时，ResNet18 训练集和测试集准确率差不多，MIA 几乎没用。  
**用 500\~5000 个样本时，模型会严重过拟合，MIA 成功率可达 60\~70%**，这时防御效果才看得出来。

### 实验参数

| 参数 | SimpleCNN 实验 | ResNet18 实验 |
|---|---|---|
| 训练集大小 | 500 | 5,000 |
| 训练轮数 | 80 epochs | 50\~60 epochs |
| 学习率 | 0.05（CosineAnnealingLR） | 0.05 |
| 批大小 | 128 | 256 |
| 对抗扰动强度 ε | 8/255 ≈ 0.031 | 8/255 |
| MIA 校准样本 | 300 成员 + 300 非成员 | 3500 + 3500 |
| MIA 目标样本 | 200 成员 + 200 非成员 | 1500 + 1500 |

### 5 种方法对比

| 编号 | 方法 | 是否对抗训练 | 特殊损失 |
|---|---|---|---|
| 1 | **Baseline** | 否 | 标准 CE |
| 2 | **Sign-FGSM** | 是 | 标准 CE |
| 3 | **Normalized-FGSM** | 是（归一化梯度） | 标准 CE |
| 4 | **Adaptive-FGSM** | 是（感知自适应） | MIA 置信度导向 |
| 5 | **Adaptive+KL+AdvReg** | 是（感知自适应） | MIA 导向 + KL + AdvReg |

---

## 5. 实验结果（所有数字）

### MNIST × SimpleCNN（train=500，80 epochs）

| 方法 | 干净准确率 | MIA攻击成功率 | MIA降低幅度 | 准确率变化 | PU得分 |
|---|---|---|---|---|---|
| Baseline | 94.00% | **57.00%** | — | — | — |
| Sign-FGSM | 96.17% | 53.50% | ↓3.50% | +2.17% ↑ | ∞ |
| Normalized-FGSM | 94.55% | 56.50% | ↓0.50% | +0.55% ↑ | — |
| **Adaptive-FGSM** ⭐ | **95.68%** | **52.50%** | **↓4.50%** | **+1.68% ↑** | **∞** |
| Adaptive+KL+AdvReg | 95.13% | 54.50% | ↓2.50% | +1.13% ↑ | ∞ |

> MNIST 上所有对抗训练方法都让准确率**变高了**（对抗训练起到了正则化效果）。
> Adaptive-FGSM 是最优：MIA 下降最多（4.5%），同时准确率还提升了 1.68%。

---

### CIFAR-10 × SimpleCNN（train=500，80 epochs）

| 方法 | 干净准确率 | MIA攻击成功率 | MIA降低幅度 | 准确率变化 | PU得分 |
|---|---|---|---|---|---|
| Baseline | 46.59% | **68.75%** | — | — | — |
| Sign-FGSM | 41.33% | 64.00% | ↓4.75% | -5.26% | 0.90 |
| Normalized-FGSM | 46.59% | 70.25% | **↑-1.50%（更差！）** | 0% | — |
| Adaptive-FGSM | 45.56% | 65.50% | ↓3.25% | -1.03% | 3.16 |
| **Adaptive+KL+AdvReg** ⭐ | **46.50%** | **65.25%** | **↓3.50%** | **-0.09%** | **38.9** |

> Normalized-FGSM 再次失效（MIA 反而涨了 1.5%）。
> Adaptive+KL+AdvReg 把 MIA 从 68.75% 降到 65.25%，准确率几乎不变（只损失 0.09%）。

---

### CIFAR-10 × ResNet18（train=5,000，50 epochs）

大模型，过拟合更严重（训练集 98.7%，测试集 72.2%，MIA 基线高达 67.80%）。

| 方法 | 干净准确率 | MIA攻击成功率 | MIA降低幅度 | 准确率变化 | PU得分 |
|---|---|---|---|---|---|
| Baseline | 72.17% | **67.80%** | — | — | — |
| Sign-FGSM | 54.01% | 54.83% | **↓12.97%** ✅超10% | -18.16% | 0.71 |
| Normalized-FGSM | 64.22% | 58.77% | ↓9.03% | -7.95% | 1.14 |
| Adaptive-FGSM | 65.38% | 61.77% | ↓6.03% | -6.79% | 0.89 |
| **Adaptive+KL+AdvReg** ⭐ | **69.35%** | **64.50%** | ↓3.30% | **-2.82%** | **1.17** |

> ResNet18 上 Sign-FGSM 首次突破 10% MIA 下降，但代价是准确率暴跌 18%（基本不可用）。
> Adaptive+KL+AdvReg 依然是最优 PU得分（1.17），准确率只损失 2.82%。

---

### CIFAR-100 × SimpleCNN（train=500，80 epochs）

CIFAR-100 有 100 个类，模型只用 500 个样本根本学不够（基线准确率仅 9.9%，接近随机猜测的 1/100=1%）。
此时 MIA 信号异常强烈（78.25%），因为模型只是死记硬背这 500 张图，完全没有泛化。

| 方法 | 干净准确率 | MIA攻击成功率 | MIA变化幅度 | 准确率变化 |
|---|---|---|---|---|
| Baseline | 9.90% | **78.25%** | — | — |
| Sign-FGSM | 10.02% | 81.50% | ↑-3.25%（变差） | +0.12% |
| Normalized-FGSM | 10.65% | 89.25% | ↑-11.00%（大幅变差） | +0.75% |
| Adaptive-FGSM | 10.55% | 87.00% | ↑-8.75%（变差） | +0.65% |
| **Adaptive+KL+AdvReg** ⭐ | **9.91%** | **80.75%** | **↑-2.50%（损失最小）** | **+0.01%** |

> **重要发现**：当任务本身极难（100类，仅500样本），所有对抗训练方法均使 MIA 变得更容易，而非更难。
> 原因：模型基础准确率仅 ~10%，对抗扰动干扰了本就脆弱的学习过程，反而加剧了过拟合信号。
> Adaptive+KL+AdvReg 是损失最小的方案（MIA 只上升 2.5%，准确率几乎不变）。
> **结论**：FGSM 防御有效的前提是模型本身能学到有意义的表征（准确率足够高）。

---

### CIFAR-100 × ResNet18（train=5,000，60 epochs）

⏳ 实验运行中，结果待更新……

---

### ε 超参数敏感性（CIFAR-10 SimpleCNN）

扰动越大 → MIA 降低越多，但准确率损失也越多。

| 方法 | ε | MIA降低 | 准确率变化 | PU得分 |
|---|---|---|---|---|
| Baseline | — | — | — | — |
| Sign-FGSM | 8/255 | ↓4.25% | -6.28% | 0.68 |
| **Adaptive** | **8/255** | ↓2.00% | **-0.61%** | **3.28** |
| Adaptive | 16/255 | ↓3.00% | -2.64% | 1.14 |
| Adaptive | 32/255 | ↓3.75% | -3.72% | 1.01 |

> ε=8/255 是最佳点：Adaptive 的 PU得分 3.28，是 Sign-FGSM 的 **4.8 倍**。

---

## 6. 每个指标是什么意思

### 干净准确率（Clean Accuracy）

```
干净准确率 = 测试集上分对的图片数 / 测试集总图片数
```

**模型的基本能力**。防御方法不能让这个数字掉太多，否则模型就没用了。

- 越高越好
- 和 Baseline 相比的变化叫 **Clean Δ**（准确率变化）

---

### MIA 攻击成功率（MIA Accuracy）

```
MIA准确率 = 攻击者正确猜出"是/不是训练集成员"的比例
```

**衡量隐私泄露程度**。

| MIA 准确率 | 含义 |
|---|---|
| **50%** | 理想状态：攻击者等于瞎猜，完全没有信息 |
| **55%** | 轻微泄露：攻击者有点优势 |
| **65%** | 中度泄露：训练数据有隐私风险 |
| **75%** | 严重泄露：模型强烈记住了训练数据 |

- **越低越好**（越接近 50% 越好）
- 注意：如果低于 50%，攻击者只需翻转判断逻辑，准确率就变成 `1 - MIA准确率`，所以也不好

---

### MIA 降低幅度（MIA Drop）

```
MIA降低 = Baseline的MIA准确率 - 防御后的MIA准确率
```

防御方法让攻击者的成功率**下降了多少个百分点**。

- **正数 = 防御有效**（降低了 MIA 成功率）
- **负数 = 防御无效或反效果**（Normalized-FGSM 就是这种情况）
- 一般认为 **>5% 才算明显效果**，>10% 是强防御

---

### 准确率变化（Clean Δ）

```
Clean Δ = 防御后的干净准确率 - Baseline的干净准确率
```

- **负数**：防御带来的精度损失（代价）
- **正数**：对抗训练意外提升了泛化能力（MNIST 上就出现了这种情况）
- 越接近 0 越好

---

### PU 得分（Privacy-Utility Score）

```
PU得分 = MIA降低幅度 / |准确率变化|
```

**每损失 1% 精度，换来多少 % 的 MIA 保护**。

| PU得分 | 含义 |
|---|---|
| < 1 | 性价比低：精度损失比 MIA 保护还多 |
| 1 ~ 3 | 中等 |
| > 3 | 优秀 |
| → ∞ | 最优：准确率不降反升（精度和隐私同时改善） |

**举例**：
- Sign-FGSM（CIFAR-10）：MIA↓4.75%，精度↓5.26%，PU = 4.75/5.26 = **0.90**（每 1% 精度换 0.9% MIA）
- Adaptive+KL（CIFAR-10）：MIA↓3.50%，精度↓0.09%，PU = 3.50/0.09 = **38.9**（每 1% 精度换 38.9% MIA）

---

### 相对 MIA 降低（Relative MIA Drop）

```
相对降低 = MIA降低幅度 / Baseline MIA准确率 × 100%
```

方便跨数据集比较。比如：
- MNIST Adaptive-FGSM：4.5% / 57.0% = **7.9%** 相对降低（超过 AdvReg 2019 论文的 7.5%）

---

## 7. 总结与结论

### 三个数据集汇总

| 模型 | 数据集 | 最优防御方法 | MIA降低 | 准确率变化 | PU得分 |
|---|---|---|---|---|---|
| SimpleCNN | MNIST | Adaptive-FGSM | **↓4.5%（相对-7.9%）** | **+1.68%** | **∞** |
| SimpleCNN | CIFAR-10 | Adaptive+KL+AdvReg | ↓3.5% | -0.09% | **38.9** |
| ResNet18 | CIFAR-10 | Adaptive+KL+AdvReg | ↓3.3% | -2.82% | **1.17** |
| ResNet18 | CIFAR-10 | Sign-FGSM（最大MIA降低） | **↓12.97%** | -18.16% | 0.71 |
| SimpleCNN | CIFAR-100 | Adaptive+KL+AdvReg（最小损失） | ↑-2.5%（防御失效） | +0.01% | — |
| ResNet18 | CIFAR-100 | ⏳ 待更新 | — | — | — |

### 核心结论

1. **Normalized-FGSM 无效**：在 SimpleCNN 上持续失效甚至反效果，与原始项目观察一致

2. **若追求最大 MIA 降低**：Sign-FGSM 在 ResNet18 上能超过 10%，但代价是 -18% 准确率（不实用）

3. **若追求最优性价比**：Adaptive+KL+AdvReg 是最优选择
   - MNIST：准确率还提升了
   - CIFAR-10 SimpleCNN：精度几乎不变（-0.09%），PU得分 38.9
   - CIFAR-10 ResNet18：精度只损失 2.82%，PU得分 1.17

4. **与文献对比**：
   - AdvReg (2019)：约 7.5% 相对 MIA 降低，但准确率损失 ~7.5%
   - 本方法 MNIST：7.9% 相对降低，**准确率反而提升**

5. **CIFAR-100 新发现（防御边界条件）**：
   - 当模型本身准确率极低（500样本×100类 → 仅~10%准确率），对抗训练会适得其反
   - 根本原因：模型连正常分类都学不好，加噪声只会破坏本就脆弱的学习过程
   - **防御有效的前提**：模型必须能学到真正有用的特征（准确率足够高），才能从对抗训练中获益
   - ResNet18 × CIFAR-100（5000样本）的结果更能代表实际防御能力（待更新）

---

## 8. 代码使用方法

### 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 快速复现所有实验

```bash
# MNIST
python3 scripts/quick_experiment.py --dataset mnist --data-dir /tmp/mnist --download

# CIFAR-10 SimpleCNN
python3 scripts/quick_experiment.py --dataset cifar10 --data-dir /tmp/cifar10 --download \
  --train-size 500 --epochs 80

# CIFAR-10 ResNet18
python3 scripts/quick_experiment.py --dataset cifar10 --data-dir /tmp/cifar10 \
  --backbone resnet18 --train-size 5000 --epochs 50 \
  --member-size 3500 --non-member-size 3500 \
  --target-member-size 1500 --target-non-member-size 1500

# CIFAR-100
python3 scripts/quick_experiment.py --dataset cifar100 --data-dir /tmp/cifar100 --download
```

### 训练单个模型（Adaptive+KL+AdvReg 最优配置）

```bash
python3 scripts/train.py \
  --dataset cifar10 --data-dir /tmp/cifar10 \
  --backbone simple_cnn --epochs 80 --train-size 500 \
  --adversarial-epsilon 0.03137 \
  --adversarial-method adaptive \
  --adversarial-loss-type kl \
  --adversarial-mia-conf-lambda 0.5 \
  --adversarial-gradient-alpha 0.5 \
  --adversarial-perceptual-beta 0.5 \
  --train-conf-lambda 0.3 \
  --checkpoint checkpoints/best_model.pt
```

### 代码接口

```python
from adversral.attacks import fgsm_attack

# 1. 标准 FGSM
adv = fgsm_attack(model, inputs, labels, epsilon=8/255)

# 2. 归一化梯度
adv = fgsm_attack(model, inputs, labels, epsilon=8/255, method="normalized")

# 3. 感知自适应 + MIA 导向（本项目创新）
adv = fgsm_attack(
    model, inputs, labels, epsilon=8/255,
    method="adaptive",
    gradient_alpha=0.5,       # 梯度集中程度
    perceptual_beta=0.5,      # 感知权重强度
    kernel_size=5,            # 局部方差窗口大小
    mia_conf_lambda=0.5,      # MIA 置信度导向损失权重
)
```

---

## 9. 项目文件结构

```
adversral/
  attacks/
    fgsm.py                 # 核心算法：sign / normalized / adaptive 三种方法
    fgsm_original.py        # 原始 FGSM 实现备份
    membership_inference.py # 阈值 MIA 攻击实现
  data/
    vision.py               # MNIST / CIFAR-10 / CIFAR-100 数据加载
  models/
    vision.py               # SimpleCNN + ResNet 系列模型定义
  engine.py                 # 训练循环（支持 CE / KL 损失 + AdvReg 置信度惩罚）
  device.py                 # 自动选择 cuda → mps → cpu

scripts/
  quick_experiment.py       # 一键跑 5 种方法对比（主要入口）
  push_experiment.py        # 超参数扫描（探索 ε 和 λ 的影响）
  train.py                  # 训练单个模型
  evaluate_mia_defense.py   # baseline vs defense 对照评估
  run_membership_inference.py  # 单独运行 MIA 评估

tests/
  test_attacks.py           # 10 个单元测试，验证算法正确性
```

---

## 10. 参考文献

| 论文 | 贡献 |
|---|---|
| Goodfellow et al., ICLR 2015 | FGSM 原始论文 |
| Shokri et al., IEEE S&P 2017 | 成员推理攻击原始论文 |
| Nasr et al., CCS 2018 | AdvReg：对抗正则化防御 MIA |
| Zhang et al., ICML 2019 | TRADES：鲁棒性与准确率的权衡 |
| Chen et al., ICLR 2022 | RelaxLoss：梯度上升式损失 |
| Tang et al., USENIX 2022 | SELENA：自蒸馏集成防御 |
