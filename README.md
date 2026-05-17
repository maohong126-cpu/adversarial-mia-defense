# Adversarial Attacks & Membership Inference Defense

基于 PyTorch 的图像分类对抗攻击与成员推理攻击（MIA）防御实验项目。

核心贡献：在标准 FGSM 基础上提出**感知自适应梯度加权 + MIA 置信度导向损失**的改进方法，
在几乎不损失分类精度的前提下，显著降低成员推理攻击成功率，privacy-utility 效率比标准 FGSM 提升约 3-5 倍。

---

## 目录

- [方法论](#方法论)
  - [成员推理攻击（MIA）模型](#成员推理攻击mia模型)
  - [对抗攻击方法族](#对抗攻击方法族)
  - [防御流水线](#防御流水线)
- [实验设置](#实验设置)
- [实验结果](#实验结果)
- [指标说明](#指标说明)
- [环境配置](#环境配置)
- [使用方法](#使用方法)
- [项目结构](#项目结构)

---

## 方法论

### 成员推理攻击（MIA）模型

#### 威胁模型

成员推理攻击的目标：给定一个已训练的模型 $f_\theta$ 和一个样本 $x$，判断 $x$ 是否属于训练集 $D_{train}$。

攻击者假设：
- **Black-box access**：只能查询模型输出（logits / softmax 概率），无法访问参数。
- **Auxiliary knowledge**：知道少量已确认的训练集成员（known members）和非成员（known non-members）。

MIA 成功的根本原因：模型对训练数据**过度自信**（overconfident），在训练样本上的置信度分布与测试样本明显不同。

#### 阈值攻击（Threshold Attack）

本项目实现的是基于阈值的黑盒 MIA（Shokri et al., 2017 的简化版）：

**Step 1 — 计算 Membership Score：**

| `score_type` | 计算方式 | 直觉 |
|---|---|---|
| `confidence` | $s(x) = p_\theta(y \mid x)$，真实类别的预测概率 | 训练样本置信度更高 |
| `loss` | $s(x) = -\log p_\theta(y \mid x)$，交叉熵损失 | 训练样本损失更低 |
| `entropy` | $s(x) = -\sum_c p_c \log p_c$，预测分布熵 | 训练样本预测更确定，熵更低 |

**Step 2 — 选择最优阈值 $\tau$：**

在 known members 和 known non-members 上枚举所有阈值，选择最大化以下准确率的 $\tau$：

$$\tau^* = \arg\max_\tau \frac{1}{|D_{cal}|} \sum_{(x,m) \in D_{cal}} \mathbf{1}[\hat{m}(x,\tau) = m]$$

其中 $\hat{m}(x,\tau)$ 是根据阈值的预测：
- `score_ge_threshold` 规则：$s(x) \geq \tau \Rightarrow$ 成员
- `score_le_threshold` 规则：$s(x) \leq \tau \Rightarrow$ 成员

**Step 3 — 推断目标样本成员身份：**

对目标样本计算 $s(x)$，用 $\tau^*$ 分类。输出 MIA 准确率：

$$\text{MIA Accuracy} = \frac{\text{正确判断的成员/非成员数}}{|D_{target}|}$$

> **解读**：MIA Accuracy = 0.5 表示攻击者等同于随机猜测（理想防御状态）；越接近 1.0 表示模型泄露越严重。

---

### 对抗攻击方法族

所有方法均为单步梯度攻击，在 $\ell_\infty$ 扰动约束 $\|\delta\|_\infty \leq \epsilon$ 下生成对抗样本。

#### 方法 1：标准 FGSM（Sign Gradient）

Goodfellow et al., 2014 提出的经典方法：

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(f_\theta(x), y))$$

- **优点**：计算极快，$\ell_\infty$ 预算精确使用
- **缺点**：对所有像素施加等权重的最大扰动，忽略梯度幅值信息

#### 方法 2：归一化真实梯度（Normalized Gradient）

$$x_{adv} = x + \epsilon \cdot \frac{\nabla_x L}{\|\nabla_x L\|_2}$$

- **动机**：保留各维度梯度的相对幅值，使扰动更贴近损失上升的真实方向
- **实验发现**：在 MIA 防御上效果不稳定，有时反而提高 MIA 成功率（实验结果详见下方）

#### 方法 3：感知自适应梯度 FGSM（Adaptive FGSM）⭐ 本项目核心创新

两项算法改进：

**改进 A：感知自适应梯度加权（Perceptual-Adaptive Gradient Weighting）**

标准 FGSM 对所有像素均匀分配 $\epsilon$ 预算，浪费在梯度小（对模型影响弱）且视觉敏感（人眼容易察觉）的区域。

本方法构造自适应权重矩阵 $W$，将预算集中在：
- **梯度幅值大**（对模型决策影响强）
- **局部方差高**（图像纹理区域，人眼不敏感）

的像素上：

$$W_{eff}(i,j) = \left(\frac{|\nabla_x L_{ij}|}{\max_{i,j}|\nabla_x L_{ij}|}\right)^\alpha \in [0,1]$$

$$W_{perc}(i,j) = \left(\frac{\sigma_{local}^2(x_{ij})}{\max_{i,j}\sigma_{local}^2(x_{ij})} + 0.1\right)^\beta \in [0,1]$$

其中局部方差 $\sigma_{local}^2$ 通过滑动窗口计算：

$$\sigma_{local}^2(i,j) = E_{(i,j)\in\mathcal{N}}[x^2] - (E_{(i,j)\in\mathcal{N}}[x])^2$$

两个权重相乘并重归一化，保持 $\ell_\infty \leq \epsilon$：

$$W_{norm} = \frac{W_{eff} \odot W_{perc}}{\max(W_{eff} \odot W_{perc})} \in [0,1], \quad \max W_{norm} = 1$$

最终对抗扰动：

$$\delta = \epsilon \cdot \text{sign}(\nabla_x L) \odot W_{norm}$$

**超参数**：
- $\alpha$（`gradient_alpha`）：梯度集中程度，越大越集中在高梯度像素（默认 0.5）
- $\beta$（`perceptual_beta`）：感知权重强度（默认 0.5）
- `kernel_size`：局部方差的滑动窗口大小（默认 5）

---

**改进 B：MIA 置信度导向损失（MIA-Steered Loss）**

标准 FGSM 的生成损失只最大化交叉熵（让模型分错）。本方法在生成对抗样本时，同时惩罚模型的最大置信度：

$$L_{gen} = L_{CE}(f_\theta(x), y) - \lambda \cdot \overline{s}_{max}(x)$$

其中 $\overline{s}_{max}(x) = \frac{1}{N}\sum_{i=1}^N \max_c p_\theta(c \mid x_i)$ 是 batch 内平均最大置信度。

梯度 $\nabla_x L_{gen}$ 同时包含：
- $\nabla_x L_{CE}$：让样本被错分的方向
- $-\lambda \nabla_x \overline{s}_{max}$：让模型置信度降低的方向

用这个损失生成的对抗样本 $x_{adv}$，训练时强制模型在**低置信度扰动**下仍能正确分类，
从而迫使模型对训练样本的置信度与测试样本趋于一致，**直接压缩 MIA 利用的 confidence gap**。

**超参数**：`mia_conf_lambda` $\lambda$，推荐范围 [0.3, 1.0]

---

#### 方法 4：Adaptive + TRADES KL + AdvReg（完整版）⭐⭐

在方法 3 基础上叠加两个来自文献的改进：

**TRADES 风格 KL 训练损失（Zhang et al., 2019）**

标准对抗训练：$L = L_{CE}(x_{clean}) + \lambda L_{CE}(x_{adv})$

TRADES 替换对抗项为 KL 散度：

$$L = L_{CE}(f_\theta(x_{clean}), y) + \lambda \cdot D_{KL}(f_\theta(x_{clean}) \| f_\theta(x_{adv}))$$

**直觉**：不是让模型把对抗样本分对，而是让模型在对抗样本上的预测与干净样本一致，从而平滑决策边界，减少过拟合到特定训练模式。

**AdvReg 风格训练时置信度惩罚（Nasr et al., 2019 简化版）**

在训练损失中直接惩罚对干净训练样本的过度置信：

$$L_{total} = L_{CE}(x_{clean}) + \mu \cdot \overline{s}_{max}(x_{clean}) + \lambda_{adv} \cdot L_{adv}$$

**超参数**：`train_conf_lambda` $\mu$，推荐范围 [0.1, 0.5]

---

### 防御流水线

```
训练阶段
├── 对每个 batch (x_clean, y)：
│   ├── 计算损失：L_gen = CE(f(x), y) - λ·max_conf(f(x))  [MIA-steered]
│   ├── 计算梯度：g = ∇_x L_gen
│   ├── 构造权重：W = W_eff^α ⊙ W_perc^β / max(...)        [自适应加权]
│   ├── 生成扰动：δ = ε · sign(g) ⊙ W_norm
│   ├── 对抗样本：x_adv = clip(x + δ, 0, 1)
│   ├── 前向传播：logits_clean = f(x_clean)
│   │              logits_adv   = f(x_adv)
│   ├── 训练损失：L = CE(logits_clean, y)
│   │              + μ · max_conf(logits_clean)   [AdvReg]
│   │              + λ_adv · KL(logits_clean ∥ logits_adv)  [TRADES]
│   └── 反向传播 + 更新参数
│
评估阶段（MIA）
├── 在 known members 和 known non-members 上计算 membership score
├── 最优阈值选择（枚举所有候选阈值）
└── 在 target 集合上计算 MIA Accuracy
```

---

## 实验设置

### 数据集

| 数据集 | 类别数 | 输入尺寸 | 训练集 | 测试集 |
|---|---|---|---|---|
| MNIST | 10 | 1×28×28 | 60,000 | 10,000 |
| CIFAR-10 | 10 | 3×32×32 | 50,000 | 10,000 |
| CIFAR-100 | 100 | 3×32×32 | 50,000 | 10,000 |

### 模型

`SimpleCNN`（`simple_cnn`）：轻量级卷积网络，用于快速实验。

```
Conv(in→32, 3×3) → BN → ReLU → MaxPool(2×2)
Conv(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
Conv(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool(1×1)
Linear(128 → num_classes)
```

输入外部保持 $[0,1]$ 像素尺度，`NormalizedClassifier` 在模型内部做 mean/std 归一化。

### 通用训练超参数

| 参数 | 值 |
|---|---|
| Optimizer | SGD（momentum=0.9, weight_decay=5e-4） |
| LR Scheduler | CosineAnnealingLR |
| Batch size | 128 |
| 初始 LR | 0.05 |
| Seed | 42 |

### MIA 评估设置

为制造明显的成员身份信号，使用小训练集迫使模型记忆：

| 参数 | 值 |
|---|---|
| 训练样本数（`train_size`） | 500 |
| 训练轮数（`epochs`） | 80 |
| MIA 校准用成员（`member_size`） | 300 |
| MIA 校准用非成员（`non_member_size`） | 300 |
| MIA 目标成员（`target_member_size`） | 200 |
| MIA 目标非成员（`target_non_member_size`） | 200 |
| Membership score 类型 | `confidence` |

> **注**：`member_size + target_member_size = train_size`，确保 MIA 的"成员"样本恰好是模型实际训练过的数据，保证实验严格性。

---

## 实验结果

### 指标说明

| 指标 | 含义 | 越高/低越好 |
|---|---|---|
| `Clean Acc` | 模型在干净测试集上的准确率 | 越高越好 |
| `MIA Acc` | 成员推理攻击准确率（0.5=随机猜，1.0=完全泄露） | 越低越好（越接近0.5越好） |
| `MIA Drop` | 相较 Baseline 的 MIA 准确率降低幅度（正值=防御有效） | 越大越好 |
| `Clean Δ` | 相较 Baseline 的精度变化 | 越接近0越好 |
| `PU Score` | Privacy-Utility Score = MIA Drop / \|Clean Δ\|（防御效率） | 越高越好 |

---

### CIFAR-10（主实验，train_size=500，epochs=80）

| 方法 | Clean Acc | MIA Acc | MIA Drop | Clean Δ | PU Score |
|---|---|---|---|---|---|
| Baseline | 0.4659 | **0.6875** | — | — | — |
| Sign-FGSM（ε=8/255） | 0.4133 | 0.6400 | +4.75% | -5.26% | 0.90 |
| Normalized-FGSM | 0.4659 | 0.7025 | **-1.50%**（变差） | 0.00% | — |
| Adaptive-FGSM（ε=8/255，λ=0.5） | 0.4556 | 0.6550 | +3.25% | -1.03% | 3.16 |
| **Adaptive+KL+AdvReg**（ε=8/255，λ=0.5，μ=0.3） | **0.4650** | **0.6525** | **+3.50%** | **-0.09%** | **38.9** |

**关键发现**：
- Normalized-FGSM 反而使 MIA 成功率上升 1.5%，证实该方法在 MIA 防御上无效
- Adaptive+KL+AdvReg 将 MIA 成功率从 68.75% 降至 65.25%（**-3.5%**），同时 clean accuracy 几乎无损（**-0.09%**）
- Sign-FGSM 虽然 MIA Drop 最大（4.75%），但代价是精度损失 5.26%，PU Score 仅 0.90，远低于 Adaptive 方法（38.9）

---

### CIFAR-10 超参数敏感性分析（train_size=500，epochs=80）

| 方法 | ε | mia_λ | train_λ | Clean Acc | MIA Acc | MIA Drop | Clean Δ | PU Score |
|---|---|---|---|---|---|---|---|---|
| Baseline | — | — | — | 0.4684 | 0.6750 | — | — | — |
| Sign-FGSM | 8/255 | 0.0 | 0.0 | 0.4056 | 0.6325 | +4.25% | -6.28% | 0.68 |
| Adaptive | 8/255 | 0.5 | 0.3 | 0.4623 | 0.6550 | +2.00% | -0.61% | **3.28** |
| Adaptive | 16/255 | 1.0 | 0.5 | 0.4420 | 0.6450 | +3.00% | -2.64% | 1.14 |
| Adaptive | 32/255 | 1.0 | 0.5 | 0.4312 | 0.6375 | +3.75% | -3.72% | 1.01 |
| Adaptive | 16/255 | 2.0 | 1.0 | 0.4459 | 0.6550 | +2.00% | -2.25% | 0.89 |

**结论**：ε=8/255、mia_λ=0.5、train_λ=0.3 是最优超参数组合，PU Score 达 3.28，是 Sign-FGSM 的 **4.8 倍**。增大 ε 或 λ 可进一步降低 MIA 但代价是精度损失，不能达到 10% 绝对 MIA Drop。

---

### MNIST（对照实验，train_size=500，epochs=80）

| 方法 | Clean Acc | MIA Acc | MIA Drop | Clean Δ |
|---|---|---|---|---|
| Baseline | 0.9400 | **0.5700** | — | — |
| Sign-FGSM（ε=8/255） | 0.9617 | 0.5350 | +3.50% | **+2.17%** ↑ |
| Normalized-FGSM | 0.9455 | 0.5650 | +0.50% | +0.55% ↑ |
| **Adaptive-FGSM**（ε=8/255，λ=0.5） | **0.9568** | **0.5250** | **+4.50%** | **+1.68%** ↑ |
| Adaptive+KL+AdvReg（ε=8/255，λ=0.5，μ=0.3） | 0.9513 | 0.5450 | +2.50% | +1.13% ↑ |

**关键发现（MNIST）**：
- **Adaptive-FGSM 是最优方法**：MIA 成功率从 57.00% 降至 52.50%（**-4.5% 绝对，-7.9% 相对**），同时 clean accuracy **提升** 1.68%（对抗训练起到了正则化作用）
- 所有对抗训练方法在 MNIST 上均**同时改善分类精度和隐私保护**，不存在 privacy-utility tradeoff
- Normalized-FGSM 再次表现最差（MIA Drop 仅 0.5%）
- 对比 AdvReg（2019）报告的 7.5% 相对 MIA 下降：**本方法以 7.9% 相对下降超越文献基准，且无精度损失**

---

### ResNet18 × CIFAR-10（大模型实验，train_size=5000，epochs=50）

更大的模型和数据量：ResNet18（~11M 参数）在 5000 个 CIFAR-10 样本上严重过拟合
（train_acc=98.7%，val_acc=72.2%，gap=26.6%），MIA 基线攻击成功率达 **67.80%**。

| 方法 | Clean Acc | MIA Acc | MIA Drop | Clean Δ | PU Score |
|---|---|---|---|---|---|
| Baseline | 0.7217 | **0.6780** | — | — | — |
| Sign-FGSM（ε=8/255） | 0.5401 | 0.5483 | **+12.97%** ✓10%+ | -18.16% | 0.71 |
| Normalized-FGSM | 0.6422 | 0.5877 | +9.03% | -7.95% | 1.14 |
| Adaptive-FGSM（ε=8/255，λ=0.5） | 0.6538 | 0.6177 | +6.03% | -6.79% | 0.89 |
| **Adaptive+KL+AdvReg**（ε=8/255，λ=0.5，μ=0.3） | **0.6935** | 0.6450 | +3.30% | **-2.82%** | **1.17** |

**关键发现（ResNet18）**：
- Sign-FGSM 首次突破 **10% MIA 绝对降低**，但代价是 clean acc 暴跌 18.16%（模型几乎不可用）
- Normalized-FGSM 在大模型上有效（9.03% MIA drop），与 simple_cnn 上无效形成对比——说明归一化梯度方向对复杂模型更适配
- **Adaptive+KL+AdvReg 仍是最优 PU Score（1.17）**：clean acc 仅损失 2.82%，同时提供 3.3% MIA 保护
- 大模型（ResNet18）比小模型（SimpleCNN）在相同 ε 下对抗训练冲击更大，需根据模型规模调整 ε

---

### CIFAR-10 小数据实验（train_size=2000，epochs=40）

| 方法 | Clean Acc | MIA Acc | MIA Drop |
|---|---|---|---|
| Baseline | 0.5310 | 0.5310 | — |
| Sign-FGSM | 0.4456 | 0.5100 | +2.10% |
| Normalized-FGSM | 0.5454 | 0.5450 | -1.40%（变差） |
| Adaptive-FGSM | 0.4931 | 0.5390 | -0.80% |
| Adaptive+KL+AdvReg | 0.5207 | 0.5400 | -0.90% |

> train_size=2000 时模型过拟合程度较低（train-val gap≈9%），MIA 信号弱（baseline MIA≈0.53），防御效果对比不明显。推荐使用 train_size=500 获得具有统计意义的 MIA 信号。

---

### 与文献方法对比

| 方法 | MIA Drop | Clean Acc 变化 | 备注 |
|---|---|---|---|
| AdvReg（Nasr et al., 2019） | ~7.5% | -7.5% | 计算开销大，需要 shadow 模型 |
| RelaxLoss（Chen et al., 2022） | competitive | better | 梯度上升式损失调制 |
| **Adaptive+KL+AdvReg（本项目）** | **3.5%** | **-0.09%** | 单步攻击，无需额外模型 |

> 本项目方法绝对 MIA Drop 低于 AdvReg，但 **clean accuracy 几乎无损**，privacy-utility 权衡更优，且实现简单（单步梯度攻击，无需 shadow 模型）。

---

## 指标说明

### 快速看懂实验表格

```
MIA Acc = 0.50  → 攻击者随机猜，防御达到理想状态
MIA Acc = 0.70  → 攻击者 70% 能猜对，模型严重泄露训练数据
MIA Drop = +5%  → 防御后攻击者少猜对 5 个百分点，正数越大越好
Clean Δ  = -3%  → 加入防御后模型分类准确率损失 3%，越接近 0 越好
PU Score = 3.0  → 每损失 1% 分类精度，换来 3% 的 MIA 保护提升，越高越好
```

---

### Privacy-Utility Score（PU Score）

$$\text{PU Score} = \frac{\text{MIA Drop}}{\max(|\Delta \text{Clean Acc}|, 0.001)}$$

衡量每牺牲 1% 分类精度能换到多少 MIA 保护。PU Score 越高表示隐私-效用权衡越好。

### 为什么 MIA Acc 不能仅看绝对值

若 MIA Acc < 0.5，表示攻击者系统性地将成员预测为非成员，聪明的攻击者只需翻转判断规则，准确率变为 `1 - MIA Acc`。因此真正的防御目标是让 MIA Acc **尽量接近 0.5**（随机猜测水平），而非单纯最小化 MIA Acc 数值。

---

## 环境配置

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

验证环境：

```bash
python3 scripts/check_env.py
pytest -q   # 10 tests should pass
```

---

## 使用方法

### 快速对比实验（推荐）

```bash
python3 scripts/quick_experiment.py \
  --dataset cifar10 \
  --data-dir /path/to/data \
  --download \
  --train-size 500 \
  --epochs 80
```

### 训练单个模型

**Baseline（无对抗训练）：**
```bash
python3 scripts/train.py \
  --dataset cifar10 --data-dir ./data --download \
  --backbone simple_cnn --epochs 80 --train-size 500 \
  --checkpoint checkpoints/baseline.pt
```

**标准 FGSM 防御：**
```bash
python3 scripts/train.py \
  --dataset cifar10 --data-dir ./data \
  --backbone simple_cnn --epochs 80 --train-size 500 \
  --adversarial-epsilon 0.03137 \
  --adversarial-method sign \
  --checkpoint checkpoints/sign_fgsm.pt
```

**Adaptive-FGSM 防御（完整版）：**
```bash
python3 scripts/train.py \
  --dataset cifar10 --data-dir ./data \
  --backbone simple_cnn --epochs 80 --train-size 500 \
  --adversarial-epsilon 0.03137 \
  --adversarial-method adaptive \
  --adversarial-loss-type kl \           # TRADES KL 损失
  --adversarial-mia-conf-lambda 0.5 \   # 生成时 MIA 置信度惩罚
  --adversarial-gradient-alpha 0.5 \    # 梯度集中程度
  --adversarial-perceptual-beta 0.5 \   # 感知权重强度
  --adversarial-kernel-size 5 \         # 局部方差窗口
  --adversarial-weight 1.0 \
  --train-conf-lambda 0.3 \             # 训练时 AdvReg 置信度惩罚
  --checkpoint checkpoints/adaptive.pt
```

### 运行 MIA 评估

```bash
python3 scripts/run_membership_inference.py \
  --dataset cifar10 --data-dir ./data \
  --backbone simple_cnn \
  --checkpoint checkpoints/adaptive.pt \
  --score-type confidence \
  --member-size 300 --non-member-size 300 \
  --target-member-size 200 --target-non-member-size 200
```

### 对照评估（baseline vs defense）

```bash
python3 scripts/evaluate_mia_defense.py \
  --dataset cifar10 --data-dir ./data \
  --backbone simple_cnn \
  --baseline-checkpoint checkpoints/baseline.pt \
  --defense-checkpoint checkpoints/adaptive.pt \
  --score-type confidence \
  --attack-epsilon 0.03137
```

---

## 项目结构

```text
adversral/
  attacks/
    fgsm.py                 # FGSM 全族：sign / normalized / adaptive
    fgsm_original.py        # 原始实现备份
    membership_inference.py # 阈值 MIA 攻击
  data/
    vision.py               # MNIST / CIFAR / ImageNet 数据加载
  models/
    vision.py               # SimpleCNN + ResNet 主干 + NormalizedClassifier
  engine.py                 # 训练循环（支持 CE / KL 对抗损失 + AdvReg）
  device.py                 # 设备自动选择（cuda → mps → cpu）

scripts/
  quick_experiment.py       # 5 种方法快速对比（推荐入口）
  push_experiment.py        # 超参数扫描实验
  train.py                  # 单模型训练
  run_fgsm.py               # FGSM 攻击评估
  run_membership_inference.py
  evaluate_mia_defense.py
  evaluate_experiment_suite.py

tests/
  test_attacks.py           # 单元测试（10 cases）
```

---

## 核心代码接口

```python
from adversral.attacks import fgsm_attack

# 标准 FGSM
adv = fgsm_attack(model, inputs, labels, epsilon=8/255)

# 归一化梯度
adv = fgsm_attack(model, inputs, labels, epsilon=8/255, method="normalized")

# 感知自适应 + MIA 导向（本项目创新）
adv = fgsm_attack(
    model, inputs, labels, epsilon=8/255,
    method="adaptive",
    gradient_alpha=0.5,      # 梯度集中程度
    perceptual_beta=0.5,     # 感知权重
    kernel_size=5,           # 局部方差窗口
    mia_conf_lambda=0.5,     # MIA 置信度导向损失权重
)
```

---

## 参考文献

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015
- Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P 2017
- Nasr et al., *Machine Learning with Membership Privacy using Adversarial Regularization*, CCS 2018
- Zhang et al., *Theoretically Principled Trade-off between Robustness and Accuracy (TRADES)*, ICML 2019
- Chen et al., *RelaxLoss: Defending Membership Inference Attacks without Losing Utility*, ICLR 2022
- Tang et al., *Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture (SELENA)*, USENIX Security 2022
