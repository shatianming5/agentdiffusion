# AgentDiffusion：基于潜空间扩散模型的百万级金融市场 Agent-Based Model 加速框架

## 完整实现执行方案

---

## 一、项目定位与核心贡献

### 1.1 问题陈述

传统金融 Agent-Based Model (ABM) 在百万级 agent 规模下面临严重的计算瓶颈：每个时间步需要逐个更新所有 agent 的状态，通过订单簿撮合交易，计算复杂度为 O(N·T)，其中 N 为 agent 数量，T 为时间步数。一个百万级 agent、1000 步的模拟可能需要数小时甚至数天。

### 1.2 核心思路

将 ABM 的状态演化过程建模为一个条件生成问题：给定 t 时刻的全局 agent 状态张量 S_t ∈ R^{H×W×C}，生成 t+1 时刻的状态张量 S_{t+1}。使用 Latent Diffusion Model 学习这个转移分布 p(S_{t+1} | S_t, M_t)，其中 M_t 为宏观市场条件。

### 1.3 论文贡献点（面向顶会投稿）

1. **架构创新**：首个将 Latent Diffusion + Perceiver-style Cross-Attention 应用于大规模 ABM 代理的框架
2. **约束满足**：三层约束注入策略（软约束训练 + Guided Diffusion + 投影兜底），首次在生成式 ABM 中严格保证市场出清等硬约束
3. **规模突破**：百万级 agent 的一次性并行生成，相比传统 ABM 实现 100x-1000x 加速
4. **涌现验证**：模型在训练分布外的极端条件下自发产生闪崩、泡沫等涌现现象

---

## 二、文献基础与技术选型依据

### 2.1 核心参考论文

| 编号 | 论文 | 关键技术 | 与本项目关系 |
|------|------|----------|-------------|
| R1 | Peebles & Xie (2022). "Scalable Diffusion Models with Transformers" (DiT). arXiv:2212.09748 | Transformer 替代 U-Net 做 diffusion backbone；adaLN-Zero 条件注入；patch 化处理 | **核心架构基础**：我们的去噪网络直接基于 DiT，将 image patch 替换为 agent patch |
| R2 | Jaegle et al. (2021). "Perceiver IO: A General Architecture for Structured Inputs & Outputs". arXiv:2107.14795 | Cross-attention latent bottleneck；O(n·m) 复杂度处理任意长度输入 | **全局交互机制**：市场状态 token 的设计直接借鉴 Perceiver 的 latent array |
| R3 | Amrouni et al. (2021). "ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation". arXiv:2110.14771 | 基于 ABIDES 的金融市场离散事件模拟；多种 agent 类型；Gym 接口 | **训练数据来源**：使用 ABIDES 生成训练数据 |
| R4 | Dwarakanath et al. (2024). "ABIDES-Economist: Agent-Based Simulator with Learning Agents". arXiv:2402.09563 | ABIDES 扩展版，支持学习型 agent；宏观经济模拟 | **数据生成器升级版**：更丰富的 agent 行为模式 |
| R5 | Wheeler & Varner (2023). "Scalable Agent-Based Modeling for Complex Financial Market Simulations". arXiv:2312.14903 | 大规模 ABM 分布式计算框架；多资产同时交易 | **Baseline 对比**：传统 ABM 扩展方案的性能基准 |
| R6 | Liu & Thuerey (2023). "Uncertainty-aware Surrogate Models with DDPM". arXiv:2312.05320 | Diffusion 模型作为物理模拟的 surrogate；不确定性量化 | **方法论先例**：证明 diffusion 可以做 surrogate simulator |
| R7 | Camburn (2025). "Universal Physics Simulation: A Foundational Diffusion Approach". arXiv:2507.09733 | 通用物理模拟的基础 diffusion 方法；从边界条件数据直接学习物理定律 | **方法论验证**：diffusion 作为通用模拟器的可行性 |
| R8 | Jung (2025). "Guided Discrete Diffusion for Constraint Satisfaction Problems". arXiv:2512.14765 | 离散 diffusion 的 guidance 方法用于约束满足 | **约束注入参考**：guided diffusion 处理硬约束的具体技术 |
| R9 | Meijer & Chen (2024). "The Rise of Diffusion Models in Time-Series Forecasting". arXiv:2401.03006 | Diffusion 用于时间序列生成/预测的综述 | **方法论综述**：diffusion 在时序数据上的适用性 |
| R10 | Kollovieh et al. (2023). "Predict, Refine, Synthesize: Self-Guiding Diffusion for Time Series". arXiv:2307.11494 | Self-guidance 机制用于时间序列 diffusion | **生成策略参考**：自引导去噪的具体实现 |
| R11 | Xiao et al. (2024). "TradingAgents: Multi-Agents LLM Financial Trading Framework". arXiv:2412.20138 | LLM 驱动的多 agent 金融交易框架 | **对比方法**：LLM agent vs 我们的 diffusion agent |
| R12 | Alfi et al. (2008). "Minimal ABM for Financial Markets: Origin of Stylized Facts". arXiv:0808.3562 | 最小 ABM 模型如何产生 stylized facts | **评估理论基础**：stylized facts 的来源机制 |
| R13 | Safari et al. (2025). "International Financial Markets Through 150 Years: Evaluating Stylized Facts". arXiv:2504.08611 | 150年金融数据的 stylized facts 定量评估 | **评估标准**：最新的 stylized facts 检验方法 |
| R14 | Bodor & Carlier (2024). "Stylized Facts and Market Microstructure". arXiv:2401.10722 | 债券期货市场的 stylized facts 与微观结构 | **评估细节**：微观结构层面的指标 |

### 2.2 关键开源资源

| 资源 | 用途 | 地址 |
|------|------|------|
| ABIDES | 训练数据生成 | github.com/jpmorganchase/abides-jpmc-public |
| DiT | 去噪网络基础架构 | github.com/facebookresearch/DiT |
| diffusers (HuggingFace) | Diffusion 训练/推理框架 | github.com/huggingface/diffusers |
| FlashAttention | 高效 attention 实现 | github.com/Dao-AILab/flash-attention |

---

## 三、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AgentDiffusion Framework                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Data Layer   │───▶│  Encoding Layer  │───▶│  Diffusion Core  │  │
│  │              │    │                  │    │                  │  │
│  │ ABIDES 模拟器 │    │ Per-Agent MLP    │    │ DiT Backbone     │  │
│  │ 数据预处理    │    │ Spatial Patchify │    │ Market CrossAttn │  │
│  │ Agent 分组    │    │ Patch Encoder    │    │ Constraint Guide │  │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                       │            │
│  ┌──────────────┐    ┌──────────────────┐             │            │
│  │  Eval Layer   │◀───│  Decoding Layer  │◀────────────┘            │
│  │              │    │                  │                           │
│  │ Stylized Facts│    │ Patch Decoder    │                           │
│  │ ABM 保真度   │    │ Per-Agent MLP    │                           │
│  │ 加速比测量   │    │ Projection Layer │                           │
│  └──────────────┘    └──────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、详细模块设计

### 4.1 Agent 状态表示

每个 agent 的原始状态向量 a_i ∈ R^C，其中 C = 128 维，包含：

| 维度范围 | 含义 | 备注 |
|---------|------|------|
| [0:32] | 持仓向量 | 各资产持有量 |
| [32:48] | 资金状态 | 现金、杠杆、保证金 |
| [48:64] | 策略参数 | agent 类型编码 + 策略超参 |
| [64:80] | 历史统计 | 近期收益、波动率、夏普等 |
| [80:96] | 行为特征 | 下单频率、撤单率、方向偏好 |
| [96:112] | 市场观察 | agent 感知到的价格、深度 |
| [112:128] | 社交/信息 | 信息来源、跟风系数、恐慌指数 |

**Agent 网格排列**：N 个 agent 排列成 H × W 的 2D 网格（H × W = N），排列规则如下：
- 按策略类型分大区（做市商、趋势跟踪、基本面、噪声交易者）
- 同类型内按资金规模排列
- 这样保证 spatial patch 内的 agent 具有相似特性，局部 attention 有意义

### 4.2 编码层（Encoding Layer）

#### 4.2.1 Per-Agent MLP Encoder

```
输入: a_i ∈ R^128
      ↓
  LayerNorm
      ↓
  Linear(128, 64) + GELU
      ↓
  Linear(64, 32) + GELU
      ↓
  Linear(32, d_agent)       # d_agent = 16
      ↓
输出: z_i ∈ R^16
```

- 所有 agent 共享同一个 encoder（参数共享）
- 可选：按 agent 类型分组使用不同 encoder（4 种类型 × 4 个 encoder）
- 训练目标：autoencoder 重建损失 L_recon = ‖decode(encode(a_i)) - a_i‖²

#### 4.2.2 Spatial Patchification

```
输入: Z ∈ R^{H × W × d_agent}     # 例如 1024×1024×16
      ↓
  划分为 p×p 的 patch            # p = 4，得到 256×256 个 patch
      ↓
  每个 patch 展平: R^{p×p×d_agent} → R^{p²·d_agent}  # 4×4×16 = 256
      ↓
  Linear(p²·d_agent, d_model)  # d_model = 512
      ↓
  + Learnable 2D Position Embedding
      ↓
输出: tokens ∈ R^{(H/p)×(W/p) × d_model}  # 256×256×512 = 65536 个 token
```

**关键超参数**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| patch_size p | 4 | 每 patch 16 个 agent；太大信息损失严重，太小 token 数不减 |
| d_agent | 16 | agent 压缩维度；需要通过 recon loss 实验确定 |
| d_model | 512 | transformer hidden dim；参考 DiT-L 配置 |
| num_tokens | 65536 | 256×256；可用分层 attention 处理 |

### 4.3 扩散核心（Diffusion Core）

#### 4.3.1 噪声调度

采用 **cosine schedule**（改进自 Nichol & Dhariwal 2021）：

```
β_t 由 cosine schedule 确定
α_t = 1 - β_t
ᾱ_t = ∏_{s=1}^{t} α_s

前向过程: q(z_t | z_0) = N(z_t; √ᾱ_t · z_0, (1-ᾱ_t) · I)
```

- 总去噪步数 T = 1000（训练），推理时用 DDIM 加速到 50-100 步
- v-prediction 参数化（优于 epsilon 和 x0 参数化，参考 Salimans & Ho 2022）

#### 4.3.2 去噪网络：Market-Aware DiT

每个 DiT Block 的结构：

```
输入: x ∈ R^{N_tokens × d_model}
      market_tokens ∈ R^{M × d_model}   # M = 128 个市场 token

  ┌─── DiT Block ───────────────────────────────────────────┐
  │                                                         │
  │  1. Patch-内局部 Self-Attention (window_size = 8×8)     │
  │     - 每 64 个相邻 token 做 local attention             │
  │     - 使用 FlashAttention 加速                          │
  │     - adaLN-Zero 条件注入（时间步 t + 市场条件 c）      │
  │                                                         │
  │  2. Agent → Market Cross-Attention                      │
  │     - Q = agent tokens, K/V = market tokens             │
  │     - 每个 agent token 从市场 token 读取全局信息         │
  │     - 复杂度 O(N_tokens × M) ≈ 65536 × 128             │
  │                                                         │
  │  3. Market Token Self-Attention                         │
  │     - market tokens 之间做 full attention               │
  │     - 聚合来自所有 agent 的信息                          │
  │     - 复杂度 O(M²) = 128² = 16384（可忽略）             │
  │                                                         │
  │  4. Market → Agent Cross-Attention                      │
  │     - Q = agent tokens, K/V = updated market tokens     │
  │     - 把聚合后的全局信息广播回每个 agent                 │
  │                                                         │
  │  5. FFN (Feed-Forward Network)                          │
  │     - 标准 DiT FFN，adaLN-Zero 条件注入                 │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

**网络规模**：

| 配置名 | 层数 | d_model | heads | 市场 token 数 M | 参数量估计 |
|--------|------|---------|-------|----------------|-----------|
| AgentDiT-S | 12 | 384 | 6 | 64 | ~120M |
| AgentDiT-B | 12 | 512 | 8 | 128 | ~250M |
| AgentDiT-L | 24 | 768 | 12 | 128 | ~650M |
| AgentDiT-XL | 28 | 1024 | 16 | 256 | ~1.2B |

推荐从 **AgentDiT-B** 开始实验，验证通过后尝试 L 和 XL。

#### 4.3.3 条件注入

采用 DiT 的 **adaLN-Zero** 机制：

```
条件信号 c = [t_embed, market_cond, scenario_cond]

t_embed:       时间步 t 的正弦位置编码，经 MLP 映射到 d_model
market_cond:   当前市场宏观状态（价格、波动率、利率等），经 MLP 映射
scenario_cond: 场景标签（正常/危机/泡沫等），embedding 后拼接

c_total = MLP(concat(t_embed, market_cond, scenario_cond))

adaLN-Zero: γ, β, α = chunk(Linear(c_total), 3)
            x = α · Attention(γ · LayerNorm(x) + β)
```

### 4.4 约束注入系统

#### 4.4.1 约束定义

| 约束名 | 数学形式 | 类型 | 注入方式 |
|--------|---------|------|---------|
| 市场出清 | Σ_i Δposition_i = 0 | 硬/等式 | Guided + 投影 |
| 预算约束 | cash_i + Σ_j price_j · position_{i,j} ≥ 0 | 软/不等式 | Guided |
| 持仓非负（多头） | position_i ≥ 0（对多头 agent） | 软/不等式 | Guided + clamp |
| 杠杆上限 | leverage_i ≤ L_max | 软/不等式 | Guided |
| 总量守恒 | Σ_i position_{i,j} = S_j（对每种资产 j） | 硬/等式 | 投影 |

#### 4.4.2 三层约束策略实现

**第一层：训练阶段软约束**

```python
L_total = L_diffusion + λ_clear · L_clearing + λ_budget · L_budget + λ_conserve · L_conservation

L_clearing = ‖Σ_i Δposition_i‖²                    # 市场出清
L_budget = Σ_i max(0, -budget_i)²                    # 预算约束（ReLU 惩罚）
L_conservation = ‖Σ_i position_i - S_target‖²        # 总量守恒

λ 的初始值：λ_clear=1.0, λ_budget=0.5, λ_conserve=1.0
可选：自适应 λ（参考 Lagrangian relaxation，每 N 步更新）
```

**第二层：推理阶段 Guided Diffusion**

```python
# 在每个去噪步 t:
x_t_hat = denoise(x_t, t, c)                        # 标准去噪

# 计算约束函数的梯度
g_clear = ∇_{x_t} ‖Σ_i Δposition_i(x_t)‖²
g_budget = ∇_{x_t} Σ_i max(0, -budget_i(x_t))²

# Guidance（仅在 t < T/2 时开启，早期噪声太大无效）
x_t_hat = x_t_hat - s_clear · g_clear - s_budget · g_budget

# s 为 guidance scale，可设为固定值或随 t 衰减
s_clear(t) = s_0 · (1 - t/T)    # 越接近 t=0，guidance 越强
```

**第三层：输出投影兜底**

```python
# 对等式约束做精确投影
def project_market_clearing(delta_positions):
    """确保所有 agent 的持仓变化之和为零"""
    mean_delta = delta_positions.mean(dim=0)  # 对 agent 维度求均值
    return delta_positions - mean_delta        # 减去均值

def project_conservation(positions, target_total):
    """确保各资产总持仓等于目标"""
    current_total = positions.sum(dim=0)
    correction = (target_total - current_total) / N
    return positions + correction

# 对不等式约束做 clamp
positions = positions.clamp(min=0)             # 非负约束
leverage = leverage.clamp(max=L_max)           # 杠杆上限
```

### 4.5 解码层（Decoding Layer）

```
输入: denoised_tokens ∈ R^{(H/p)×(W/p) × d_model}

  1. Linear(d_model, p²·d_agent)
      ↓
  2. Reshape 为 patch grid: R^{(H/p)×(W/p) × p × p × d_agent}
      ↓
  3. 拼接还原为完整网格: R^{H × W × d_agent}
      ↓
  4. Per-Agent MLP Decoder (共享参数):
     Linear(d_agent, 32) + GELU → Linear(32, 64) + GELU → Linear(64, 128)
      ↓
  5. 约束投影层 (Project)
      ↓
输出: S_{t+1} ∈ R^{H × W × 128}
```

---

## 五、训练数据生成管线

### 5.1 ABIDES 数据生成配置

#### Phase 1: 小规模验证（10K agents）

```yaml
# configs/abides_small.yaml
simulation:
  num_agents: 10000
  num_steps: 2000
  tick_size: 0.01

agents:
  market_makers:
    count: 200            # 2%
    spread_range: [0.01, 0.05]
    inventory_limit: 1000

  trend_followers:
    count: 3000           # 30%
    lookback_range: [10, 200]
    momentum_threshold: [0.01, 0.05]

  fundamentalists:
    count: 2000           # 20%
    fair_value_noise: 0.1
    mean_reversion_speed: [0.01, 0.1]

  noise_traders:
    count: 4800           # 48%
    arrival_rate: 0.1
    order_size_range: [1, 100]

data_generation:
  num_simulations: 5000   # 不同初始条件
  save_interval: 10       # 每 10 步保存一次快照
  output_dir: "data/abides_small/"

  # 变化参数（每次模拟随机采样）
  vary_params:
    - agent_count_ratios   # agent 类型比例
    - initial_price        # 初始价格 [50, 200]
    - volatility_regime    # 波动率环境 [low, medium, high, crisis]
```

#### Phase 2: 大规模训练（100K-1M agents）

```yaml
# configs/abides_large.yaml
simulation:
  num_agents: 100000       # 先 100K，验证后扩展到 1M
  num_steps: 1000
  tick_size: 0.01

  # 多资产支持
  assets:
    - name: "STOCK_A"
      initial_price: 100
    - name: "STOCK_B"
      initial_price: 50
      correlation_with_A: 0.6

agents:
  # ... 同上但数量等比放大 ...

data_generation:
  num_simulations: 1000    # 大规模模拟数量少但每个更大
  save_interval: 5
  parallel_workers: 32     # 并行生成
  output_dir: "data/abides_large/"
```

### 5.2 数据预处理流程

```
原始 ABIDES 输出（JSON/Pickle）
       ↓
  1. 提取每个 agent 每个时间步的状态向量（128维）
       ↓
  2. 归一化：
     - 持仓类：log(1 + |x|) · sign(x)
     - 资金类：除以初始资金做相对化
     - 策略参数：已经是归一化的
     - 统计量：z-score 标准化
       ↓
  3. Agent 排列：按策略类型分区 → 同区内按资金排序 → 填充到 H×W 网格
       ↓
  4. 时间对（S_t, S_{t+1}）配对，加上市场条件 M_t
       ↓
  5. 保存为 .pt 文件（内存映射）

单个样本格式：
{
    "state_t":     Tensor[H, W, 128],    # 当前状态
    "state_t1":    Tensor[H, W, 128],    # 下一步状态
    "market_cond": Tensor[32],           # 宏观市场条件
    "agent_types": Tensor[H, W],         # agent 类型标签
    "time_index":  int,                  # 时间步索引
    "sim_id":      int,                  # 模拟实例 ID
}
```

### 5.3 数据规模估算

| 阶段 | Agent 数 | 网格尺寸 | 模拟数 | 时间步/对 | 总样本数 | 存储估算 |
|------|---------|---------|--------|----------|---------|---------|
| Phase 1 | 10K | 100×100 | 5000 | 200 | 1M | ~50GB |
| Phase 2 | 100K | 316×316 | 1000 | 200 | 200K | ~250GB |
| Phase 3 | 1M | 1024×1024 | 200 | 100 | 20K | ~500GB |

---

## 六、训练流程

### 6.1 两阶段训练

#### Stage 1: Autoencoder 预训练

```
目标：训练 Per-Agent Encoder/Decoder，学习压缩表示

损失函数：
L_ae = L_recon + β · L_kl（如果使用 VAE）
     = ‖decode(encode(a_i)) - a_i‖² + β · KL(q(z|a) ‖ p(z))

β = 0.01（弱 KL 约束，优先保证重建质量）

训练配置：
- Batch size: 4096（agent 级别）
- Learning rate: 1e-4, cosine decay
- Epochs: 100
- 评估指标：Per-dimension 重建误差（确保关键维度如持仓、资金误差 < 1%）
```

#### Stage 2: Diffusion 模型训练

```
目标：在 latent space 训练条件 diffusion 模型

损失函数：
L = L_v-pred + λ_clear · L_clearing + λ_budget · L_budget

L_v-pred = E_{t,ε} [‖v_θ(z_t, t, c) - v_target‖²]

其中 v_target = √ᾱ_t · ε - √(1-ᾱ_t) · z_0

训练配置：
- Batch size: 32（状态帧级别，每帧是完整的 agent 网格）
- Learning rate: 1e-4, warmup 5000 steps, cosine decay
- Total steps: 500K (Phase 1), 200K (Phase 2)
- EMA decay: 0.9999
- Gradient clipping: 1.0
- Mixed precision: bf16
- 硬件需求：8× A100 80GB（Phase 1），32× A100（Phase 2/3）
```

### 6.2 训练技巧

| 技巧 | 细节 |
|------|------|
| 课程学习 | 先在小网格（100×100）训练，逐步增大到 1024×1024 |
| 时间步采样 | Importance sampling，偏向中等噪声水平（t ∈ [200,800]） |
| Agent dropout | 训练时随机 mask 10% 的 agent（替换为零向量），增强鲁棒性 |
| 多步预测 | 除了单步 (S_t → S_{t+1})，加入 K 步预测损失 |
| 梯度累积 | 大网格时 batch size 可能只有 1-2，用梯度累积到等效 32 |

---

## 七、推理流程

### 7.1 单步推理

```python
def inference_step(state_t, market_cond, num_denoise_steps=50):
    """从 S_t 生成 S_{t+1}"""

    # 1. Encode
    z_t = encode(state_t)                    # [H, W, 128] → [H, W, 16]
    z_patches = patchify(z_t)                # [H, W, 16] → [H/p, W/p, d_model]

    # 2. 初始化噪声
    z_T = torch.randn_like(z_patches)

    # 3. DDIM 去噪循环
    for t in ddim_schedule(num_denoise_steps):
        # 预测 v
        v_pred = model(z_T, t, market_cond)

        # DDIM 更新
        z_T = ddim_step(z_T, v_pred, t)

        # Constraint guidance（仅 t < T/2）
        if t < num_denoise_steps // 2:
            z_T = apply_constraint_guidance(z_T, guidance_scales)

    # 4. Decode
    z_decoded = unpatchify(z_T)              # [H/p, W/p, d_model] → [H, W, 16]
    state_t1 = decode(z_decoded)             # [H, W, 16] → [H, W, 128]

    # 5. 约束投影
    state_t1 = project_constraints(state_t1, state_t)

    return state_t1
```

### 7.2 长时间步滚动推理

```python
def rollout(initial_state, num_steps, market_conditions):
    """生成完整的模拟轨迹"""
    states = [initial_state]

    for k in range(num_steps):
        state_next = inference_step(
            states[-1],
            market_conditions[k],
            num_denoise_steps=50
        )
        states.append(state_next)

        # 可选：每 N 步做一次完整约束检查和修正
        if k % 10 == 0:
            states[-1] = full_constraint_check(states[-1])

    return torch.stack(states)
```

### 7.3 加速策略

| 策略 | 加速比 | 质量影响 |
|------|--------|---------|
| DDIM (1000→50 steps) | 20x | 轻微 |
| 半精度推理 (bf16) | 2x | 几乎无 |
| FlashAttention | 2-3x | 无 |
| Distillation (50→4 steps) | 12.5x | 需要额外训练 |
| Batch 化多场景 | Nx | 无 |

综合预期：单次前向传播 ~0.5s（A100），vs 传统 ABM 单步 ~60s → **~120x 加速**。

---

## 八、评估框架

### 8.1 Stylized Facts 检验清单

| Stylized Fact | 检验方法 | 通过标准 |
|--------------|---------|---------|
| 收益率肥尾 | Hill estimator 估计尾指数 α | 2 < α < 5 |
| 波动率聚集 | |r_t| 的 ACF 衰减形态 | 慢衰减，lag=100 时 ACF > 0.1 |
| 杠杆效应 | corr(r_t, σ²_{t+k}) 对 k 的曲线 | 负相关，peak 在 k=1-5 |
| 成交量-波动率正相关 | Pearson corr(volume_t, |r_t|) | ρ > 0.3 |
| 收益率无自相关 | r_t 的 ACF | lag > 1 时 |ACF| < 0.05 |
| Gain/Loss 不对称 | 上涨和下跌的持续时间分布 | K-S 检验 p < 0.05 |

### 8.2 ABM 保真度指标

```python
metrics = {
    # 分布距离
    "wasserstein_position":   wasserstein_1d(gen_positions, abm_positions),
    "wasserstein_cash":       wasserstein_1d(gen_cash, abm_cash),
    "wasserstein_returns":    wasserstein_1d(gen_returns, abm_returns),

    # 时序特征
    "acf_l2_returns":         l2_distance(acf(gen_returns), acf(abm_returns)),
    "acf_l2_volatility":      l2_distance(acf(gen_vol), acf(abm_vol)),

    # 多步预测
    "mse_1step":              mse(gen_state_t1, abm_state_t1),
    "mse_10step":             mse(gen_state_t10, abm_state_t10),
    "mse_100step":            mse(gen_state_t100, abm_state_t100),

    # 约束违反度
    "clearing_violation":     abs(sum(delta_positions)).mean(),
    "budget_violation":       relu(-budgets).mean(),
    "conservation_violation": abs(sum(positions) - target).mean(),
}
```

### 8.3 加速比测量

```python
# 公平对比设置
benchmark = {
    "agent_counts":   [10_000, 100_000, 1_000_000],
    "time_steps":     [100, 500, 1000],
    "metrics":        ["wall_clock_time", "gpu_memory", "cpu_memory"],
    "baselines":      ["ABIDES", "ABIDES-parallel", "Mesa"],
    "hardware":       "1x A100 80GB",
}

# 报告格式
# Agent数 | 步数 | ABIDES耗时 | 我们耗时 | 加速比 | 保真度(W-dist)
```

### 8.4 涌现现象测试

| 测试场景 | 初始条件 | 观察目标 |
|---------|---------|---------|
| 闪崩 | 大量趋势跟踪者 + 外部冲击 | 价格瞬间暴跌 > 5% 后反弹 |
| 泡沫周期 | 乐观情绪偏高 + 低利率 | 价格持续上涨后崩盘 |
| 流动性枯竭 | 做市商减少 50% | Spread 急剧扩大 |
| 羊群效应 | 高跟风系数 | 成交量和波动率同步爆发 |

---

## 九、项目目录结构

```
agentdiffusion/
├── configs/                          # 配置文件
│   ├── model/
│   │   ├── agent_dit_s.yaml
│   │   ├── agent_dit_b.yaml
│   │   └── agent_dit_l.yaml
│   ├── data/
│   │   ├── abides_small.yaml
│   │   └── abides_large.yaml
│   └── train/
│       ├── stage1_ae.yaml
│       └── stage2_diffusion.yaml
│
├── agentdiffusion/                   # 核心代码包
│   ├── __init__.py
│   ├── data/                         # 数据层
│   │   ├── __init__.py
│   │   ├── abides_generator.py       # ABIDES 数据生成脚本
│   │   ├── dataset.py                # PyTorch Dataset
│   │   ├── preprocessing.py          # 归一化、网格排列
│   │   └── agent_state.py            # Agent 状态定义
│   │
│   ├── models/                       # 模型层
│   │   ├── __init__.py
│   │   ├── autoencoder.py            # Per-Agent AE / VAE
│   │   ├── patchify.py               # Spatial Patch 化
│   │   ├── dit_block.py              # 单个 DiT Block（含 market cross-attn）
│   │   ├── agent_dit.py              # 完整 AgentDiT 模型
│   │   ├── market_tokens.py          # 可学习市场状态 token
│   │   ├── embeddings.py             # 时间步 / 位置 / 条件 embedding
│   │   └── attention.py              # Local Attention + Cross-Attention
│   │
│   ├── diffusion/                    # 扩散过程
│   │   ├── __init__.py
│   │   ├── scheduler.py              # Noise schedule (cosine)
│   │   ├── ddpm.py                   # DDPM 训练逻辑
│   │   ├── ddim.py                   # DDIM 推理加速
│   │   └── v_prediction.py           # v-prediction 参数化
│   │
│   ├── constraints/                  # 约束系统
│   │   ├── __init__.py
│   │   ├── soft_loss.py              # 训练阶段软约束 loss
│   │   ├── guidance.py               # 推理阶段 Guided Diffusion
│   │   └── projection.py             # 输出投影（市场出清、守恒）
│   │
│   ├── train/                        # 训练
│   │   ├── __init__.py
│   │   ├── train_ae.py               # Stage 1: AE 训练
│   │   ├── train_diffusion.py        # Stage 2: Diffusion 训练
│   │   ├── optimizer.py              # 优化器配置
│   │   └── callbacks.py              # Checkpoint、日志、EMA
│   │
│   ├── infer/                        # 推理
│   │   ├── __init__.py
│   │   ├── single_step.py            # 单步推理
│   │   ├── rollout.py                # 长时间步滚动
│   │   └── batch_scenarios.py        # 批量场景生成
│   │
│   ├── eval/                         # 评估
│   │   ├── __init__.py
│   │   ├── stylized_facts.py         # Stylized Facts 检验
│   │   ├── fidelity.py               # ABM 保真度指标
│   │   ├── speedup.py                # 加速比测量
│   │   ├── emergence.py              # 涌现现象测试
│   │   └── visualize.py              # 可视化工具
│   │
│   └── utils/                        # 工具
│       ├── __init__.py
│       ├── config.py                 # 配置加载
│       ├── distributed.py            # 分布式训练工具
│       └── logging.py                # 日志工具
│
├── scripts/                          # 启动脚本（连通性验证后自删）
├── tests/                            # 单元测试
│   ├── test_autoencoder.py
│   ├── test_patchify.py
│   ├── test_dit_block.py
│   ├── test_constraints.py
│   └── test_data_pipeline.py
│
├── pyproject.toml                    # 项目配置
└── IMPLEMENTATION_PLAN.md            # 本文件
```

---

## 十、分阶段实施计划

### Phase 0: 基础设施搭建（Week 1-2）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 初始化项目、依赖管理 | pyproject.toml, 虚拟环境 | 无 |
| 实现 Agent 状态定义 | agent_state.py | 无 |
| 实现配置系统 | config.py + yaml 文件 | 无 |
| 编写数据预处理管线 | preprocessing.py | agent_state.py |
| 安装和测试 ABIDES | abides_generator.py | ABIDES 环境 |
| 生成小规模验证数据（1K agents, 100 sims） | data/abides_tiny/ | ABIDES |

### Phase 1: Autoencoder（Week 3-4）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 实现 Per-Agent MLP Encoder/Decoder | autoencoder.py | agent_state.py |
| 实现 Spatial Patchify/Unpatchify | patchify.py | 无 |
| AE 训练脚本 | train_ae.py | autoencoder.py |
| 在 tiny 数据上训练 AE | AE checkpoint | data/abides_tiny/ |
| 评估重建质量 | 重建误差报告 | AE checkpoint |
| 确定 d_agent 和 patch_size | 超参数 | 重建误差报告 |

**里程碑 1**：AE 重建误差 < 1%（关键维度），patch 化不丢失个体 agent 可辨识性。

### Phase 2: Diffusion 核心（Week 5-8）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 实现 noise scheduler | scheduler.py | 无 |
| 实现 v-prediction 参数化 | v_prediction.py | scheduler.py |
| 实现 DiT Block（含 adaLN-Zero） | dit_block.py | 无 |
| 实现 Market Token 模块 | market_tokens.py | 无 |
| 实现 Cross-Attention 机制 | attention.py | 无 |
| 组装完整 AgentDiT | agent_dit.py | 上述所有 |
| 实现 DDPM 训练逻辑 | ddpm.py | agent_dit.py |
| 实现 DDIM 推理逻辑 | ddim.py | agent_dit.py |
| 在 tiny 数据上训练 AgentDiT-S | DiT checkpoint | 全部上述 |
| 验证单步生成质量 | 定性结果 | DiT checkpoint |

**里程碑 2**：在 1K agent 上，单步生成的状态分布与 ABIDES 真值的 Wasserstein 距离 < 阈值。

### Phase 3: 约束系统（Week 9-10）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 实现软约束 loss | soft_loss.py | 无 |
| 将软约束加入训练 | 更新 train_diffusion.py | soft_loss.py |
| 实现 Guided Diffusion | guidance.py | ddim.py |
| 实现投影层 | projection.py | 无 |
| 约束 ablation 实验 | 约束违反度对比表 | 全部上述 |

**里程碑 3**：市场出清约束违反度降至 < 1e-6；预算约束违反率 < 0.1%。

### Phase 4: 规模扩展（Week 11-14）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 生成 Phase 1 数据（10K agents） | data/abides_small/ | ABIDES |
| 分布式训练适配 | distributed.py | 无 |
| 在 10K agents 上训练 AgentDiT-B | checkpoint | data + AgentDiT |
| 评估 Stylized Facts | SF 报告 | checkpoint |
| 生成 Phase 2 数据（100K agents） | data/abides_large/ | ABIDES parallel |
| 在 100K agents 上训练 AgentDiT-L | checkpoint | data + 多 GPU |
| 1M agents 推理测试 | 加速比数据 | checkpoint |

**里程碑 4**：10K agents 上 5/6 个 stylized facts 通过；加速比 > 50x。

### Phase 5: 评估与论文（Week 15-18）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 完整 Stylized Facts 评估 | 表格 + 图 | Phase 4 checkpoint |
| ABM 保真度全面评估 | 多维度对比表 | Phase 4 checkpoint |
| 加速比 Benchmark | 性能对比表 | 多规模实验 |
| 涌现现象 case study | 定性分析 + 图 | Phase 4 checkpoint |
| Ablation 实验 | 消融对比表 | 各变体模型 |
| 撰写论文 | LaTeX 初稿 | 全部实验结果 |

---

## 十一、关键依赖与版本

```toml
[project]
name = "agentdiffusion"
version = "0.1.0"
requires-python = ">=3.10"

[project.dependencies]
torch = ">=2.2.0"
torchvision = ">=0.17.0"
einops = ">=0.7.0"
flash-attn = ">=2.5.0"
diffusers = ">=0.27.0"           # HuggingFace diffusers（参考实现）
accelerate = ">=0.27.0"          # 分布式训练
wandb = ">=0.16.0"               # 实验追踪
hydra-core = ">=1.3.0"           # 配置管理
scipy = ">=1.12.0"               # 统计检验
abides-core = ">=0.2.0"          # ABIDES 模拟器
```

---

## 十二、风险与缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|---------|
| AE 重建损失过大导致个体信息丢失 | 中 | 高 | 增大 d_agent；尝试 VQ-VAE；分类型训练 encoder |
| 百万级 token attention 内存溢出 | 高 | 高 | 分层 attention + FlashAttention；必要时增大 patch_size |
| 长时间步滚动推理误差累积 | 高 | 中 | 多步预测损失；定期重新校准；课程学习长度 |
| ABIDES 百万级运行极慢，数据生成瓶颈 | 高 | 中 | 并行多实例；设计轻量级大规模模拟器替代 |
| 约束 guidance 在高噪声时无效 | 中 | 中 | 仅在后半程开启 guidance；投影兜底保证最终结果 |
| Stylized facts 不通过 | 中 | 高 | 增加训练数据多样性；调整 agent 类型比例；检查数据预处理 |

---

## 十三、论文大纲（面向 NeurIPS / ICML / ICLR）

```
Title: AgentDiffusion: Accelerating Million-Scale Agent-Based Financial
       Market Simulation with Latent Diffusion Models

Abstract: ...

1. Introduction
   - ABM 计算瓶颈
   - Diffusion 作为 surrogate 的动机
   - 主要贡献

2. Related Work
   - 金融 ABM (ABIDES, Mesa, etc.)
   - Diffusion Models (DiT, LDM)
   - Neural Surrogates for Simulation
   - Constrained Generation

3. Method
   3.1 Problem Formulation
   3.2 Agent State Representation & Encoding
   3.3 Market-Aware Diffusion Transformer (AgentDiT)
       - Patch-based Latent Diffusion
       - Market Token Cross-Attention
       - adaLN-Zero Conditioning
   3.4 Constraint Satisfaction Framework
       - Training: Soft Loss
       - Inference: Guided Diffusion
       - Post-processing: Projection

4. Experiments
   4.1 Data Generation (ABIDES)
   4.2 Stylized Facts Reproduction
   4.3 ABM Fidelity Metrics
   4.4 Computational Speedup
   4.5 Emergent Phenomena
   4.6 Ablation Studies
       - Latent dim vs Reconstruction
       - Market tokens vs Full attention
       - Constraint layers ablation

5. Discussion & Limitations

6. Conclusion
```
