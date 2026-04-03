# ABIDES 完整参考手册（面向 AgentDiffusion 数据生成）

---

## 一、项目概览

**ABIDES** = Agent-Based Interactive Discrete Event Simulation

由 J.P. Morgan AI Research 开发，BSD-3 开源许可。**仓库于 2025-06-02 归档（只读）**，代码仍可用。

### 三大组件

| 组件 | 说明 |
|------|------|
| **abides-core** | 通用多 agent 离散事件模拟器内核，agent 之间通过带延迟模型的消息系统通信 |
| **abides-markets** | 金融市场扩展——模拟 NASDAQ 交易所、订单簿撮合、多种交易 agent |
| **abides-gym** | OpenAI Gym 封装层，用于 RL 训练 |

### 核心论文

| 论文 | 年份 | arXiv |
|------|------|-------|
| ABIDES: Towards High-Fidelity Market Simulation for AI Research (Byrd, Hybinette, Balch) | 2019 | 1904.12066 |
| ABIDES-Gym: Gym Environments for Multi-Agent DES (Amrouni, Moulin, Vann) | 2021 | 2110.14771 |
| ABIDES-Economist: Agent-Based Simulator with Learning Agents (Dwarakanath, Balch, Vyetrenko) | 2024 | 2402.09563 |

---

## 二、安装方式

```bash
git clone https://github.com/jpmorganchase/abides-jpmc-public
cd abides-jpmc-public
sh install.sh
```

安装脚本会安装三个子包：`abides-core`、`abides-markets`、`abides-gym`。

---

## 三、Agent 类型详解

### 3.1 ExchangeAgent（交易所 agent）

- 角色：中心化撮合引擎，管理每只股票的 OrderBook
- 关键参数：
  - `symbols`: 交易标的列表
  - `book_logging`: 是否记录订单簿快照
  - `book_log_depth`: 记录深度（如 10 档）
  - `stream_history_length`: 订单流历史长度（如 500）
  - `pipeline_delay`: 订单处理延迟
- 提供数据接口：
  ```python
  order_book.get_L1_snapshots()  # 最优买卖价
  order_book.get_L2_snapshots(nlevels=10)  # 10 档盘口
  ```

### 3.2 TradingAgent（基础交易 agent）

所有策略 agent 的父类。核心状态变量：

| 属性 | 类型 | 说明 |
|------|------|------|
| `holdings` | dict | 持仓映射：`{"CASH": cents, "ABM": shares}` |
| `starting_cash` | int | 初始资金（单位：美分） |
| `orders` | dict | 当前挂单：`{order_id: Order}` |
| `executed_orders` | list | 已成交订单列表 |
| `known_bids` / `known_asks` | dict | 最近收到的盘口信息 |
| `last_trade` | dict | 最新成交价 |

核心方法：
```python
place_limit_order(symbol, quantity, side, price)
place_market_order(symbol, quantity, side)
cancel_order(order)
cancel_all_orders()
```

日志事件类型：`ORDER_SUBMITTED`, `ORDER_EXECUTED`, `CANCEL_SUBMITTED`, `HOLDINGS_UPDATED`, `STARTING_CASH`, `FINAL_HOLDINGS`

### 3.3 ValueAgent（基本面 agent）

- 基于 Ornstein-Uhlenbeck 过程估计标的内在价值
- 按泊松过程到达（`lambda_a`），在估计价值附近提交限价单
- 关键参数：
  - `r_bar`: 100,000（基本面均值，单位美分 = $1000）
  - `kappa`: 1.67e-15（均值回归速率）
  - `lambda_a`: 5.7e-12（到达率）
  - `sigma_n`: r_bar/100（噪声标准差）

### 3.4 MomentumAgent（趋势跟踪 agent）

- 监测短期价格趋势，顺势下单
- 关键参数：
  - `min_size` / `max_size`: 1-10 手
  - `wake_up_freq`: 37 秒（唤醒频率）
  - `num_momentum_agents`: 默认 12

### 3.5 NoiseAgent（噪声交易者）

- 随机时间到达，随机方向和大小下单
- 可以比市场开盘早到达（`NOISE_MKT_OPEN = 09:00`）
- `num_noise_agents`: 默认 1000-5000

### 3.6 MarketMakerAgent（做市商 agent）

- 自适应做市策略，在买卖两侧挂多档限价单
- 关键参数：
  - `mm_pov`: 0.025（占市场成交量比例）
  - `mm_num_ticks`: 10（挂单档数）
  - `mm_wake_up_freq`: "60S"
  - `mm_spread_alpha`: 0.75（spread 调整系数）
  - `mm_min_order_size`: 1
  - `mm_level_spacing`: 5（档位间距）
  - `mm_cancel_limit_delay`: 50 ns

---

## 四、预置配置

### RMSC03（5126 agents）

| Agent 类型 | 数量 |
|-----------|------|
| Exchange | 1 |
| Market Maker (POV) | 1 |
| Value | 100 |
| Momentum | 25 |
| Noise | 5,000 |

### RMSC04（1117 agents）

| Agent 类型 | 数量 |
|-----------|------|
| Exchange | 1 |
| Adaptive Market Maker | 2 |
| Value | 102 |
| Momentum | 12 |
| Noise | 1,000 |

配置参数：
```python
seed = int(pd.Timestamp.now().timestamp() * 1_000_000) % (2**32 - 1)
date = "20210205"
end_time = "10:00:00"
ticker = "ABM"
starting_cash = 10_000_000  # 美分 = $100,000
MKT_OPEN = "09:30:00"
MKT_CLOSE = end_time
```

---

## 五、运行模拟 & 提取数据

### 5.1 Python API

```python
from abides_core import abides
from abides_core.utils import parse_logs_df, ns_date, str_to_ns
from abides_markets.configs import rmsc04

# 构建配置
config = rmsc04.build_config(seed=42, end_time="10:00:00")

# 运行模拟
end_state = abides.run(config)

# --- 提取订单簿数据 ---
order_book = end_state["agents"][0].order_books["ABM"]
L1 = order_book.get_L1_snapshots()
L2 = order_book.get_L2_snapshots(nlevels=10)

best_bids = pd.DataFrame(L1["best_bids"], columns=["time", "price", "qty"])
best_asks = pd.DataFrame(L1["best_asks"], columns=["time", "price", "qty"])

# --- 提取 agent 日志 ---
logs_df = parse_logs_df(end_state)
# 筛选噪声 agent 的下单事件
noise_orders = logs_df[
    (logs_df.agent_type == "NoiseAgent") &
    (logs_df.EventType == "ORDER_SUBMITTED")
]

# --- 提取每个 agent 的最终状态 ---
for agent in end_state["agents"]:
    if hasattr(agent, 'holdings'):
        print(agent.name, agent.holdings)
```

### 5.2 命令行

```bash
abides abides-markets/abides_markets/configs/rmsc04.py --end_time "10:00:00" --seed 42
```

### 5.3 Gym 环境

```python
import gym
import abides_gym

env = gym.make("markets-daily_investor-v0", background_config="rmsc04")
env.seed(0)
state = env.reset()
state, reward, done, info = env.step(0)
```

---

## 六、Oracle（基本面价值过程）

ABIDES 使用 Oracle 对象生成标的资产的"真实"基本面价值路径：

```python
# Oracle 参数
kappa_oracle = 1.67e-16    # 基本面均值回归速率
sigma_s = 0                 # 基本面波动率
fund_vol = 5e-5             # 基金波动率
megashock_lambda_a = 2.77778e-18  # 极端事件到达率
megashock_mean = 1000       # 极端事件均值偏移
megashock_var = 50_000      # 极端事件方差
```

Oracle 为 ValueAgent 提供"内在价值"参考，agent 在此基础上加入噪声形成自己的估价。

---

## 七、延迟模型

ABIDES 的通信系统支持逼真的延迟建模：

- 每条消息在 agent 之间传递时可以附加随机延迟
- `computation_delay`: 每条消息处理延迟（默认 50 ns）
- 通过 `latency_model` 生成器自动为 N 个 agent 构建延迟矩阵

---

## 八、面向 AgentDiffusion 的数据生成方案

### 8.1 需要提取的 Agent 状态向量

从每个 TradingAgent 的内部状态提取 128 维向量：

```python
def extract_agent_state(agent, exchange_agent, symbol="ABM"):
    """从单个 agent 提取 128 维状态向量"""
    state = np.zeros(128)

    # [0:32] 持仓信息
    state[0] = agent.holdings.get(symbol, 0)        # 股票持仓
    state[1] = agent.holdings.get("CASH", 0)         # 现金

    # [32:48] 资金状态
    state[32] = agent.holdings.get("CASH", 0)         # 现金
    state[33] = ...                                    # 杠杆等

    # [48:64] 策略参数
    state[48] = agent_type_encoding                    # one-hot

    # [64:80] 历史统计
    # 从 executed_orders 计算近期收益、波动率等

    # [80:96] 行为特征
    # 从日志计算下单频率、撤单率

    # [96:112] 市场观察
    state[96] = agent.last_trade.get(symbol, 0)
    state[97:107] = L2_bid_prices[:10]                 # 10 档买价

    # [112:128] 社交/信息因子
    # 自定义

    return state
```

### 8.2 批量数据生成脚本框架

```python
from abides_core import abides
from abides_markets.configs import rmsc04
import torch, numpy as np

def generate_abides_dataset(
    num_simulations=100,
    seed_start=0,
    end_time="10:00:00",
    snapshot_interval_ns=60_000_000_000,  # 每 60 秒一个快照
    output_dir="data/abides_real",
):
    os.makedirs(output_dir, exist_ok=True)
    sample_idx = 0

    for sim in range(num_simulations):
        seed = seed_start + sim
        config = rmsc04.build_config(seed=seed, end_time=end_time)
        end_state = abides.run(config)

        # 提取时间序列快照
        order_book = end_state["agents"][0].order_books["ABM"]
        L1 = order_book.get_L1_snapshots()

        # 提取所有 trading agent 的最终状态
        agents = [a for a in end_state["agents"] if hasattr(a, 'holdings')]

        # 提取日志
        logs_df = parse_logs_df(end_state)

        # 按时间窗口切片，生成 (S_t, S_{t+1}) 对
        # ... 构建 grid, 保存 ...

        for t in range(num_snapshots - 1):
            torch.save({
                "state_t": grid_states[t],
                "state_t1": grid_states[t+1],
                "market_cond": market_conditions[t],
                "agent_types": agent_types_grid,
            }, f"{output_dir}/sample_{sample_idx:06d}.pt")
            sample_idx += 1
```

### 8.3 规模扩展策略

RMSC04 默认只有 ~1100 agents。扩展到万级/百万级的方案：

| 方案 | 做法 | 优缺点 |
|------|------|--------|
| **等比放大** | 将各类型 agent 数量按比例放大（如 10x → 11170 agents） | 最简单，但 ABIDES 速度会极慢 |
| **多实例拼接** | 跑多个独立 RMSC04 模拟（不同 seed），合并为大网格 | 无跨实例交互，但统计性质保留 |
| **混合参数** | 每次模拟随机采样 agent 参数（kappa, lambda_a, spread 等），增加多样性 | 推荐：多样性好，适合训练泛化模型 |
| **自定义轻量模拟器** | 放弃 ABIDES 的消息系统，实现简化版 ABM（直接矩阵更新） | 最快，但失去微观结构保真度 |

**推荐方案**：在 1000-5000 agents 规模上用 ABIDES 并行跑 1000-5000 次模拟（不同参数），然后用"多实例拼接+插值"的方式构造大网格训练数据。

---

## 九、关键 API 速查

### abides_core

| 函数/类 | 说明 |
|---------|------|
| `abides.run(config)` | 运行模拟，返回 `end_state` dict |
| `parse_logs_df(end_state)` | 将所有 agent 日志解析为 DataFrame |
| `ns_date(ns)` | 纳秒时间戳 → 日期部分 |
| `str_to_ns("09:30:00")` | 字符串 → 纳秒时间戳 |

### abides_markets.agents

| 类 | 父类 | 说明 |
|----|------|------|
| `ExchangeAgent` | `FinancialAgent` | 交易所撮合引擎 |
| `TradingAgent` | `FinancialAgent` | 所有策略 agent 基类 |
| `ValueAgent` | `TradingAgent` | 基本面交易者 |
| `MomentumAgent` | `TradingAgent` | 趋势跟踪 |
| `NoiseAgent` | `TradingAgent` | 噪声交易 |
| `AdaptiveMarketMakerAgent` | `TradingAgent` | 自适应做市商 |

### OrderBook

| 方法 | 返回 |
|------|------|
| `get_L1_snapshots()` | `{"best_bids": [(time, price, qty)], "best_asks": [...]}` |
| `get_L2_snapshots(nlevels)` | 多档盘口快照 |
| `get_L3_snapshots(nlevels)` | 含每笔订单明细 |

### TradingAgent 状态

| 属性 | 说明 |
|------|------|
| `.holdings` | `{"CASH": int, "ABM": int}` 当前持仓 |
| `.orders` | `{order_id: Order}` 未成交订单 |
| `.executed_orders` | 已成交订单列表 |
| `.last_trade` | `{symbol: price}` 最新成交价 |
| `.known_bids` / `.known_asks` | 最近盘口 |
