# CompactFlow

**自进化紧凑工作流构造与受保护的机会式执行**

[English](./README.md) | [简体中文](./README-zh.md)

> **代码审核范围。** 本仓库以
> [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) 作为基础 Agent
> 框架。本工作的新增贡献是下文所述的 CompactFlow 构造面、执行面、
> 策略演化、实验基础设施和测试。核心实现位于
> `evoagentx/compactflow`，不属于上游 EvoAgentX 项目。

CompactFlow 针对 Agent 工作流中的两类效率问题：

1. **构造阶段冗余：** 工作流可能包含不必要的 Agent、边、路由或停止步骤。
2. **执行阶段阻塞：** 下游调用通常等待完整的上游结果，即使自己依赖的精确字段
   已经稳定可用。

本实现通过显式的质量、就绪性、资源和副作用约束同时优化这两个阶段。
CompactFlow 以可选 Sidecar 的形式集成，不替换 EvoAgentX 原有的工作流类和
默认运行时。

## 审核入口

以下路径是本工作的主要新增实现：

| 路径 | 审核内容 |
| --- | --- |
| `evoagentx/compactflow/` | 构造面、策略库与演化、图编译器、Guarded Runtime、EvoAgentX 适配器、指标和实验工具 |
| `examples/compactflow/run_experiments.py` | 可执行的离线 Smoke 实验和配置校验 CLI |
| `examples/compactflow/configs/` | 可运行的 Smoke 配置和不可运行的正式评测协议模板 |
| `examples/compactflow/policies/` | 紧凑性策略种子模板 |
| `tests/src/compactflow/` | 单元、集成、安全性、指标和产物隐私测试 |
| `examples/compactflow/README.md` | 完整 API、不变量、配置和实验说明 |

`evoagentx/` 中其他文件主要来自本实现所依赖的上游框架。这样的目录隔离让
审核者可以直接定位 CompactFlow 贡献，而不会把它和 EvoAgentX 原始代码混淆。

## 我们实现了什么

| 模块 | 已交付行为 | 状态 |
| --- | --- | --- |
| 构造面 | 检索并重排紧凑性策略，选择兼容策略子集，条件化原生 Planner，校验生成的 `WorkFlowGraph`，记录 Planner 和回退 Trace。 | 已实现 |
| 策略演化 | 维护质量-成本 Pareto Archive，严格配对基线与候选证据，执行确定性的 `ADMIT`/`MERGE`/`REJECT`，原子持久化策略库。 | 已实现；候选策略蒸馏模型可插拔 |
| 执行编译器 | 将 EvoAgentX `WorkFlowGraph` 降低为 Guarded Function-Call Relation Graph（GFRG），校验 JSON Schema，解析精确字段路径，合成就绪 Guard。 | 已实现 |
| Guarded Runtime | 处理 `Partial`、`Complete` 和 `Failure` 事件，仅在编译器校验过的显式 Guard 成立时释放消费者，检查副作用和资源约束，传播失败，并保证每个逻辑调用最多派发一次。 | 已实现 |
| 原生集成 | 执行原生 Action Graph 和 Agent 节点；需要流式与提前派发时支持显式 Async Operation 覆盖。 | 已实现 |
| 评测支持 | 提供确定性数据划分、质量/成本/延迟/结构配对指标、就绪性与安全诊断、Trace、匿名 Manifest 和离线 Smoke 实验。 | 已实现 |
| 正式四基准实验 | 在通过校验的配置模板中定义 MBPP、HotPotQA、MATH 和 GAIA 协议。 | 仅协议骨架；未包含正式 Runner 和结果 |

## 架构

```text
                               构造面

 任务 + Schema
      |
      v
 策略检索 --> 适用性/历史效用重排 --> 兼容策略选择
      ^                                  |
      |                                  v
 策略库 <-- ADMIT / MERGE / REJECT <-- 策略条件化 Planner
      ^                                  |
      |                                  v
 配对证据 <-- 质量与成本指标 <-- 已校验 WorkFlowGraph
                                         |
                               执行面    |
                                         v
                              原生图降低
                                  |
                                  v
                  精确字段依赖 + 副作用依赖
                                  |
                                  v
                 Schema 校验 + 稳定字段 Guard 合成
                                  |
                                  v
              Partial / Complete / Failure 事件调度
                                  |
                                  v
             工作流结果 + 延迟/Token/结构/安全 Trace
```

## 构造面实现

### 策略检索与兼容选择

`PolicyLibrary` 使用确定性 JSON 保存带类型的“条件-操作”策略和执行证据。
检索过程依次执行：

1. 语义 top-\(K_0\) 召回；
2. 根据语义相似度、结构适用性、历史效用和置信度重排；以及
3. 选择最多 \(k\) 个彼此兼容的策略，并记录其他候选被跳过的原因。

默认检索只使用 `verified` 策略。仓库附带的种子模板初始状态是
`candidate`，只有显式启用探索时才参与检索，不能把它们当作已经通过基准验证的
策略。

主要代码：

- `evoagentx/compactflow/models.py`
- `evoagentx/compactflow/policy.py`
- `evoagentx/compactflow/construction.py`

### 策略条件化构造

`PolicyGuidedWorkflowGenerator` 包装 EvoAgentX 的 `WorkFlowGenerator`：

- 把已选策略渲染进 Planner 输入；
- 直接生成紧凑工作流；
- 校验 DAG、可达性、节点执行器、带类型接口和图一致性；以及
- 保存 `PlannerTrace`。

条件化规划或校验失败时可以回退到原始 Planner；严格实验可以关闭回退。
主要代码位于 `evoagentx/compactflow/planner.py`。

### 基于证据的策略演化

执行证据按照 `(benchmark, split, task_id, seed)` 严格匹配，而不是按照写入
顺序配对。候选策略使用实验 Runner 指定的留出 Split 上的配对证据进行判断：

```text
质量：  mean(candidate - baseline) >= -epsilon_Q
成本：  mean(normalized cost reduction) >= delta_C
有效性：candidate valid rate >= configured minimum
新颖性：nearest-policy similarity < merge threshold
```

判定规则是确定性的：

- 质量、成本、有效性和新颖性全部通过时 `ADMIT`；
- 前三项通过但候选与已有策略近似重复时 `MERGE`；
- 其他情况 `REJECT`。

本仓库实现了证据配对、Pareto 筛选、判定、合并和持久化。把模型输出蒸馏为
`CompactnessPolicy` 候选的模型接口保持可插拔，不绑定某一个 LLM 服务商。

## 执行面实现

### 原生工作流降低

`lower_workflow_graph` 把 EvoAgentX `WorkFlowGraph` 转换为 GFRG：

- 输入输出 `Parameter` 转为 JSON Schema；
- Producer 与 Consumer 的同名字段转为字段级数据依赖；
- 纯控制边转为副作用顺序依赖；
- Action Graph 节点直接执行；
- Agent 节点作为隔离的单节点工作流执行；以及
- 显式 Operation 可以覆盖以上形式并暴露 Async Stream。

主要代码位于：

- `evoagentx/compactflow/schema.py`
- `evoagentx/compactflow/adapter.py`

### 精确就绪 Guard

只有同时满足以下条件，Consumer 才能基于 Producer 的 Partial 输出提前执行：

1. 依赖声明了精确的源字段和目标字段路径；
2. 该依赖允许 Early Use；
3. Producer 声明单调 `StreamContract`；
4. 精确源字段位于 Contract 的 `stable_fields`；
5. 当前 `Partial` 事件也把该字段标记为稳定；
6. Consumer 声明 `early_safe=True`；
7. Consumer 是可取消的原生 Coroutine 或 Async Generator；以及
8. 副作用顺序和资源谓词成立。

任一条件不满足时保留 Complete Barrier。字段不会做前缀推断：
`document` 稳定不代表 `document.title` 自动稳定。

编译器通过 Metaschema 校验 JSON Schema；运行时进一步校验调用参数、累计
Partial 输出和 Complete 输出。Partial 校验只放宽 `required`，已出现的字段仍需
满足类型、嵌套约束和 `additionalProperties`。

主要代码位于 `evoagentx/compactflow/compiler.py`。

### 事件驱动运行时

`CompactFlowRuntime` 提供三个可直接比较的模式：

| 模式 | 调度行为 |
| --- | --- |
| `sequential` | 同一时刻最多运行一个语义就绪的调用 |
| `complete` | 独立调用并发，但依赖数据必须等待 Producer 完成 |
| `guarded` | 在编译器校验过的显式精确字段 Guard 条件成立时提前释放依赖调用 |

三种模式使用同一个 `Partial`/`Complete`/`Failure` 事件循环。运行时记录调用
状态、参数、输出、相对时间戳、端到端延迟、首次运行时输出时间、标准化 Trace
和违规项，同时保证：

- 每个逻辑调用最多派发一次；
- 失败传播给不可达的后继；
- 向量资源容量得到约束；以及
- 除非显式声明 Partial Effect 安全，否则副作用保留 Complete Barrier。

主要代码位于 `evoagentx/compactflow/runtime.py`。

## 快速运行

安装项目和开发依赖：

```bash
python -m pip install -e ".[dev]"
```

原生 Planner 和 Agent Adapter 集成可能还需要 EvoAgentX 可选依赖：

```bash
python -m pip install -e ".[all,dev]"
```

运行无需网络的实现 Smoke 实验：

```bash
python examples/compactflow/run_experiments.py smoke \
  --output outputs/compactflow/smoke
```

输出包括：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 清洗后的配置、配置摘要哈希、数据集指纹/数量和运行时版本 |
| `records.jsonl` | 严格的 benchmark-task-seed-method 记录 |
| `records.csv` | 同一记录的扁平化导出 |
| `summary.json` | 配对质量、成本、时间和诊断汇总 |
| `traces.json` | 相对时间调度 Trace |

本地 Smoke 的耗时只是实现检查，不是论文实验结果。完整 API 与代码示例见
[CompactFlow 实现说明](./examples/compactflow/README.md)。

## 实验与复现边界

当前仓库可以直接执行：

- 离线构造面/执行面 Smoke 实验；
- 严格配置校验；
- 确定性的 source/validation/target 划分；
- 配对记录聚合；
- 质量、延迟、图结构、就绪性和安全性诊断；
- Token 指标结构和聚合，但不宣称已经实现 LLM Token 采集；以及
- 经过清洗的 Manifest，以及仅含合成数据的 Smoke JSON、JSONL、CSV 和
  Trace 产物。

`examples/compactflow/configs/paper_template.json` 被明确标记为
`"runner": "protocol_template"`、`"runnable": false`。它只描述正式比较协议；
当前 CLI 尚未实现：

- MBPP、HotPotQA、MATH 和 GAIA 端到端编排；
- AFlow、EvoAgentX 等 Baseline 编排；
- 在线 source/validation/target 策略演化循环；
- LLM Token 统计与受控 Replay；
- independent-only 和 percentage-threshold 调度基线；以及
- GAIA Benchmark Adapter。

论文材料也没有给出 Backbone Model、精确数据划分、样本数、Rollout Budget、
Seed、选择权重、准入阈值、资源容量和 Replay 配置。因此本仓库不复制示意表格
数值，也不宣称已经复现论文正式结果。

## 当前边界

- 执行图必须是有限静态 DAG。
- 每个逻辑调用目前只尝试一次；尚未实现 Retry 感知的输入版本失效和补偿副作用。
- Provider/Tool Timeout 需要在 Operation 内设置；Scheduler 没有 Per-call Timeout。
- 尚未实现动态的 Per-item 调用实例化。
- 原生 Agent 节点只返回 Complete；提前重叠需要显式实现 Async Streaming
  Operation 和 `StreamContract`。
- 同步 Operation 在线程中执行且无法安全强制取消，因此不会获得 Partial Data
  或 Partial Effect 的提前派发。
- `ttfo` 表示运行时任意位置的第一次输出，不一定是 Sink 的 First Useful Output。

## 验证

从仓库根目录运行：

```bash
python -m pytest -q tests/src/compactflow
ruff check evoagentx/compactflow \
  examples/compactflow/run_experiments.py \
  tests/src/compactflow
```

只安装 `.[dev]` 时，缺少 EvoAgentX 可选依赖的集成测试会跳过。安装
`.[all,dev]` 后可运行原生 Planner 和 Adapter 的集成测试。

测试覆盖策略持久化与检索、兼容性冲突、配对准入、Schema 与路径校验、安全和
不安全流式场景、At-most-once 派发、失败传播、副作用与资源约束、原生图降低、
指标、配置校验、稳定数据划分和 Manifest 清洗。

## 基础框架与许可证

CompactFlow 构建在开源
[EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) 框架之上。上游框架说明请
查阅其原始仓库。本仓库保留上游许可证，见 [LICENSE](./LICENSE)。
