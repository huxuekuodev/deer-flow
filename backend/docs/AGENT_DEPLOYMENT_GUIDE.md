# Agent 服务部署指南：从 FastAPI Lifespan 到蓝绿部署

> 本文完整阐述一个 LangGraph Agent 服务从进程生命周期管理到生产部署的全部环节。覆盖 FastAPI lifespan 机制、Checkpointer 生命周期管理、Graceful Shutdown、节点原子性保证、子 Agent Checkpointer 透传，以及蓝绿部署在生产环境的落地实践。

---

## 目录

1. [问题的起点](#1-问题的起点)
2. [FastAPI Lifespan 与资源生命周期管理](#2-fastapi-lifespan-与资源生命周期管理)
3. [Checkpointer 的生命周期：为什么不是模块级单例](#3-checkpointer-的生命周期为什么不是模块级单例)
4. [Graceful Shutdown：时序的保证](#4-graceful-shutdown时序的保证)
5. [节点原子性：Interrupt After 机制](#5-节点原子性interrupt-after-机制)
6. [子 Agent Checkpointer 透传](#6-子-agent-checkpointer-透传)
7. [蓝绿部署：生产实践](#7-蓝绿部署生产实践)
8. [完整时序综合推演](#8-完整时序综合推演)
9. [总结](#9-总结)

---

## 1. 问题的起点

一个基于 LangGraph 的 Agent 服务，本质上是一个**有状态的长连接计算引擎**。它不像传统的 REST API 那样请求-响应即可结束，而是可能：

- 一次对话运行多个节点（LLM 调用、工具执行、子 Agent 调度）
- 一个节点内部可能是子 Agent，执行时间从几秒到几分钟不等
- 节点之间有 side effect（发送邮件、写入 DB、调用外部 API）
- 状态通过 Checkpointer 持久化到数据库

这就给部署带来了传统服务没有的挑战：

| 场景 | 传统 REST API | Agent 服务 |
|------|--------------|-----------|
| 请求处理时间 | 毫秒到秒级 | 秒到分钟级 |
| 状态 | 无状态 | 有状态（Checkpoint） |
| 原子边界 | HTTP 请求/响应 | 图节点 |
| Side Effect | 通常无或幂等 | 可能非幂等 |
| 进程退出 | 等当前请求完成即可 | 需要等当前节点完成 |

**核心问题**：当需要升级服务时，如何保证正在执行的 Agent 不被粗暴中断，同时新版本能正常接管？

**注意：当我们说一个长任务是说的是一个agent loop，而不是一个节点执行很长。**

---

## 2. FastAPI Lifespan 与资源生命周期管理

### 2.1 Lifespan 机制

FastAPI 的 Lifespan 协议（ASGI Lifespan）是管理应用生命周期的标准方式：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动阶段 ──
    startup_config = get_app_config()
    async with langgraph_runtime(app, startup_config):
        # ── 服务运行阶段 ──
        yield
    # ── 关闭阶段 ──
    # lifespan yield 返回后执行清理
    logger.info("Shutting down API Gateway")
```

执行顺序：

```
启动：
  lifespan() 进入
    → 加载配置
    → langgraph_runtime() 进入（创建资源）
      → yield（服务开始接受请求）
        
关闭（收到 SIGTERM）：
      → yield 返回
    → langgraph_runtime() 退出（清理资源）
  → lifespan() 退出
```

### 2.2 AsyncExitStack：多层资源嵌套管理

`langgraph_runtime` 内部使用 `AsyncExitStack` 管理多个 async context manager 的嵌套生命周期：

```python
@asynccontextmanager
async def langgraph_runtime(app, startup_config):
    async with AsyncExitStack() as stack:
        # 启动顺序（入栈）
        app.state.stream_bridge = await stack.enter_async_context(
            make_stream_bridge(config)
        )
        await init_engine_from_config(config.database)
        app.state.checkpointer = await stack.enter_async_context(
            make_checkpointer(config)     # PostgreSQL AsyncConnectionPool
        )
        app.state.store = await stack.enter_async_context(
            make_store(config)
        )
        # ... 更多资源

        yield  # 服务运行
        # ↓ SIGTERM 后从这里继续

        # 关闭顺序（出栈，LIFO）
        _drain_inflight_runs(run_manager)   # 1. 先等存量 Run 完成
        await close_engine()                # 2. 关 ORM 引擎
        # 3. 自动出栈（LIFO）：
        #    store.__aexit__()
        #    checkpointer.__aexit__()  → pool.close()
        #    stream_bridge.__aexit__()
```

**LIFO 顺序的意义**：

| 关闭顺序 | 操作 | 为什么这个顺序？ |
|---------|------|----------------|
| 1 | Drain in-flight runs | 先等所有 agent 跑完，期间不能关任何依赖 |
| 2 | Close ORM engine | SQLAlchemy 引擎关掉，防止后续写 event |
| 3 | Close checkpointer pool | 等所有 run 不再写 checkpoint 了，再关连接 |
| 4 | Close store | Store 依赖的线程/连接最后关 |

如果顺序颠倒，比如先关 pool 再 drain，就会出现在 drain 期间 run 还要写 checkpoint 但 pool 已不可用的竞态（即 PoolClosed 异常）。

---

## 3. Checkpointer 的生命周期：为什么不是模块级单例

### 3.1 Checkpointer 的本质

LangGraph 的 `Checkpointer` 不是一个普通的 Python 对象——它**持有数据库连接**：

```python
# PostgreSQL 场景
checkpointer = AsyncPostgresSaver(conn=pool)
# pool = AsyncConnectionPool(conn_string, ...)
#          ├── TCP 连接到 PostgreSQL
#          ├── 默认持有 5 个连接
#          └── 需要显式 close() 释放
```

### 3.2 模块级单例的问题

可能的第一直觉是把 Checkpointer 设为模块级单例，编译时直接注入：

```python
# ❌ 模块级单例的问题
from deerflow.runtime.checkpointer.provider import get_checkpointer

_AGENT = _GRAPH_BUILDER.compile(checkpointer=get_checkpointer())
```

这个做法有多个致命问题：

**问题一：无法处理 Async 生命周期**

PostgreSQL Checkpointer 的创建是 async 的，需要 `async with` 管理连接池：

```python
# async_provider.py
async def make_checkpointer(config):
    pool = _build_postgres_pool(conn_string)
    async with pool:                              # ← 需要 async with
        saver = AsyncPostgresSaver(conn=pool)
        await saver.setup()                       # ← 需要 await
        yield saver
```

模块级别无法使用 `async with`，也无法控制 pool 的关闭时机。

**问题二：关闭时序失控**

```python
# 模块级单例的关闭时序：
SIGTERM → uvicorn 停 HTTP → lifespan 退出
  → 没有任何代码能控制 checkpointer 何时关闭
  → Python GC 可能在任何时候回收 pool
  → 但在回收时，asyncio Task 可能还在写 checkpoint
  → PoolClosed ❌
```

**问题三：测试隔离**

每次测试都需要一个干净的 Checkpointer（通常是 `InMemorySaver`）。模块级单例意味着测试之间共享状态，需要手动 reset，容易泄漏。

### 3.3 依赖注入的正确做法

```python
# deps.py — FastAPI 依赖注入
@asynccontextmanager
async def langgraph_runtime(app, startup_config):
    async with AsyncExitStack() as stack:
        app.state.checkpointer = await stack.enter_async_context(
            make_checkpointer(config)
        )
        yield

# Router 中获取
def get_checkpointer(request: Request) -> Checkpointer:
    val = request.app.state.checkpointer
    if val is None:
        raise HTTPException(status_code=503, detail="Checkpointer not available")
    return val
```

**优势**：

| 维度 | 模块级单例 | AsyncExitStack + DI |
|------|-----------|-------------------|
| 生命周期控制 | ❌ 不可控 | ✅ Lifespan 精确控制 |
| 关闭顺序 | ❌ 依赖 GC | ✅ LIFO 顺序保证 |
| 测试隔离 | ❌ 共享状态 | ✅ 每次测试传 InMemorySaver |
| 启动时验证 | ❌ 模块导入时隐式执行 | ✅ Lifespan 明确执行，失败即报错 |
| 多个实例共享池 | ❌ 每个 import 可能独立创建 | ✅ 所有 Run 共享一个池 |

---

## 4. Graceful Shutdown：时序的保证

### 4.1 问题的本质

收到 SIGTERM 时，系统中可能同时存在多个状态：

```
当前节点已开始执行（正在运行）
    │        当前节点已完成，等待调度下一节点
    │            │        Run 刚创建还没开始
    │            │           │
    v            v           v
  [node3]      [checkpoint] [pending]
    │            │           │
    ├─ 允许跑完   ├─ 中断     ├─ 拒绝
    └─ 保证原子性 └─ 保存状态 └─ 返回错误
```

### 4.2 三层拦截机制

```text
层级                  新请求能进来吗？    存量 Run 怎么处理？
─────────────────────────────────────────────────────────────
K8s/Docker            ❌ 切走流量      | —
（编排层）                               |

uvicorn               ❌ 停 HTTP 连接   | 已有 HTTP 请求正常完成
（网络层）                              | （但后台 Task 可能还在跑）

RunManager            ❌ shutting_down | 等当前节点完成 → checkpoint
（业务层）              = True 拒绝注册   | 超时 → cancel + rollback

Checkpointer          —               | all drain 完成后关 pool
（持久化层）                            |
```

### 4.3 `_drain_inflight_runs` 的实现

```python
async def _drain_inflight_runs(run_manager, timeout=30.0):
    # Step 1: 标记 shutting down，拒绝新 run 注册
    run_manager.shutting_down = True

    # Step 2: 对每个 in-flight run，注入"当前节点完成后停止"
    for record in run_manager.active_runs():
        record.interrupt_after_current_node = True

    # Step 3: 等待所有 run 自然走到 checkpoint
    tasks = [r.task for r in run_manager.active_runs() if r.task]
    done, pending = await asyncio.wait(tasks, timeout=timeout)

    # Step 4: 超时后强制取消极少数长节点
    for task in pending:
        task.cancel()
        # 走 except CancelledError → rollback
```

### 4.4 为什么超时不能无限长

```yaml
# Kubernetes 的视角
spec:
  terminationGracePeriodSeconds: 45  # K8s 给的总预算
  # ├── 30s 给 _drain_inflight_runs
  # ├── 10s 给 close_engine + checkpointer cleanup
  # └── 5s  余量
```

即使你设置 `timeout=9999`，K8s 在 `terminationGracePeriodSeconds` 后会发 `SIGKILL`（不可捕获），进程瞬间死亡。**你的 timeout 必须在 `terminationGracePeriodSeconds` 预算之内。**

---

## 5. 节点原子性：Interrupt After 机制

### 5.1 LangGraph 的 Checkpoint 边界

LangGraph 的 checkpoint 是在**节点之间**写入的，不是在节点中间：

```text
        Start
          │
          ▼
      ┌──────┐
      │node1 │ ← LLM 调用
      └──┬───┘
         │ checkpoint-1 写入（node1 完整状态）
         ▼
      ┌──────┐
      │node2 │ ← 工具调用（发送邮件）
      └──┬───┘
         │ checkpoint-2 写入（node2 完整状态）
         ▼
      ┌──────┐
      │node3 │ ← 子 Agent（可能执行几分钟）
      └──┬───┘
         │ checkpoint-3 写入（node3 完整状态）
         ▼
        End
```

**关键保证**：节点要么完整执行并写入 checkpoint，要么不写入。不存在"节点执行了一半"的状态。

### 5.2 传统 `task.cancel()` 的问题

```python
# ❌ 当前代码的问题
task.cancel()
# → asyncio.CancelledError
# → 节点可能正在执行工具调用
# → 邮件发了，数据库写了，但 checkpoint 没写入
# → 恢复时从上一个 checkpoint 重跑
# → 邮件又发一次 ❌
```

### 5.3 正确做法：`interrupt_after`

LangGraph 的 `CompiledStateGraph` 提供了 `interrupt_after` 属性，可以在运行时设置，让图在**当前节点完成后自然停止**：

```python
class CompiledStateGraph:
    interrupt_before: list[str]  # 在这些节点开始前中断
    interrupt_after: list[str]   # 在这些节点完成后中断
```

`interrupt_after = ["*"]` 的意思是：**任意节点完成后，检查是否需要中断，如果是则停止执行并写入 checkpoint。**

```python
# shutdown 时的注入
for record in run_manager.active_runs():
    # 注入中断指令
    agent = record.agent
    agent.interrupt_after = ["*"]
```

Pregel 执行循环中的检查：

```python
# Pregel 执行循环（简化）
while True:
    # ── 从 checkpoint 加载状态 ──
    state = checkpointer.get(thread_id)
    
    # ── 检查是否应该中断 ──
    if next_node in interrupt_after:
        # 写入当前 checkpoint（节点已完成）
        checkpointer.put(state)
        return PENDING  # 等下次 resume
    
    # ── 调度下一个节点 ──
    result = await dispatch_node(next_node, state)
    # 节点内执行不会被中断
    # 工具调用、子 Agent 完整运行
    
    # ── 节点完成，写入 checkpoint ──
    state = apply_result(state, result)
    checkpointer.put(state)
```

**效果对比**：

| | `task.cancel()` | `interrupt_after` |
|--|---------------|-------------------|
| 节点执行 | **可能被中断** | **完整执行** |
| Checkpoint | 可能半写 | **节点完成后完整写入** |
| Side Effect | 重复/不一致 | **一次性** |
| 恢复位置 | 上一个 node 完成时 | **当前 node 完成后** |
| 中断延迟 | 立即 | 等当前节点执行完（最多到节点天然结束） |

### 5.4 如何注入 `interrupt_after`

需要在 `GraphAgent` 上暴露这个能力：

```python
class GraphAgent:
    def __init__(self, config, *, checkpointer=None):
        self.config = config
        self._agent = (
            StateGraph(ThreadState, context_schema=GraphContext)
            .add_node("plan_model_node", plan_model_node)
            .set_entry_point("plan_model_node")
            .compile(checkpointer=checkpointer)
        )

    @property
    def interrupt_after(self):
        return self._agent.interrupt_after

    @interrupt_after.setter
    def interrupt_after(self, value):
        self._agent.interrupt_after = value

    async def astream(self, messages):
        ctx = get_context(self.config, app_config=get_app_config())
        async for state in self._agent.astream(...):
            yield state
```

---

## 6. 子 Agent Checkpointer 透传

### 6.1 问题

LangGraph 的子 Agent（Sub Agent）需要能独立调度内部节点。如果子 Agent 没有自己的 Checkpointer，那么：

```python
# subagent/executor.py — 当前代码
class SubagentExecutor:
    def _create_subagent_graph(self):
        return CompiledStateGraph(
            checkpointer=False  # ← 不继承父 run 的 checkpointer
        )
```

后果：

```text
父 Agent node3 = 子 Agent 执行

子 Agent 内部：
  ├── step1 (LLM)            ← checkpoint 写入的是父 Agent 的 node 边界
  ├── step2 (工具调用)         
  ├── step3 (另一个子调度)     ← 这些状态没有独立持久化
  └── step4 (工具调用)
       ↓
  shutdown 触发，父 Agent 中断
       ↓
  node3 整体回退到上一个 checkpoint
       ↓
  子 Agent 恢复时全部重跑 ❌
```

### 6.2 解决方案：透传 Checkpointer

```python
class SubagentExecutor:
    def _create_subagent_graph(self, parent_checkpointer):
        return CompiledStateGraph(
            checkpointer=parent_checkpointer,  # ← 透传
        )
```

这样：

```text
父 Agent node3 = 子 Agent 执行

子 Agent 内部（共享同一个 checkpointer 存储）：
  ├── step1 (LLM)            ← checkpoint-3.1
  ├── step2 (工具调用)        ← checkpoint-3.2
  ├── step3                  ← checkpoint-3.3
  └── step4                  ← checkpoint-3.4
       ↓
  shutdown 触发，interrupt_after 生效
  → 子 Agent 当前 step 完成后写入 checkpoint-3.x
  → 父 Agent node3 也写入完整的 node3 checkpoint
       ↓
  恢复时从 checkpoint-3.x 继续
  子 Agent 内部不需要重跑 ✅
```

### 6.3 共享 Checkpointer 的依赖注入

子 Agent 的 Checkpointer 应该从父 Run 传递，而不是自己创建：

```python
# worker.py
async def run_agent(bridge, run_manager, record, ctx, ...):
    checkpointer = ctx.checkpointer  # ← app.state 共享的 checkpointer
    ...

    # 创建父 Agent
    agent = GraphAgent(config, checkpointer=checkpointer)

    # 创建子 Agent 时传递同一个 checkpointer
    executor = SubagentExecutor(parent_checkpointer=checkpointer)
```

---

## 7. 蓝绿部署：生产实践

### 7.1 为什么不能滚动升级

滚动升级在这里不适用：

```text
传统滚动升级（3 个节点 A、B、C 依次升级）：

T0: 用户会话在 A 上执行 node3（子 Agent，预计 2 分钟）
T1: 升级开始，A 开始 drain
    → 用户被中断
    → 等 node3 完成（30s guard → cancel）
T2: A 升级到 v2 → 重新加入 LB
T3: 用户恢复，连到 A(v2)，从 checkpoint-2 恢复
    → node3 重新执行（2 分钟）
T4: B 开始 drain ← 用户还在执行 node3！
    → 用户又一次被中断
    同一次会话被中断 2 次 ❌
```

### 7.2 蓝绿部署

蓝绿部署的核心思想：**新旧两套环境同时运行，流量一次性切换。**

```text
升级前（全部 v1）：

         ┌─ LB ─┐
         │      │
    ┌────┴────┐ │
    │ A(v1)   │ │
    │ B(v1)   │ │
    │ C(v1)   │ │
    └─────────┘ │
         │      │
         └──────┘


升级时：

         ┌─ LB ─┐
         │      │
    ┌─ 蓝(v1) ─┐    ┌─ 绿(v2) ─┐
    │ A(v1)    │    │ A(v2)    │  ← 绿环境全新部署
    │ B(v1)    │    │ B(v2)    │
    │ C(v1)    │    │ C(v2)    │
    └──────────┘    └──────────┘

步 1：绿环境部署，连同上一个 PostgreSQL
步 2：绿环境启动并验证
步 3：LB 切换流量到绿环境（秒级）
步 4：蓝环境 drain 存量的 in-flight run → 下线


切换后（全部 v2，蓝环境保留回滚）：

         ┌─ LB ─┐
         │      │
    ┌────┴────┐ │
    │ A(v2)   │ │
    │ B(v2)   │ │
    │ C(v2)   │ │
    └─────────┘ │
         │      │
         └──────┘
```

### 7.3 关键基础设施：共享 PostgreSQL

蓝绿切换成立的前提——**Checkpointer 存储在共享数据库中，不在本地内存或文件**：

```text
✅ 正确架构：

  蓝环境(v1) ───┐
                 ├──▶ PostgreSQL ──┐
  绿环境(v2) ───┘   │ checkpoints │
                    │ threads     │
                    │ runs        │
                    └─────────────┘

  切换后：
  绿环境从同一个 PostgreSQL 读取 checkpoint
  用户连到绿环境后，直接从上次中断的 checkpoint 恢复
```

对比：

| Checkpointer 类型 | 是否支持蓝绿切换 | 恢复位置 |
|------------------|----------------|---------|
| InMemorySaver | ❌ 重启后丢失 | 从头开始 |
| SqliteSaver（本地文件） | ❌ 只有旧环境有 | 需要手动迁移文件 |
| PostgresSaver（共享 DB） | ✅ 多环境共享 | 从 checkpoint 精确恢复 |

### 7.4 蓝绿 drain 完整流程

```python
# 蓝环境切换后的 drain 流程
async def blue_environment_drain():
    # Step 1: LB 已切走流量，不再有新请求
    # Step 2: Uvicorn 停 HTTP
    # Step 3: Lifespan yield 返回

    # Step 4: _drain_inflight_runs
    run_manager.shutting_down = True

    for record in run_manager.active_runs():
        agent = record.agent  # LangGraph CompiledStateGraph
        agent.interrupt_after = ["*"]
        # 当前节点完整执行后写入 checkpoint，自然停止

    # Step 5: 监控等待
    remaining = len(run_manager.active_runs())
    while remaining > 0:
        logger.info(f"蓝环境 drain 中：{remaining} 个 run 等待完成")
        await asyncio.sleep(2)
        remaining = len(run_manager.active_runs())

    # Step 6: 关闭基础设施
    await close_engine()
    # AsyncExitStack 自动关 pool

    logger.info("蓝环境 drain 完成，全部退出")
```

### 7.5 回滚策略

如果绿环境（v2）有问题：

```text
┌─ LB ─┐
│      │
│      ├──▶ 蓝环境(v1) ── 留 30 分钟，不回收到 K8s
│      │                  （ip 不变，pool 不关）
│      │
└──────┘ ← LB 切回蓝环境（秒级）

用户无感知——检测到 v2 不稳定，一键切回
```

### 7.6 与滚动升级的对比

| 维度 | 滚动升级 | 蓝绿部署 |
|------|---------|---------|
| 用户中断次数 | 可能多次（A 升→断，B 升→再断） | **最多一次** |
| Side Effect 风险 | 多次中断 → 多次重跑 → 重复副作用 | 一次中断 → 一次恢复 |
| LLM 成本 | 多次重新调用 | 从 checkpoint 恢复 |
| 回滚速度 | 逐个回滚，数分钟 | **一键切换，秒级** |
| 资源成本 | 不需要额外资源 | 需要**双倍资源** |
| 部署时长 | 逐个升级，总时长 = sum(每个节点 drain) | 绿环境提前部署好，切换秒级 |

---

## 8. 完整时序综合推演

### 8.1 正常运行时

```text
┌─ FastAPI Lifespan ──────────────────────────────────────────┐
│                                                              │
│  async with langgraph_runtime(app, config):                  │
│                                                              │
│    1. make_stream_bridge()     → stream_bridge              │
│    2. init_engine_from_config() → SQLAlchemy engine          │
│    3. make_checkpointer()      → PostgreSQL pool             │
│    4. make_store()             → Store                       │
│    5. RunManager()             → run_manager                 │
│                                                              │
│    → yield                                                    │
│                                                              │
│    服务运行期间：                                              │
│    ┌──────────────────────────────────────────────────┐      │
│    │ 用户请求 → run_agent() → GraphAgent.astream()     │      │
│    │                          ├── plan_model_node      │      │
│    │                          ├── dispatch_node        │      │
│    │                          ├── sub_agent_node       │      │
│    │                          └── ...                  │      │
│    │                                                    │      │
│    │ 每个节点完成后写入 PostgreSQL checkpoint              │      │
│    │ Checkpointer 来自 app.state，所有 Run 共享一个池     │      │
│    └──────────────────────────────────────────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 升级推演

```text
T0: 用户 A 的 Agent 正在执行 node3（子 Agent，预计跑 45s）
    用户 B 的 Agent 正在执行 node1（LLM 调用，预计 3s）
    用户 C 的 Agent 刚完成 node2，等待调度 node3

T1: 绿环境（v2）部署完成，连接同一个 PostgreSQL
    绿环境启动验证通过

T2: LB 切换
    全部流量从蓝环境（v1）切到绿环境（v2）
    蓝环境不再接受新请求

T3: 蓝环境收到 SIGTERM

T4: FastAPI lifespan yield 返回
    langgraph_runtime finally 开始

T5: _drain_inflight_runs(timeout=30)
    ┌─────────────────────────────────────────────────────┐
    │ run_manager.shutting_down = True                     │
    │                                                      │
    │ 对每个 in-flight run：                                │
    │   agent.interrupt_after = ["*"]                      │
    │                                                      │
    │ wait(all_tasks, timeout=30):                         │
    │                                                      │
    │  +2s: 用户 B 的 LLM 调用完成，写入 checkpoint        │
    │       检查 interrupt_after → 图停止                   │
    │                                                      │
    │  +5s: 用户 C 从 checkpoint 恢复                        │
    │       检查 interrupt_after → 下一个节点被拦截         │
    │       图停止，等待 resume                              │
    │                                                      │
    │  +45s: 用户 A 的子 Agent 执行完毕                     │
    │        写入完整的 checkpoint-3                        │
    │        检查 interrupt_after → 图停止                   │
    │                                                      │
    │  30s timeout 到（此时所有 run 已完成）：               │
    │    pending = [] → 全部正常退出                        │
    └─────────────────────────────────────────────────────┘

T6: close_engine() 关闭 SQLAlchemy 引擎

T7: AsyncExitStack.__aexit__
    → store.__aexit__()
    → checkpointer.__aexit__() → PostgreSQL pool.close()
    → stream_bridge.__aexit__()

T8: 蓝环境全部进程退出

T9: 用户 A 连到绿环境（v2）
    → LangGraph 从 checkpoint-3 恢复
    → 从 node3 之后继续执行
    → 不重跑子 Agent，side effect 一致

T10: 用户 B、C 同样从各自的 checkpoint 恢复
    整个过程最多中断一次 ✅
```

---

## 9. 总结

### 核心原则

1. **蓝绿部署，不滚动升级**：3 个（及以上）服务节点同步切换，避免多次中断。

2. **Checkpointer 共享数据库**：PostgreSQL 作为统一 checkpoint 存储，是蓝绿切换和多环境共享状态的基础设施。

3. **`interrupt_after` 而非 `task.cancel()`**：保证当前节点的原子性——节点完整执行完，写入完整 checkpoint，再自然停止。

4. **子 Agent Checkpointer 透传**：父和子使用同一个 Checkpointer 存储，子 Agent 内部每一步也能原子化持久化。

5. **AsyncExitStack LIFO 生命周期**：先 drain run、再关 engine、最后关 pool，保证任何时刻 pool 关闭时已经没有任何 run 在写 checkpoint。

6. **有限的超时窗口**：超时不是用来"等完所有任务"，而是防止进程被 K8s SIGKILL 时数据损坏。超时后 cancel 的极少数长节点，靠 Checkpointer 持久化 + 幂等性兜底。

### 架构总图

```text
┌──────────────┐     ┌──────────────┐
│  蓝环境(v1)   │     │  绿环境(v2)   │
│              │     │              │
│  FastAPI     │     │  FastAPI     │
│    ↓         │     │    ↓         │
│  Lifespan    │     │  Lifespan    │
│    ↓         │     │    ↓         │
│  RunManager  │     │  RunManager  │
│    ↓         │     │    ↓         │
│  GraphAgent  │     │  GraphAgent  │
│    ↓         │     │    ↓         │
│  interrupt_  │     │  checkpoint  │
│  after       │     │  resume      │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
       ┌────────▼────────┐
       │   PostgreSQL    │
       │  ─────────────  │
       │  checkpoints    │
       │  threads        │
       │  runs           │
       └─────────────────┘
```

### 最终观点

> 优雅关闭不是一个"能不能不等"的问题，而是一个"**等待 vs 控制**"的问题。
>
> 你不能等所有任务完成（进程等不起），也不能不等（数据会损坏）。正确的做法是让每个**正在执行的节点完整跑完**，然后自然停下来。这个保证的力度正好等于 LangGraph checkpoint 的原子边界——节点之间。
>
> 再配合蓝绿部署 + 共享 PostgreSQL，用户会话在一次升级中最多中断一次，恢复时精确从上一个 checkpoint 继续，不需要重跑已完成的工作。这就是 Agent 服务在生产环境中可以落地的升级方案。
