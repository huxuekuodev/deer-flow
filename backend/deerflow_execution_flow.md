# DeerFlow 请求执行流程（用户消息 → 最终回复）

> 以下是从一条用户消息进入系统到最终返回给用户的完整执行路径，包括异常、循环、跳转等所有分支。

## 总览：Agent 内循环

```
                    ┌─────────────────────────────────────────────────┐
                    │                                                 │
                    │   Agent 核心循环                                 │
                    │                                                 │
                    │   before_agent                                   │
                    │       ↓                                         │
                    │   ┌─→ wrap_model_call  ←── 有 tool_calls ──┐   │
                    │   │     ↓                                   │   │
                    │   │   before_model                           │   │
                    │   │     ↓                                   │   │
                    │   │   ⚡ LLM                                 │   │
                    │   │     ↓                                   │   │
                    │   │   after_model                            │   │
                    │   │     ↓  ── jump_to: model ──→ 回到 model │   │
                    │   │   wrap_tool_call                         │   │
                    │   │     ↓                                   │   │
                    │   │   🔧 Tool 执行                           │   │
                    │   └──────←────── 返回 tool_calls 结果 ──────┘   │
                    │                                                 │
                    │   after_agent  ←── 没有 tool_calls ──────────    │
                    │                                                 │
                    └─────────────────────────────────────────────────┘
```

---

## 详细执行流程

```
用户发消息 → API 端接收 → 创建/恢复 checkpoint → 启动 Agent run
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 阶段1: before_agent（准备环境）                                  │
│ 方向: 正序（先append先执行）                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ① ThreadDataMiddleware                                          │
│    创建: workspace/ uploads/ outputs 目录                        │
│    ↓ 无条件通过                                                │
│                                                                │
│ ② UploadsMiddleware                                             │
│    检查 state 中是否有上传文件列表                                │
│      ├─ 有 → 上次处理过的文件列表注入 state                      │
│      └─ 无 → 跳过                                              │
│    ↓ 无条件通过                                                │
│                                                                │
│ ③ SandboxMiddleware                                             │
│    获取或创建该线程的沙箱环境                                    │
│      ├─ 成功 → 沙箱 ID 写入 state                               │
│      └─ 失败 → 抛出异常（这会导致 run 失败）                     │
│    ↓ 成功则继续                                                │
│                                                                │
│ ④ DynamicContextMiddleware                                      │
│    读取 memory.json → 格式化长期记忆                             │
│    拼接 <system-reminder><memory>...<current_date>...</>        │
│    注入到第一条 HumanMessage 前                                  │
│    ↓ 无条件通过                                                │
│                                                                │
│ ⑤ TodoMiddleware.before_agent                                   │
│    清理其他 run 残留的完成提醒数据                                │
│    ↓ 无条件通过                                                │
│                                                                │
│ ⑥ LoopDetectionMiddleware.before_agent                          │
│    清理前一个 run 遗留的循环警告                                 │
│    ↓ 无条件通过                                                │
│                                                                │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│ 阶段2: wrap_model_call（包裹 LLM 调用）                          │
│ 方向: 正序（外层先执行，先append的先包裹）                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ① ToolOutputBudgetMiddleware.wrap_model_call                    │
│    检查历史 ToolMessage 中是否有漏网之鱼（超大输出）              │
│      ├─ 有 → 截断到 fallback_max_chars                          │
│      └─ 无 → 跳过                                              │
│    然后调用内层 handler                                         │
│    ↓                                                            │
│                                                                │
│ ② LLMErrorHandlingMiddleware.wrap_model_call                    │
│    ┌─ 执行内层 handler ──────────────────────────────┐          │
│    │  ③ SkillActivationMiddleware                       │        │
│    │    检查消息列表最后一条 HumanMessage：               │        │
│    │      ├─ 以 /skill-name 开头？                      │        │
│    │      │   ├─ skill 存在且 enabled →                  │        │
│    │      │   │  读取完整 SKILL.md                       │        │
│    │      │   │  在用户消息前插入隐藏 HumanMessage       │        │
│    │      │   │  → 继续内层                              │        │
│    │      │   ├─ skill 不存在/disabled →                  │        │
│    │      │   │  返回 AIMessage(错误提示)                 │        │
│    │      │   │  → 跳过 LLM，直接返回                    │        │
│    │      │   └─ 不以 / 开头 → 直接透传                   │        │
│    │      │                                              │        │
│    │      │  ④ TodoMiddleware.wrap_model_call              │        │
│    │      │    检查是否有排队中的 completion_reminder       │        │
│    │      │      ├─ 有 → 追加到消息列表末尾                │        │
│    │      │      └─ 无 → 跳过                            │        │
│    │      │    → 继续内层                                 │        │
│    │      │                                              │        │
│    │      │  ⑤ LoopDetectionMiddleware.wrap_model_call      │        │
│    │      │    检查是否有排队中的循环警告                   │        │
│    │      │      ├─ 有 → 追加 HumanMessage(警告文本)       │        │
│    │      │      └─ 无 → 跳过                            │        │
│    │      │    → 继续内层                                 │        │
│    │      │                                              │        │
│    │      │  ★ 真正的 LLM 调用（同步/异步）              │        │
│    │      │    组装好的消息 → 调用 create_chat_model       │        │
│    │      └──────────────────────────────────────────┘   │        │
│    │                                                      │        │
│    │  外层 handler 返回后：                                 │        │
│      ├─ 正常返回 ModelResponse → 传给内层结果               │        │
│      ├─ 可重试异常 → 指数退避重试（最多3次）                │        │
│      │    等待: 1s → 2s → 放弃                            │        │
│      ├─ 熔断已打开 → 直接返回 Fallback AIMessage           │        │
│      │    "deerflow_error_fallback: True"                 │        │
│      └─ 不可重试异常(额度/鉴权) → 返回 Fallback AIMessage  │        │
│                                                          │        │
│  注: 如果 SkillActivation 直接返回了 AIMessage（不是       │        │
│  调 LLM），则内层直接返回这个 AIMessage，不走 LLM           │        │
│                                                          │        │
└───────────────────────┬────────────────────────────────────┘     │
                        │                                          │
                        ▼                                          │
┌────────────────────────────────────────────────────────────────┐ │
│ 阶段3: before_model（准备 LLM 输入）                             │ │
│ 方向: 正序（先append先执行）                                     │ │
├────────────────────────────────────────────────────────────────┤ │
│                                                                │ │
│ ① DanglingToolCallMiddleware                                    │ │
│    检查消息列表：AIMessage(tool_calls) 后面有没有对应的          │ │
│       ToolMessage？                                              │ │
│      ├─ 有缺失 → 注入占位 ToolMessage(空内容)                    │ │
│      └─ 完整 → 跳过                                            │ │
│    ↓                                                            │ │
│                                                                │ │
│ ② SummarizationMiddleware.before_model                          │ │
│    计算当前消息总 token 数                                       │ │
│    检查 trigger 阈值（如 tokens>=4000 或 messages>=50）          │ │
│      ├─ 未达到 → 跳过，return None                               │ │
│      └─ 达到 → 执行摘要压缩                                     │ │
│          1. 按 keep 策略找 cutoff_index                          │ │
│          2. 技能救援（skill 指令从待压缩移到保留）                 │ │
│          3. 动态上下文救援（记忆/日期从待压缩移到保留）            │ │
│          4. 调用摘要模型生成摘要                                  │ │
│          5. 重建消息列表:                                        │ │
│             RemoveMessage(ALL) + 摘要 HumanMessage + 保留消息    │ │
│    ↓                                                            │ │
│                                                                │ │
│ ③ TodoMiddleware.before_model                                   │ │
│    state["todos"] 有内容？                                      │ │
│      ├─ 无 → 跳过                                               │ │
│      └─ 有 → 检查 write_todos 调用是否还在消息列表中              │ │
│          ├─ 还在 → LLM 自己能看到 todo → 跳过                    │ │
│          ├─ 已被压缩掉 → 注入 todo_reminder HumanMessage          │ │
│          │   "你的 todo 列表不再可见，但仍在进行中..."             │ │
│          └─ 已有 todo_reminder → 不重复插入                      │ │
│    ↓                                                            │ │
│                                                                │ │
│ ④ ViewImageMiddleware                                           │ │
│    检查消息中是否包含图片？                                      │ │
│      ├─ 有 → 读取图片文件，转为 base64 注入 state                 │ │
│      └─ 无 → 跳过                                              │ │
│    ↓                                                            │ │
│                                                                │ │
└───────────────────────┬────────────────────────────────────────┘ │
                        │                                          │
                        ▼                                          │
                  ╔═══════════════╗                                │
                  ║   LLM 调用    ║                                │
                  ╚═══════════════╝                                │
                        │                                          │
                        ▼                                          │
┌────────────────────────────────────────────────────────────────┐ │
│ 阶段4: after_model（处理 LLM 输出）                              │ │
│ 方向: 逆序（后append的先执行）                                   │ │
├────────────────────────────────────────────────────────────────┤ │
│                                                                │ │
│ ① LoopDetectionMiddleware.after_model (最先执行)                │ │
│    获取最后一条 AIMessage 的 tool_calls                          │ │
│      ├─ 没有 tool_calls → 跳过                                  │ │
│      └─ 有 → 计算哈希，放入滑动窗口                              │ │
│                                                                │ │
│      检查结果：                                                  │ │
│      ├─ count >= 5（硬停阈值）                                   │ │
│      │    → 清除 tool_calls                                     │ │
│      │    → 替换 content 为 [FORCED STOP]                       │ │
│      │    → finish_reason 改为 "stop"                            │ │
│      │    → 返回更新后的 AIMessage                               │ │
│      │    → ★ 后续的 after_model 继续执行，但 tool_calls=[]      │ │
│      │                                                          │ │
│      ├─ count >= 3（警告阈值）且未警告过                          │ │
│      │    → 排队警告到 pending_warnings                          │ │
│      │    → 返回 None（不修改 state）                             │ │
│      │                                                          │ │
│      └─ count < warn → 跳过                                     │ │
│    ↓                                                            │ │
│                                                                │ │
│ ② SubagentLimitMiddleware.after_model                           │ │
│    检查 task 工具调用数量                                        │ │
│      ├─ task 数量 <= 3 → 跳过                                   │ │
│      └─ task 数量 > 3 → 截断                                    │ │
│          保留前 3 个 task，丢弃后面的                             │ │
│          → 返回替换后的 AIMessage（只有 3 个 task）               │ │
│          → ★ 被丢弃的 task 静默消失，LLM 不知道                  │ │
│    ↓                                                            │ │
│                                                                │ │
│ ③ TokenUsageMiddleware.after_model                              │ │
│    1. 反向扫描 ToolMessage 链                                    │ │
│       查找子任务缓存的 token 用量                                 │ │
│       合并到对应的 AIMessage 的 usage_metadata                    │ │
│    2. 读最后一条 AIMessage 的 usage_metadata                      │ │
│       输出日志: "LLM token usage: input=xxx output=xxx total=xxx"│ │
│    3. 构建 step_attribution 写入 additional_kwargs               │ │
│    ↓ （不修改消息，只记录日志+标注）                              │ │
│                                                                │ │
│ ④ TitleMiddleware.after_model                                   │ │
│    首次触发时（1 条 Human + 1 条 AI）                            │ │
│      ├─ 异步 → 调用 LLM 生成标题                                │ │
│      │   失败 → 截取用户前 50 字符做兜底                        │ │
│      └─ 同步 → 直接走兜底标题                                   │ │
│    非首次 → 跳过                                                │ │
│    ↓                                                            │ │
│                                                                │ │
│ ⑤ TodoMiddleware.after_model (最后执行)                         │ │
│    检查 3 个条件：                                               │ │
│      1. AIMessage 没有 tool_calls（模型想结束）                   │ │
│      2. state["todos"] 还有未完成项                              │ │
│      3. 该 run 还没被拦过 2 次                                   │ │
│                                                                │ │
│    ├─ 三个条件都满足 →                                           │ │
│    │  排队 completion_reminder                                   │ │
│    │  return {"jump_to": "model"} ← ★ 关键跳转                   │ │
│    │  → LangGraph 跳回 model 节点                                │ │
│    │  → 重新经过 wrap_model_call（夹带提醒）                      │ │
│    │  → before_model → LLM → after_model（再次 Todo 检查）       │ │
│    │                                                             │ │
│    └─ 不满足 → return None（正常结束 after_model）                │ │
│                                                                  │ │
└───────────────────────┬──────────────────────────────────────────┘ │
                        │                                            │
                        ▼                                            │
┌──────────────────────────────────────────────────────────────────┐ │
│ 分支判断: 阶段4的结果                                            │ │
├──────────────────────────────────────────────────────────────────┤ │
│                                                                  │ │
│   ★ 如果某个 after_model 返回了 {"jump_to": "model"}             │ │
│   → LangGraph 跳过 wrap_tool_call 和 Tool 节点                    │ │
│   → 直接回到 wrap_model_call（阶段2）                              │ │
│   → 注意: 此时不经过 before_agent 和 before_model                 │ │
│                                                                  │ │
│   ★ 如果跳转来源是 TodoMiddleware.after_model →                   │ │
│   阶段5、6全部跳过 → 回到阶段2                                      │ │
│                                                                  │ │
│   ★ 否则（正常路径）：                                            │ │
│   → 进入 wrap_tool_call（阶段5）                                   │ │
│                                                                  │ │
└───────────────────────┬──────────────────────────────────────────┘ │
                        │                                            │
                        ▼                                            │
┌────────────────────────────────────────────────────────────────┐   │
│ 阶段5: wrap_tool_call（包裹工具调用）                            │   │
│ 方向: 正序（外层先执行）                                        │   │
├────────────────────────────────────────────────────────────────┤   │
│                                                                │   │
│ 注意: 这个阶段对每个 tool_call 单独执行一次                       │   │
│ 如果 AIMessage 有 5 个 tool_calls，这个流程走 5 遍              │   │
│ LangGraph 用 asyncio.gather 并发启动这 5 路                     │   │
│                                                                │   │
│ ┌─ 每个 tool_call 的执行路径 ──────────────────────┐            │   │
│ │                                                    │          │   │
│ │ ① ToolOutputBudgetMiddleware.wrap_tool_call         │          │   │
│ │    工具返回后检查大小                                │          │   │
│ │      ├─ 超过 12000 字符 → 外部化到文件              │          │   │
│ │      │  替换为: "结果已保存到 .tool-results/xxx"     │          │   │
│ │      │  + 预览前 2000 字符 + 后 1000 字符          │          │   │
│ │      │  + 总行数提示                                │          │   │
│ │      ├─ 超过 30000 字符（无法保存）→ head+tail 截断  │          │   │
│ │      └─ 小于阈值 → 跳过                             │          │   │
│ │    → 调用内层 handler                               │          │   │
│ │                                                    │          │   │
│ │ ② GuardrailMiddleware.wrap_tool_call（可选）          │          │   │
│ │    检查工具名是否在拒绝名单中                         │          │   │
│ │      ├─ 被拒绝 → 返回 ToolMessage(error)             │          │   │
│ │      │  不调用 handler                              │          │   │
│ │      │  → ★ 后续中间件不执行，直接返回                │          │   │
│ │      └─ 允许 → 调用内层 handler                     │          │   │
│ │                                                    │          │   │
│ │ ③ SandboxAuditMiddleware.wrap_tool_call               │          │   │
│ │    仅限 bash 工具                                    │          │   │
│ │      ├─ 不是 bash → 直接透传                        │          │   │
│ │      ├─ 空命令 → 拦截，返回 ToolMessage(block)       │          │   │
│ │      ├─ 超过 10000 字符 → 拦截返回 block             │          │   │
│ │      ├─ 含 null 字节 → 拦截返回 block                │          │   │
│ │      ├─ 高危命令（rm -rf / 等）→ 拦截返回 block      │          │   │
│ │      ├─ 警告命令（危险但不致命）→ 执行后追加 ⚠️ 提示  │          │   │
│ │      └─ 安全命令 → 正常执行                          │          │   │
│ │    写入审计日志                                      │          │   │
│ │    → 调用内层 handler                               │          │   │
│ │                                                    │          │   │
│ │ ④ ToolErrorHandlingMiddleware.wrap_tool_call           │          │   │
│ │    try:                                              │          │   │
│ │      result = handler(request)                       │          │   │
│ │                                                      │          │   │
│ │      ├─ result 是正常 ToolMessage                    │          │   │
│ │      │  → 如果是 task 工具 → 打 subagent_status 标记   │          │   │
│ │      │  → 返回                                       │          │   │
│ │      │                                                │          │   │
│ │      ├─ result 是 Command(goto=...)                   │          │   │
│ │      │  → 返回（LangGraph 控制流）                     │          │   │
│ │      │                                                │          │   │
│ │      └─ result 是 GraphBubbleUp（中断信号）             │          │   │
│ │         raise（透传）                                  │          │   │
│ │                                                        │        │
│ │    except Exception as exc:                            │        │
│ │      构建 ToolMessage(status="error", content=错误详情) │        │
│ │      如果是 task 工具 → 打 subagent_error 标记          │        │
│ │      return 错误 ToolMessage                           │        │
│ │      → ★ run 不会崩溃，LLM 看到错误后自主决定             │        │
│ │    → 调用内层 handler                                 │        │
│ │                                                    │          │   │
│ │ ⑤ ClarificationMiddleware.wrap_tool_call（最内层）      │        │
│ │    工具名 == "ask_clarification"?                     │          │   │
│ │      ├─ 是 → 不调用 handler                           │          │   │
│ │      │  格式化问题消息                                 │          │   │
│ │      │  return Command(update={...}, goto=END)         │          │   │
│ │      │  → ★ Agent 循环被中断                          │          │   │
│ │      │  → 状态保存 checkpoint                         │          │   │
│ │      │  → 等待用户回复                                │          │   │
│ │      │  → 用户回答后从 checkpoint 恢复                  │          │   │
│ │      │  → 重新从阶段1开始                              │          │   │
│ │      └─ 不是 → handler(request) 实际执行工具           │          │   │
│ │                                                    │          │   │
│ │ ★ 最终结果: ToolMessage / Command                    │          │   │
│ └────────────────────────────────────────────────────┘          │   │
│                                                                │   │
│   LangGraph 收集所有 tool_calls 的结果                           │   │
│   asyncio.gather 等待所有路都走完                                │   │
│                                                                │   │
└───────────────────────┬────────────────────────────────────────┘   │
                        │                                            │
                        ▼                                            │
┌──────────────────────────────────────────────────────────────────┐ │
│ 分支判断: 工具执行结果                                           │ │
├──────────────────────────────────────────────────────────────────┤ │
│                                                                  │ │
│ 检查本轮所有 tool_calls 的结果：                                   │ │
│                                                                  │ │
│ ├─ 有任一个是 Command(goto=END) → 中断（来自 Clarification）      │ │
│ │  → 保存 checkpoint → 等用户回复                                 │ │
│ │                                                                  │
│ ├─ 有任一个是 Command(goto=...) 但非 END → 按 Command 跳转        │ │
│ │                                                                  │ │
│ ├─ ClarificationMiddleware 被拦截 → goto=END → 中断               │ │
│ │                                                                  │ │
│ └─ 全部正常返回 ToolMessage → 追加到 state.messages                │ │
│    → 检查本轮 AIMessage 中是否还有更多 tool_calls？                 │ │
│                                                                  │ │
│     ★ 如果是同一个 AIMessage 还有 tool_calls 没执行完               │ │
│     → 再次经过阶段5（wrap_tool_call），但不再经过 after_model       │ │
│     → 因为 after_model 只处理"新的 AIMessage 生成"，               │ │
│       这里的 AIMessage 是上一轮生成的                               │ │
│                                                                  │ │
│     ★ 如果所有 tool_calls 都执行完了                               │ │
│     → 检查最新的 AIMessage 是否还有 tool_calls 意要执行？           │ │
│     → 有 → 回到阶段2 wrap_model_call（新的一轮 LLM 调用）          │ │
│     → 没有 → 进入阶段6 after_agent                                │ │
│                                                                  │ │
└───────────────────────┬──────────────────────────────────────────┘ │
                        │                                            │
                        ▼                                            │
┌────────────────────────────────────────────────────────────────┐   │
│ 阶段6: after_agent（收尾清理）                                  │   │
│ 方向: 逆序（后append的先执行）                                  │   │
├────────────────────────────────────────────────────────────────┤   │
│                                                                │   │
│ ① MemoryMiddleware.after_agent (最先执行)                       │   │
│    过滤消息：只保留 HumanMessage + 无 tool_calls 的 AIMessage    │   │
│    检测纠正信号（"不对"、"重新来"等）                            │   │
│    检测强化信号（"对就是这样"、"继续保持"等）                    │   │
│    入队 MemoryUpdateQueue（全局单例）                           │   │
│    → 后台 30 秒防抖后：                                         │   │
│      读取 memory.json → 调用 LLM → 解析 JSON 更新               │   │
│    → return None（不修改 state）                                │   │
│    ↓                                                            │   │
│                                                                │   │
│ ② TodoMiddleware.after_agent                                    │   │
│    清理当前 run 的完成提醒内存数据                                │   │
│    → return None                                                │   │
│    ↓                                                            │   │
│                                                                │   │
│ ③ LoopDetectionMiddleware.after_agent (最后执行)                 │   │
│    清理当前 run 的 pending_warnings                              │   │
│    → return None                                                │   │
│    ↓                                                            │   │
│                                                                │   │
└───────────────────────┬────────────────────────────────────────┘   │
                        │                                            │
                        ▼                                            │
               ┌──────────────────┐                                 │
               │  用户收到回复     │                                  │
               │  本次 Agent run   │                                 │
               │  完全结束         │                                  │
               └──────────────────┘                                 │
```

## 循环路径汇总

| 路径 | 触发条件 | 从哪跳到哪 | 绕过哪些阶段 |
|------|---------|-----------|-------------|
| **LLM 重试** | LLM 调用失败且可重试 | 阶段2 内部循环（最多 3 次） | 不绕，只是重新调 LLM |
| **Todo 强制继续** | after_model 检测到未完成的 todo | 阶段4 → 阶段2（wrap_model_call） | 跳过阶段5、6 |
| **同一批 tool 多轮** | 一个 AIMessage 有多个 tool_calls 需要多轮执行 | 阶段5 之后再次阶段5 | 跳过 after_model |
| **新轮次（有 tool）** | LLM 这次回复带了新的 tool_calls | 阶段5 → 阶段2（新的 wrap_model_call） | 跳过阶段6 |
| **新轮次（无 tool）** | LLM 这次回复没有 tool_calls | 阶段5 → 阶段6 → 结束 | 正常 |
| **Clarification 中断** | 模型调用了 ask_clarification | 阶段5 → 保存 checkpoint → 等用户回复 | 中断整个循环 |
| **Loop 硬停** | 相同 tool_calls 出现 5 次 | 阶段4 清空 tool_calls → 进入剩下的阶段 | 不跳转，只清空内容 |

## 异常处理路径

| 异常位置 | 处理方式 | 结果 |
|---------|---------|------|
| **LLM 调用异常（可重试）** | LLMErrorHandlingMiddleware 指数退避重试 3 次 | 要么成功，要么最终返回错误 AIMessage |
| **LLM 调用异常（不可重试）** | 同上，不重试直接返回错误 AIMessage | run 继续，LLM "假装"回复了一条错误消息 |
| **熔断已打开** | 不调 LLM，直接返回"服务繁忙" AIMessage | run 继续，但不消耗真实 LLM 配额 |
| **工具执行异常** | ToolErrorHandlingMiddleware 捕获 → 转 ToolMessage(status="error") | run 继续，LLM 看到错误后决定下一步 |
| **Sandbox 安全拦截** | 被识别为高危命令 → 返回 block ToolMessage | bash 没执行，但 run 继续 |
| **Guardrail 拦截** | 工具在拒绝名单中 → 返回 error ToolMessage | 不执行工具，但 run 继续 |
| **ThreadData 创建目录失败** | 抛出异常 | run 失败 |
| **Sandbox 获取失败** | 抛出异常 | run 失败 |
