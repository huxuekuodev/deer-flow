# 01-project-structured

## 项目目录结构

```md
deer-flow/
├── config.example.yaml               # 主配置模板
├── extensions_config.example.json     # MCP、Skills 等扩展配置模板
├── Makefile                          # 根构建命令
│
├── scripts/
│   ├── docker.sh                     # Docker 管理脚本
│   ├── serve.sh                      # 本地服务启动脚本
│   ├── doctor.py                     # 环境诊断
│   └── wizard/                       # 配置向导
│
├── docker/
│   ├── docker-compose.yaml           # Docker Compose 配置
│   ├── docker-compose-dev.yaml       # 开发环境 Compose 配置
│   └── nginx/
│       ├── nginx.conf                # Docker 环境 Nginx 配置
│       └── nginx.local.conf          # 本地开发 Nginx 配置
│
├── backend/                         # 后端工程
│   ├── app/
│   │   ├── gateway/                 # FastAPI Gateway
│   │   └── channels/                # IM 渠道集成
│   ├── packages/
│   │   └── harness/
│   │       └── deerflow/            # DeerFlow Python 核心包
│   │           ├── agents/          # Lead Agent、状态、记忆与中间件
│   │           ├── subagents/       # 子代理注册与执行
│   │           ├── tools/           # 内置工具与工具搜索
│   │           ├── mcp/             # MCP 客户端、缓存与工具适配
│   │           ├── skills/          # Skill 加载、解析、安装与校验
│   │           ├── sandbox/         # Sandbox 抽象、本地实现与工具
│   │           ├── community/       # Tavily、Jina、Exa 等社区能力
│   │           ├── config/          # 配置模型与加载逻辑
│   │           ├── runtime/         # RunManager、Store、StreamBridge
│   │           ├── models/          # 模型工厂与供应商适配
│   │           ├── guardrails/      # 护栏实现
│   │           └── uploads/         # 上传文件管理
│   ├── tests/                       # 后端测试
│   ├── docs/                        # 后端文档
│   ├── langgraph.json               # LangGraph 运行配置
│   └── Makefile
│
├── frontend/                        # Next.js 前端应用
│   ├── src/
│   │   ├── app/                     # Next.js App Router
│   │   ├── components/              # React 组件
│   │   ├── core/                    # API、消息、线程、模型等业务核心
│   │   ├── hooks/                   # React Hooks
│   │   ├── server/                  # 服务端能力
│   │   └── styles/                  # 全局样式
│   ├── tests/                       # 前端测试
│   └── Makefile
│
└── skills/                          # Agent Skills
    └── public/                      # 内置公共 Skills
```

### 核心模块讲解

#### Gateway API

职责： 提供 RESTful API，处理非 Agent 操作。
端口： 8001
路由：

| 文件                       | 路由                                     | 说明                |
| ------------------------ | -------------------------------------- | ----------------- |
| `routers/models.py`      | `/api/models`                          | 模型列表与配置           |
| `routers/mcp.py`         | `/api/mcp`                             | MCP Server 管理     |
| `routers/memory.py`      | `/api/memory`                          | 全局记忆管理            |
| `routers/skills.py`      | `/api/skills`                          | Skills 注册与状态      |
| `routers/uploads.py`     | `/api/threads/{thread_id}/uploads`     | 文件上传              |
| `routers/threads.py`     | `/api/threads/{thread_id}`             | Thread 文件系统数据清理   |
| `routers/artifacts.py`   | `/api/threads/{thread_id}/artifacts`   | 生成物服务             |
| `routers/suggestions.py` | `/api/threads/{thread_id}/suggestions` | 后续建议              |
| `routers/agents.py`      | `/api/agents`                          | 自定义 Agent 配置      |
| `routers/runs.py`        | `/api/runs`                            | 无状态 Run 执行        |
| `routers/thread_runs.py` | `/api/threads/{thread_id}/runs`        | Thread 级 Run 生命周期 |
| `routers/channels.py`    | `/api/channels`                        | IM 渠道管理           |

#### Agent model

职责：定义Agent状态、中间件、技能等核心组件。

| 文件                | 介绍                               |
| ----------------- | -------------------------------- |
|  ` factory.py`    | create\_deerflow\_agent() SDK级入口 |
| `features.py`     | RuntimeFeatures 声明               |
| `agent.py`        | make\_lead\_agent() 应用入口         |
| `subagents.py`    | 子代理注册与执行                         |
| `thread_state.py` | Agent 状态机                        |
| `middleware.py`   | 记忆、总结、沙箱、工具错误处理等中间件              |
| `memory`          | 全局记忆管理                           |

#### subagents

职责：管理子代理的注册、执行与状态维护。

| 文件            | 介绍      |
| ------------- | ------- |
| `config.py`   | 子代理配置模型 |
| `executor.py` | 子代理执行器  |
| `registry.py` | 子代理注册表  |
| `builtins/`   | 内置子代理   |

#### skills

职责：管理Agent的技能，包括加载、解析、安装与校验。

| 文件                    | 介绍          |
| --------------------- | ----------- |
| `loader.py`           | Skill 加载器   |
| `parser.py`           | SKILL.md 解析 |
| `manager.py`          | Skill 管理    |
| `installer.py`        | Skill 安装    |
| `validation.py`       | Skill 校验    |
| `security_scanner.py` | 安全扫描        |
| `types.py`            | 类型定义        |

#### sandbox

职责：提供沙箱环境，用于执行Agent的技能。

| 文件                          | 介绍                    |
| --------------------------- | --------------------- |
| `sandbox.py`                | Sandbox 抽象            |
| `sandbox_provider.py`       | SandboxProvider 抽象    |
| `middleware.py`             | Agent Sandbox 中间件     |
| `tools.py`                  | 沙箱工具                  |
| `search.py`                 | 沙箱内搜索                 |
| `security.py`               | 安全策略                  |
| `local/`                    | 本地 Sandbox 实现         |
| `local_sandbox.py`          | 本地 Sandbox 实现         |
| `local_sandbox_provider.py` | 本地 SandboxProvider 实现 |

#### frontend

职责：提供用户界面，与后端 API 交互。

| 文件          | 介绍                          |
| ----------- | --------------------------- |
| src/        | 前端应用源代码                     |
| app/        | Next.js App Router          |
| components/ | React 组件                    |
| core/       | API、业务状态、消息、线程、Skills 等核心逻辑 |
| hooks/      | 自定义 Hooks                   |
| server/     | 服务端能力                       |
| styles/     | 样式                          |
| lib/        | 通用工具                        |
| tests/      | 前端测试                        |
| public/     | 静态资源                        |

<br />

### 配置体系

> **🏢 企业级建议**：生产环境避免使用 `.env` 文件管理 secrets，建议使用 Vault、AWS Secrets Manager 或 Kubernetes Secrets 等专用 secret 管理工具。

#### config.yaml 主配置

> deer-flow/config.yaml

```yaml
models:
  - name: gpt-4
    use: langchain_openai:ChatOpenAI
    model: gpt-4

sandbox:
  use: deerflow.community.aio_sandbox:AioSandboxProvider
```

#### extensions_config.json - 扩展配置

```json
{
  "mcp_servers": [...],
  "skills": {...}
}
```

### DeerFlow 二开建议

基于 DeerFlow 结构，可以在以下位置扩展：

| 扩展点         | 位置                                                         | 说明                                                 |
| -------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **Agent 类型** | `backend/packages/harness/deerflow/subagents/`               | 新增专业 Agent                                       |
| **Skill**      | `skills/`                                                    | 企业专用能力                                         |
| **Middleware** | `backend/packages/harness/deerflow/agents/middlewares/`      | 添加审批、检查逻辑                                   |
| **Memory**     | `backend/packages/harness/deerflow/agents/memory/`           | JSON profile/facts、LLM 更新、自定义 `MemoryStorage` |
| **Channel**    | `backend/app/channels/` 与 `backend/app/gateway/routers/channels.py` | 企业 IM 集成                                         |
