# Agent core 

​	本章节介绍DeerFlow的Langgraph编排，中间件链的设计模式与扩展方法。

## 核心入口：make_lead_agent

> 入口：packages/harness/deerflow/agents/lead_agent/agent.py

**该函数完成:**

1. 动态模型选择（是否支持思考模式）
2. 工具加载
3. 系统Prompt生成(包含 SKILL、Memory、subagent指令)
4. 中间件链的组装

```python
 raw_tools = get_available_tools(
            model_name=model_name,
            subagent_enabled=subagent_enabled,
            app_config=resolved_app_config,
        ) + [setup_agent]
  			# 
        filtered = filter_tools_by_skill_allowed_tools(
            raw_tools, skills_for_tool_policy
        )
        final_tools, setup = assemble_deferred_tools(
            filtered, enabled=resolved_app_config.tool_search.enabled
        )
        # 创建Agent
        return create_agent(
            model=create_chat_model(
                name=model_name,
                thinking_enabled=thinking_enabled,
                app_config=resolved_app_config,
                attach_tracing=False,
            ),
            tools=final_tools,
          	# 中间件加载
            middleware=build_middlewares(
                config,
                model_name=model_name,
                available_skills=set(_BOOTSTRAP_SKILL_NAMES),
                app_config=resolved_app_config,
                deferred_setup=setup,
            ),
          	# 系统提示词
            system_prompt=apply_prompt_template(
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=max_concurrent_subagents,
                available_skills=set(_BOOTSTRAP_SKILL_NAMES),
                app_config=resolved_app_config,
                deferred_names=setup.deferred_names,
            ),
            state_schema=ThreadState,
        )
```

### get_available_tools 加载工具

