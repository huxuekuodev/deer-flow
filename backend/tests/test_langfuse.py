from langfuse import get_client,Langfuse
langfuse = get_client()

langfuse = Langfuse(
            public_key="pk-lf-c33892a2-868b-4166-9186-5debffbe55ce",
            secret_key="sk-lf-613db8fe-7948-42e8-9027-8bdf0d186fad",
            host="http://82.156.254.44:3000",  # Optional, default shown
)
    
# 1. 获取所有评分器
trace_id = langfuse.get_current_trace_id()
with langfuse.start_as_current_observation(
    trace_context={"trace_id": trace_id},
    name="custom_observation",
    input="我们现在有，read_file, web_search工具， 用户问题：今天的天气是什么?",
    output="调用web_search工具"
) as span:
    span.update()