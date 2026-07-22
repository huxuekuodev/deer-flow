import sys
from pathlib import Path

from loguru import logger

from deerflow.config import get_app_config
from deerflow.core.context import trace_id_ctx_var

# 配置日志格式
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <magenta>trace_id - {extra[trace_id]}</magenta> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

app_config = get_app_config()


# 注入 trace_id 到每条日志记录
def inject_trace_id(record):
    record["extra"]["trace_id"] = trace_id_ctx_var.get() or "-"


logger.remove()  # 移除默认输出配置

# 给日志打补丁，使其自动携带 trace_id
logger = logger.patch(inject_trace_id)

# 配置日志输出
if app_config.logging.console.enable:
    logger.add(sink=sys.stdout, level=app_config.logging.console.level.upper(), format=log_format)

# 配置日志文件输出
if app_config.logging.file.enable:
    path = Path(app_config.logging.file.path)
    path.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=path / "app.log",
        level=app_config.logging.file.level.upper(),
        format=log_format,
        rotation=app_config.logging.file.rotation,
        retention=app_config.logging.file.retention,
        encoding="utf-8",
    )
