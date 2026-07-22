"""日志配置模型。

对应 config.yaml 中 logging: 下的配置项，用于配置 loguru 的输出。"""

from pydantic import BaseModel, Field


class ConsoleLoggingConfig(BaseModel):
    """控制台日志输出配置。"""

    enable: bool = Field(default=True, description="是否启用控制台日志输出")
    level: str = Field(default="INFO", description="控制台日志级别")


class FileLoggingConfig(BaseModel):
    """文件日志输出配置。"""

    enable: bool = Field(default=False, description="是否启用文件日志输出")
    level: str = Field(default="INFO", description="文件日志级别")
    path: str = Field(default="logs", description="日志文件目录")
    rotation: str = Field(default="500 MB", description="日志轮转大小")
    retention: str = Field(default="30 days", description="日志保留时间")


class LoggingConfig(BaseModel):
    """日志配置。"""

    console: ConsoleLoggingConfig = Field(default_factory=ConsoleLoggingConfig, description="控制台日志输出配置")
    file: FileLoggingConfig = Field(default_factory=FileLoggingConfig, description="文件日志输出配置")
