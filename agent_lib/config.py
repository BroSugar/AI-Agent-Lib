"""统一配置模块：模型初始化、代理绕过、环境变量加载"""

import os

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 代理绕过：防止本地 Ollama 请求被 HTTP 代理拦截
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# ──────────────────────────────────────────────
# 模型配置
# ──────────────────────────────────────────────

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b")
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))


def get_llm(
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
):
    """
    创建 LLM 实例（延迟导入，避免循环依赖）。

    所有参数优先使用传入值，否则读取环境变量，最后使用默认值。
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=model or DEFAULT_MODEL,
        model_provider=provider or DEFAULT_PROVIDER,
        base_url=base_url or DEFAULT_BASE_URL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
    )
