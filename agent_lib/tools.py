"""可复用工具集"""

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[搜索结果] 关于 '{query}' 的最新信息：这是一个模拟的搜索结果。"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` by `b`.

    Args:
        a: First int
        b: Second int (must not be zero)
    """
    if b == 0:
        return float("inf")
    return a / b
