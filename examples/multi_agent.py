"""
多 Agent 并行编排示例 — Supervisor + Send fan-out

用法：
    uv run examples/multi_agent.py
"""

import operator
from typing import Annotated, TypedDict

from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

from agent_lib.config import get_llm
from agent_lib.tools import search_web, get_weather


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    research_result: str
    weather_result: str
    final_report: str
    next_step: str


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

llm = get_llm()


def supervisor_node(state: AgentState) -> dict:
    """Supervisor：根据关键词路由"""
    messages = state.get("messages", [])
    user_msg = messages[-1].content if messages else ""

    if "天气" in user_msg and ("搜索" in user_msg or "查" in user_msg or "好玩" in user_msg):
        next_step = "parallel"
    elif "天气" in user_msg:
        next_step = "weather"
    elif "搜索" in user_msg or "查" in user_msg:
        next_step = "research"
    else:
        next_step = "parallel"

    return {"next_step": next_step}


def research_agent_node(state: AgentState) -> dict:
    """Research Agent：搜索信息"""
    llm_with_tools = llm.bind_tools([search_web])
    messages = state.get("messages", [])
    user_msg = messages[-1].content if messages else "最新科技新闻"

    response = llm_with_tools.invoke(
        [
            SystemMessage(content="你是一个研究助手，使用搜索工具查找信息并总结。"),
            HumanMessage(content=f"请搜索：{user_msg}"),
        ]
    )

    if response.tool_calls:
        tool_results = [search_web.invoke(tc["args"]) for tc in response.tool_calls]
        research_result = "\n".join(tool_results)
    else:
        research_result = response.content or "未找到相关信息。"

    return {
        "research_result": research_result,
        "messages": [AIMessage(content=f"[Research Agent] {research_result}")],
    }


def weather_agent_node(state: AgentState) -> dict:
    """Weather Agent：查询天气"""
    llm_with_tools = llm.bind_tools([get_weather])
    messages = state.get("messages", [])
    user_msg = messages[-1].content if messages else "北京"

    response = llm_with_tools.invoke(
        [
            SystemMessage(content="你是一个天气助手，使用天气工具查询天气信息。"),
            HumanMessage(content=f"请查询天气：{user_msg}"),
        ]
    )

    if response.tool_calls:
        tool_results = [get_weather.invoke(tc["args"]) for tc in response.tool_calls]
        weather_result = "\n".join(tool_results)
    else:
        weather_result = response.content or "无法获取天气信息。"

    return {
        "weather_result": weather_result,
        "messages": [AIMessage(content=f"[Weather Agent] {weather_result}")],
    }


def report_agent_node(state: AgentState) -> dict:
    """Report Agent：汇总生成报告"""
    context_parts = []
    if state.get("research_result"):
        context_parts.append(f"搜索结果：{state['research_result']}")
    if state.get("weather_result"):
        context_parts.append(f"天气信息：{state['weather_result']}")

    context = "\n".join(context_parts) or "暂无收集到的信息。"

    response = llm.invoke(
        [
            SystemMessage(content="你是一个报告撰写助手。根据提供的信息生成简洁的中文总结报告。"),
            HumanMessage(content=f"请根据以下信息生成报告：\n{context}"),
        ]
    )

    report = response.content or "报告生成失败。"
    return {
        "final_report": report,
        "messages": [AIMessage(content=f"[Report Agent]\n{report}")],
    }


# ──────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────


def fan_out_to_agents(state: AgentState) -> list[Send]:
    """条件边：返回 Send 列表实现并行 fan-out"""
    next_step = state.get("next_step", "parallel")
    if next_step == "parallel":
        return [Send("research", state), Send("weather", state)]
    elif next_step == "research":
        return [Send("research", state)]
    elif next_step == "weather":
        return [Send("weather", state)]
    else:
        return [Send("report", state)]


# ──────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("weather", weather_agent_node)
    graph.add_node("report", report_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", fan_out_to_agents)
    graph.add_edge("research", "report")
    graph.add_edge("weather", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=InMemorySaver())


# ──────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────


def main():
    agent = build_graph()

    print("=" * 60)
    print("并行执行 — 同时查天气和搜索信息")
    print("=" * 60)

    config = {"configurable": {"thread_id": str(uuid7())}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="帮我查一下北京的天气，同时搜索一下北京有什么好玩的地方")]},
        config=config,
    )

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
            print()


if __name__ == "__main__":
    main()
