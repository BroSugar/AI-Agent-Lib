"""
单 Agent 示例 — create_agent 自动 ReAct 循环

用法：
    uv run examples/react_agent.py
"""

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

from agent_lib.config import get_llm
from agent_lib.tools import get_weather, add, multiply, divide

SYSTEM_PROMPT = "You are a helpful assistant that can check weather and perform arithmetic."


def main():
    llm = get_llm()
    tools = [get_weather, multiply, add, divide]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": str(uuid7())}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="Add 3 and 4.")]},
        config=config,
    )

    for msg in result["messages"]:
        role = msg.type
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"[{role}] tool_calls: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}")
        elif hasattr(msg, "name") and msg.name:
            print(f"[{role}] {msg.name} → {msg.content}")
        else:
            print(f"[{role}] {msg.content}")


if __name__ == "__main__":
    main()
