"""
MCP (Model Context Protocol) 远程工具调用示例
HTTP 请求头携带 Bearer Token 认证。

用法：
    $env:MCP_SERVER_URL="http://your-server/mcp/sse"
    $env:MCP_TOKEN="your-auth-token"
    uv run examples/mcp_agent.py
"""

import asyncio
import os

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

from agent_lib.config import get_llm

# ──────────────────────────────────────────────
# 配置（从环境变量读取，不硬编码）
# ──────────────────────────────────────────────

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080/mcp/sse")
MCP_TOKEN = os.environ.get("MCP_TOKEN", "")


# ──────────────────────────────────────────────
# SSE 方式 + Bearer Token
# ──────────────────────────────────────────────


async def run_mcp_sse():
    """连接 MCP Server（SSE 传输），HTTP 头携带 Bearer Token"""
    llm = get_llm()

    async with MultiServerMCPClient(
        {
            "remote_server": {
                "transport": "sse",
                "url": MCP_SERVER_URL,
                "headers": {
                    "Authorization": f"Bearer {MCP_TOKEN}",
                },
                "timeout": 30,
            },
        }
    ) as client:
        tools = client.get_tools()
        print(f"已连接 MCP Server，发现 {len(tools)} 个工具：")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
        print()

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="你是一个有用的助手，可以调用远程 MCP 工具完成任务。",
            checkpointer=InMemorySaver(),
        )

        config = {"configurable": {"thread_id": str(uuid7())}}
        result = agent.invoke(
            {"messages": [HumanMessage(content="帮我查询一下今天的任务列表")]},
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


# ──────────────────────────────────────────────
# Streamable HTTP 方式
# ──────────────────────────────────────────────


async def run_mcp_streamable_http():
    """连接 MCP Server（Streamable HTTP 传输），同样支持 Token"""
    llm = get_llm()

    async with MultiServerMCPClient(
        {
            "api_server": {
                "transport": "streamable_http",
                "url": os.environ.get("MCP_STREAMABLE_URL", "http://localhost:8080/mcp"),
                "headers": {
                    "Authorization": f"Bearer {MCP_TOKEN}",
                },
                "timeout": 60,
            }
        }
    ) as client:
        tools = client.get_tools()
        print(f"[Streamable HTTP] 发现 {len(tools)} 个工具")

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="你是一个有用的助手。",
            checkpointer=InMemorySaver(),
        )

        config = {"configurable": {"thread_id": str(uuid7())}}
        result = agent.invoke(
            {"messages": [HumanMessage(content="你好，请介绍一下你能做什么")]},
            config=config,
        )
        print(result["messages"][-1].content)


# ──────────────────────────────────────────────
# 本地 stdio（无需 Token）
# ──────────────────────────────────────────────


async def run_mcp_stdio():
    """连接本地 stdio MCP Server"""
    llm = get_llm()

    async with MultiServerMCPClient(
        {
            "local_tools": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"],
            }
        }
    ) as client:
        tools = client.get_tools()
        print(f"[Local stdio] 发现 {len(tools)} 个工具")

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="你是一个文件系统助手。",
            checkpointer=InMemorySaver(),
        )

        config = {"configurable": {"thread_id": str(uuid7())}}
        result = agent.invoke(
            {"messages": [HumanMessage(content="列出当前目录下的文件")]},
            config=config,
        )
        print(result["messages"][-1].content)


# ──────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────


def main():
    if not MCP_TOKEN:
        print("=" * 60)
        print("MCP Agent — HTTP + Bearer Token 认证")
        print("=" * 60)
        print()
        print("请设置环境变量后运行：")
        print('  $env:MCP_TOKEN="your-token"')
        print(f'  $env:MCP_SERVER_URL="{MCP_SERVER_URL}"')
        print("  uv run examples/mcp_agent.py")
        return

    asyncio.run(run_mcp_sse())


if __name__ == "__main__":
    main()
