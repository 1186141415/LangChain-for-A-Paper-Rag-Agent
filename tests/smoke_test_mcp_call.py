import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("ZHIPU_API_KEY not found in .env")

    client = MultiServerMCPClient(
        {
            "zhipu_search": {
                "transport": "http",
                "url": "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp",
                "headers": {
                    "Authorization": f"Bearer {api_key}"
                }
            }
        }
    )

    tools = await client.get_tools()

    print("Loaded MCP tools:")
    for tool in tools:
        print("-", tool.name)

    search_tool = tools[0]

    print("\n=== Tool schema ===")
    if hasattr(search_tool, "args_schema") and search_tool.args_schema:
        if isinstance(search_tool.args_schema, dict):
            print(search_tool.args_schema)
        elif hasattr(search_tool.args_schema, "model_json_schema"):
            print(search_tool.args_schema.model_json_schema())
        else:
            print(search_tool.args_schema)

    print("\n=== Invoke MCP tool ===")
    result = await search_tool.ainvoke(
        {"search_query": "latest AI agent engineering trends"}
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
