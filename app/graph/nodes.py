import json
from typing import Any

from app.graph.state import AgentState
from app.llm_utils import client
from app.config import CHAT_MODEL
from app.logger_config import setup_logger

logger = setup_logger()


def build_choose_tool_node(tools: list[dict[str, Any]]):
    def choose_tool_node(state: AgentState) -> AgentState:
        query = state["query"]
        logger.info(f"[choose_tool_node] query: {query}")

        tool_desc = "\n".join([
            f"{t['name']}: {t['description']}" for t in tools
        ])

        prompt = f"""
                    You are an AI agent.
                    
                    Available tools:
                    {tool_desc}
                    
                    User question:
                    {query}
                    
                    Return JSON:
                    {{"tool": "...", "input": "..."}}
                  """

        content = ""
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            decision = json.loads(content)

            logger.info(f"[choose_tool_node] decision: {decision}")

            return {
                "decision": decision
            }

        except Exception as e:
            logger.exception("choose_tool_node failed")
            return {
                "decision": {"tool": "llm", "input": query},
                "error": f"choose_tool_node failed: {str(e)}"
            }

    return choose_tool_node


def build_execute_tool_node(tools: list[dict[str, Any]], rag=None):
    def execute_tool_node(state: AgentState) -> AgentState:
        try:
            decision = state["decision"]
            chat_history = state.get("chat_history", [])

            tool_name = decision["tool"]
            tool_input = decision["input"]

            logger.info(f"[execute_tool_node] tool_name: {tool_name}, tool_input: {tool_input}")

            for t in tools:
                if t["name"] == tool_name:
                    if tool_name == "rag":
                        result = t["func"](tool_input, rag, chat_history=chat_history)
                    elif tool_name == "llm":
                        result = t["func"](tool_input, chat_history=chat_history)
                    else:
                        result = t["func"](tool_input)

                    logger.info(f"[execute_tool_node] tool_output: {result}")

                    return {
                        "tool_result": {
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_output": result
                        }
                    }

            logger.warning(f"[execute_tool_node] tool not found: {tool_name}")

            return {
                "tool_result": {
                    "tool_name": "none",
                    "tool_input": tool_input,
                    "tool_output": "No valid tool found."
                },
                "error": f"Tool not found: {tool_name}"
            }

        except Exception as e:
            logger.exception("execute_tool_node failed")
            return {
                "tool_result": {
                    "tool_name": "error",
                    "tool_input": "",
                    "tool_output": "Tool execution failed."
                },
                "error": f"execute_tool_node failed: {str(e)}"
            }

    return execute_tool_node


def generate_answer_node(state: AgentState) -> AgentState:
    if state.get("error"):
        logger.warning(f"[generate_answer_node] error found in state: {state['error']}")
        return {
            "final_answer": f"系统执行过程中出现问题：{state['error']}"
        }

    tool_result = state["tool_result"]
    logger.info(f"[generate_answer_node] final_answer: {tool_result['tool_output']}")

    return {
        "final_answer": str(tool_result["tool_output"])
    }