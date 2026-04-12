from app.graph.builder import build_agent_graph


class AgentWorkflow:
    def __init__(self, tools, rag=None):
        self.tools = tools
        self.rag = rag
        self.graph = build_agent_graph(tools, rag=rag)

    def invoke(self, session_id: str, query: str, chat_history=None):
        if chat_history is None:
            chat_history = []

        state = {
            "session_id": session_id,
            "query": query,
            "chat_history": chat_history,
        }

        result = self.graph.invoke(state)  #返回结果仍是状态图
        return result["final_answer"]