from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import DATA_DIR
from app.data_loader import load_pdfs, process_documents
from app.rag_system import RAGSystem
from app.tools import TOOLS
from app.session_manager import SessionManager
from app.logger_config import setup_logger

from app.graph.workflow import AgentWorkflow

logger = setup_logger()

app = FastAPI()

session_manager = SessionManager(max_turns=3)

rag = None
workflow = None


class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.on_event("startup")
def startup_event():
    global rag, workflow

    logger.info("Loading RAG system...")

    docs = load_pdfs(DATA_DIR)
    logger.info(f"docs数量: {len(docs)}")

    chunks = process_documents(docs)
    logger.info(f"chunks数量: {len(chunks)}")

    rag = RAGSystem(chunks)
    rag.build_index()

    workflow = AgentWorkflow(TOOLS, rag=rag) # 初始化编排层

    logger.info("RAG + LangGraph ready!")


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        history = session_manager.get_history(req.session_id)

        answer = workflow.invoke(
            session_id=req.session_id,
            query=req.question,
            chat_history=history
        )

        session_manager.append_turn(
            req.session_id,
            req.question,
            answer
        )

        return {
            "session_id": req.session_id,
            "question": req.question,
            "answer": answer
        }

    except Exception as e:
        logger.exception("Error occurred in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear/{session_id}")
def clear_session(session_id: str):
    session_manager.clear_session(session_id)
    return {
        "session_id": session_id,
        "message": "session cleared"
    }
