import requests

FASTAPI_BASE_URL = "http://127.0.0.1:8000"


def ask_ai(session_id: str, question: str) -> dict:
    resp = requests.post(
        f"{FASTAPI_BASE_URL}/ask",
        json={
            "session_id": session_id,
            "question": question,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()