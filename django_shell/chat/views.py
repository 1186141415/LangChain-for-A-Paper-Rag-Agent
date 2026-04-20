from django.shortcuts import render
from .services.ai_client import ask_ai


def chat_home(request):
    result = None
    error = None

    if request.method == "POST":
        session_id = request.POST.get("session_id", "").strip()
        question = request.POST.get("question", "").strip()

        if session_id and question:
            try:
                result = ask_ai(session_id=session_id, question=question)
            except Exception as e:
                error = str(e)
        else:
            error = "session_id 和 question 不能为空"

    return render(
        request,
        "chat/chat_home.html",
        {
            "result": result,
            "error": error,
        }
    )