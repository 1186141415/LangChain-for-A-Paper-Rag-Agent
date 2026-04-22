from django.shortcuts import render, get_object_or_404, redirect
from .services.ai_client import ask_ai
from .models import ChatSession, ChatMessage


def chat_home(request):
    result = None
    error = None

    if request.method == "POST":
        session_id = request.POST.get("session_id", "").strip()
        question = request.POST.get("question", "").strip()

        if session_id and question:
            try:
                result = ask_ai(session_id=session_id, question=question)

                session_obj, created = ChatSession.objects.get_or_create( # 先通过session_id查找会话，有会话返回会话对象，同时created字段设置为false，没有相关的会话字段，那么创建一个会话对象，并且通过default字段设置title
                    session_id=session_id,
                    defaults={"title": question[:60]}
                )

                if not session_obj.title:
                    session_obj.title = question[:60]
                    session_obj.save()

                ChatMessage.objects.create(
                    session=session_obj,
                    role="user",
                    content=question
                )

                ChatMessage.objects.create(
                    session=session_obj,
                    role="assistant",
                    content=result.get("answer", "")
                )

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

def session_list(request):
    sessions = ChatSession.objects.all().order_by("-updated_at")
    return render(
        request,
        "chat/session_list.html",
        {"sessions": sessions}
    )


def session_detail(request, session_id):
    session_obj = get_object_or_404(ChatSession, session_id=session_id)
    messages = session_obj.messages.all().order_by("created_at")

    return render(
        request,
        "chat/session_detail.html",
        {
            "session_obj": session_obj,
            "messages": messages,
        }
    )