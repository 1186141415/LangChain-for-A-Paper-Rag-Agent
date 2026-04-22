from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_home, name="chat_home"), # 首页聊天
    path("sessions/", views.session_list, name="session_list"), # 会话列表页
    path("sessions/<str:session_id>/", views.session_detail, name="session_detail"), # 会话详情页
]