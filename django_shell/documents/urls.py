from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_page, name="upload_page"),
]