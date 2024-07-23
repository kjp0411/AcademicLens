from django.urls import path
from . import views

app_name = 'board'

urlpatterns = [
    path('', views.index, name='index'),
    path('questions/<int:pk>/', views.question_detail, name='question_detail'),
    path('questions/<int:pk>/like/', views.like_question, name='like_question'),
    path('questions/<int:pk>/comment/', views.add_comment_to_question, name='add_comment_to_question'),
    path('questions/<int:pk>/edit/', views.edit_question, name='edit_question'),
    path('questions/<int:pk>/delete/', views.delete_question, name='delete_question'),
    path('create_question/', views.create_question, name='create_question'),
    path('comment/<int:pk>/delete/', views.delete_comment, name='delete_comment'),
    path('user_questions/', views.user_questions, name='user_questions'),
]