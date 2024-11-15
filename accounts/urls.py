from django.urls import path
from .views import *
from . import views
app_name = 'accounts'

urlpatterns = [
    path('signup/', signup, name='signup'),
    path('login/', login_check, name='login'),
    path('logout/', logout, name='logout'),
    path('follow/', follow, name='follow'),
    path('password-reset/', views.password_reset_request, name='password_reset_request'),
    path('password-reset-verify/', views.password_reset_verify, name='password_reset_verify'),
    path('find-username/', views.find_username, name='find_username'),
]