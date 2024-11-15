from django.contrib.auth import authenticate, login
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth import logout as django_logout
from .forms import SignupForm, LoginForm
from .models import Profile, Follow
from django.contrib import messages
import json
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

import random
import string
from django.conf import settings
from django.core.mail import send_mail
from django.utils import timezone
from django.contrib.auth.models import User
from .models import PasswordResetCode
from .forms import PasswordResetRequestForm, PasswordResetForm

from .forms import FindUsernameForm

def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            return redirect('accounts:login')
    else:
        form = SignupForm()
        
        
    if request.is_ajax():
        profile, created_at = Profile.objects.get_or_create(user=user_id)
        if created_at:
            message = '이미 가입된 사용자입니다!'
            status = 1
        else:
            profile.delete()
            message = '회원가입 가입가능'
            status = 0
        context = {
            'message': message,
            'status': status,
        }
        return HttpResponse(json.dumps(context), content_type="application/json")
        
        return render(request, 'post/post_list_ajax.html', {
            'posts': posts,
            'comment_form': comment_form,
        })  
        
    return render(request, 'accounts/signup.html', {
        'form': form,
    })

def login_check(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        name = request.POST.get('username')
        pwd = request.POST.get('password')
        user = authenticate(username=name, password=pwd)
        
        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            messages.error(request, "로그인에 실패하셨습니다.")
            return redirect('accounts:login')
    else:
        form = LoginForm()
        return render(request, 'accounts/login.html', {"form": form})
    
def logout(request):
    django_logout(request)
    return redirect("/")

@login_required
@require_POST
def follow(request):
    from_user = request.user.profile
    pk = request.POST.get('pk')
    to_user = get_object_or_404(Profile, pk=pk)
    follow, created = Follow.objects.get_or_create(from_user=from_user, to_user=to_user)
    if created:
        message = '팔로우 시작!'
        status = 1
    else:
        follow.delete()
        message = '팔로우 취소'
        status = 0
    context = {
        'message': message,
        'status': status,
    }
    return HttpResponse(json.dumps(context), content_type="application/json")



def generate_code():
    return ''.join(random.choices(string.digits, k=6))

def password_reset_request(request):
    if request.method == 'POST':
        form = PasswordResetRequestForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = User.objects.get(email=email)
                code = generate_code()
                PasswordResetCode.objects.create(user=user, code=code)
                send_mail(
                    'Password Reset Verification',
                    f'Your verification code is: {code}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
                messages.success(request, "Verification code sent to your email.")
                return redirect('password_reset_verify')
            except User.DoesNotExist:
                form.add_error('email', 'Email not registered.')
    else:
        form = PasswordResetRequestForm()
    return render(request, 'accounts/password_reset_request.html', {'form': form})



def password_reset_verify(request):
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            code = form.cleaned_data['code']
            try:
                reset_code = PasswordResetCode.objects.get(code=code)
                if reset_code.is_valid():
                    user = reset_code.user
                    user.set_password(form.cleaned_data['new_password'])
                    user.save()
                    reset_code.delete()  # 사용된 코드 삭제
                    messages.success(request, "Password reset successfully.")
                    return redirect('accounts:login')
                else:
                    messages.error(request, "Code expired.")
            except PasswordResetCode.DoesNotExist:
                messages.error(request, "Invalid code.")
    else:
        form = PasswordResetForm()
    return render(request, 'accounts/password_reset_verify.html', {'form': form})



def find_username(request):
    if request.method == 'POST':
        form = FindUsernameForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = User.objects.get(email=email)
                send_mail(
                    'Your Username',
                    f'Your username is: {user.username}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
                messages.success(request, "Your username has been sent to your email.")
                return redirect('accounts:login')
            except User.DoesNotExist:
                form.add_error('email', 'No user with this email exists.')
    else:
        form = FindUsernameForm()
    return render(request, 'accounts/find_username.html', {'form': form})