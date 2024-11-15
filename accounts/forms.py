from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from .models import Profile
from django.contrib.auth.models import User

class LoginForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "password"]

class SignupForm(UserCreationForm):
    username = forms.CharField(label='User ID', widget=forms.TextInput(attrs={
        'pattern': '[a-zA-Z0-9]+',
        'title': '특수문자, 공백 입력불가',
    }))
    nickname = forms.CharField(label='Nickname')
    picture = forms.ImageField(label='Profile Picture - Image upload is optional.', required=False)

    class Meta(UserCreationForm.Meta):
        fields = UserCreationForm.Meta.fields + ('email',)
        
    def clean_nickname(self):
        nickname = self.cleaned_data.get('nickname')
        if Profile.objects.filter(nickname=nickname).exists():
            raise forms.ValidationError('이미 존재하는 닉네임 입니다.')
        return nickname
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        User = get_user_model()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('사용중인 이메일 입니다.')
        return email

    def save(self):
        user = super().save()
        # 기본 이미지 경로 설정
        picture = self.cleaned_data.get('picture') or 'default_images/default_profile.png'
        
        Profile.objects.create(
            user=user,
            nickname=self.cleaned_data['nickname'],
            picture=picture,
        )
        return user
    

class PasswordResetRequestForm(forms.Form):
    email = forms.EmailField()

class PasswordResetForm(forms.Form):
    code = forms.CharField(max_length=6)
    new_password = forms.CharField(widget=forms.PasswordInput, min_length=8)
    confirm_password = forms.CharField(widget=forms.PasswordInput, min_length=8)

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("new_password") != cleaned_data.get("confirm_password"):
            raise forms.ValidationError("Passwords do not match.")
        return cleaned_data
    
class FindUsernameForm(forms.Form):
    email = forms.EmailField()