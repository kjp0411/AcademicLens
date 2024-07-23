from django import forms
from .models import Question, Comment  # Comment 모델도 임포트합니다.

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['title', 'content']

class CommentForm(forms.ModelForm):  # CommentForm 클래스를 정의합니다.
    class Meta:
        model = Comment
        fields = ['content']  # 댓글 내용만 입력받는 폼입니다.
