from django.db import models
from django.contrib.auth.models import User

class Question(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    pub_date = models.DateTimeField('date published', auto_now_add=True)
    likes = models.IntegerField(default=0)
    user = models.ForeignKey(User, related_name='questions', on_delete=models.CASCADE)  # non-nullable로 변경
    image = models.ImageField(upload_to='question_images/', blank=True, null=True)

class Answer(models.Model):
    question = models.ForeignKey(Question, related_name='answers', on_delete=models.CASCADE)
    content = models.TextField()
    pub_date = models.DateTimeField('date published')

class Comment(models.Model):
    question = models.ForeignKey(Question, related_name='comments', on_delete=models.CASCADE)
    content = models.TextField()
    pub_date = models.DateTimeField('date published')
    user = models.ForeignKey(User, related_name='comments', on_delete=models.CASCADE)

class Like(models.Model):
    user = models.ForeignKey(User, related_name='board_likes', on_delete=models.CASCADE)
    question = models.ForeignKey(Question, related_name='question_likes', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
