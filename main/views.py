from django.shortcuts import render
from .models import Paper, Affiliation
from django.db import connection

def show_paper_info(request):
    papers = Paper.objects.all()
    paper_count = len(papers)
    affiliation_count = Affiliation.objects.values('name').distinct().count()
    return render(request, 'index.html', {
        'paper_count': paper_count,
        'affiliation_count': affiliation_count
    })

