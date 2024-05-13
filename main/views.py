from django.shortcuts import render
from .models import Paper, Affiliation, Author

def show_paper_info(request):
    papers = Paper.objects.all()
    paper_count = papers.count()
    affiliation_count = Affiliation.objects.values('name').distinct().count()
    author_count = Author.objects.values('name').distinct().count()
    return render(request, 'index.html', {
        'papers': papers,
        'paper_count': paper_count,
        'affiliation_count': affiliation_count,
        'author_count': author_count
    })
