from django.shortcuts import render
from .models import Paper, Affiliation
from django.db import connection

def show_paper_info(request):
    # SQL 쿼리 실행
    with connection.cursor() as cursor:
        cursor.execute('''
            SELECT 
                paper.id, paper.search, paper.title, paper.url, paper.date, paper.citations,
                paper.publisher, paper.abstract, paper.keywords, GROUP_CONCAT(author.name) AS author_names
            FROM 
                paper
            INNER JOIN 
                paper_author ON paper.id = paper_author.paper_id
            INNER JOIN 
                author ON paper_author.author_id = author.id
            GROUP BY
                paper.id;
        ''')
        # 결과 가져오기
        rows = cursor.fetchall()

    # 결과를 딕셔너리 형태로 변환
    papers = []
    for row in rows:
        paper = {
            'id': row[0],
            'search': row[1],
            'title': row[2],
            'url': row[3],
            'date': row[4],
            'citations': row[5],
            'publisher': row[6],
            'abstract': row[7],
            'keywords': row[8],
            'author_names': row[9].split(',') if row[9] else []  # 쉼표로 구분된 문자열을 리스트로 변환
        }
        papers.append(paper)

    paper_count = len(papers)
    affiliation_count = Affiliation.objects.values('name').distinct().count()

    return render(request, 'index.html', {
        'papers': papers,
        'paper_count': paper_count,
        'affiliation_count': affiliation_count,
    })
