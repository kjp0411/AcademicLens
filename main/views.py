import mariadb
from django.shortcuts import render, get_object_or_404
from .models import Paper, Author, Keyword, Affiliation, Country, PaperAuthor, PaperAffiliation, PaperKeyword, PaperCountry, SavedPaper, RecentPaper, SearchKeyword, AuthUser
from django.db import connection
from collections import Counter
from datetime import datetime
from django.core.paginator import Paginator
from django.http import JsonResponse
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from django.db.models import Count, Q
import json

from django.shortcuts import redirect
from django.contrib import messages
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.template.loader import render_to_string
from django.http import HttpResponse
from django.db.models import Case, When, IntegerField

# gpt api 라이브러리
import openai
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from braces.views import CsrfExemptMixin

import logging # 문제 발생 시 로그 띄우기
import os
from dotenv import load_dotenv

# 이미지 저장 라이브러리
import base64
from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from PIL import Image
from io import BytesIO

# 폴더 생성/수정/삭제 관리 라이브러리
from django.core.files.base import ContentFile
import shutil

# 리포팅
from django.views.decorators.csrf import csrf_exempt
# .env 파일 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 불러오기
openai.api_key = os.getenv('OPENAI_API_KEY')

db_config = {
    'host': '127.0.0.1',
    'user': 'goorm',
    'password': '123456',
    'database': 'capstone',
    'port':3307
}

# 메인 화면 논문 수, 소속 수, 인기 논문 5개 가져오기
def home(request):
    # 전체 논문 수 및 학회 수 계산
    papers = Paper.objects.all()
    paper_count = len(papers)
    affiliation_count = Affiliation.objects.values('name').distinct().count()

    # 인기 논문 5개 가져오기 (saved_count 기준)
    popular_papers = Paper.objects.order_by('-saved_count')[:5]

    # 실시간 인기 키워드 상위 10개 가져오기
    popular_keywords_1_5 = SearchKeyword.objects.all().order_by('-count')[:5]
    popular_keywords_6_10 = SearchKeyword.objects.all().order_by('-count')[5:10]

    # 유저 수 가져오기
    user_count = AuthUser.objects.count()
    
    return render(request, 'home.html', {
        'paper_count': paper_count,
        'affiliation_count': affiliation_count,
        'popular_papers': popular_papers,  # 인기 논문 데이터
        'popular_keywords_1_5': popular_keywords_1_5,  # 상위 1-5위 인기 키워드
        'popular_keywords_6_10': popular_keywords_6_10,  # 상위 6-10위 인기 키워드
        'user_count': user_count,
    })

# 소개 화면
def introduce(request):
    return render(request, 'introduce.html')

# 검색 시 논문 출력 및 필터링
def search(request):
    try:
        query = request.GET.get('query', '')
        years = request.GET.getlist('year')
        publishers = request.GET.getlist('publisher')
        author = request.GET.get('author-search')
        affiliation = request.GET.get('affiliation-search')
        countries = request.GET.getlist('country')
        search_query = query
        news_type = request.GET.get('news_type', 'international')  # 기본값을 'international'로 설정
        order = request.GET.get('order', 'desc')
        sort_by = request.GET.get('sort_by', 'title')
        items_per_page = int(request.GET.get('items_per_page', 10))  # 기본값 10
        filter_type = request.GET.get('filter', 'paper')

        # 로그인된 사용자의 저장된 논문 ID 확인
        saved_paper_ids = []
        if request.user.is_authenticated:
            saved_paper_ids = SavedPaper.objects.filter(user=request.user).values_list('paper_id', flat=True)

        # 필터(검색창-콤보박스)에 따라 논문 검색
        if filter_type == 'paper':
            paper_ids = get_paper_ids(query)
        elif filter_type == 'author':
            paper_ids = get_author_paper_ids(query)
        elif filter_type == 'country':
            paper_ids = get_country_paper_ids(query)
        elif filter_type == 'affiliation':
            paper_ids = get_affiliation_paper_ids(query)

        related_terms = []
        if query:
            related_terms, most_related_word = top_5_related_words(query, paper_ids)

        # 기본 쿼리셋 생성
        papers = Paper.objects.filter(id__in=paper_ids)

        # 정렬 처리
        if sort_by == 'title':
            # 정확하게 단어 경계로 일치하는 경우 우선순위 높이기
            papers = papers.annotate(
                exact_word_match=Case(
                    When(title__iregex=r'\b' + query + r'\b', then=1),  # 단어 경계를 포함하여 정규식 검색
                    default=0,
                    output_field=IntegerField()
                ),
                search_rank=Count('title', filter=Q(title__icontains=query))
            ).order_by('-exact_word_match', '-search_rank' if order == 'desc' else 'search_rank')
        elif sort_by == 'latest':
            papers = papers.order_by('date' if order == 'asc' else '-date')

        # 연도 필터링
        if years:
            papers = papers.filter(date__year__in=years)

        # 발행처 필터링
        if publishers:
            papers = papers.filter(publisher__in=publishers)

        # 저자 필터링
        if author:
            paper_ids_by_authors = Author.objects.filter(name__icontains=author).values_list('paperauthor__paper_id', flat=True)
            papers = papers.filter(id__in=paper_ids_by_authors)

        # 소속 필터링
        if affiliation:
            paper_ids_by_affiliations = Affiliation.objects.filter(name__icontains=affiliation).values_list('paperaffiliation__paper_id', flat=True)
            papers = papers.filter(id__in=paper_ids_by_affiliations)
            
        # 국가 필터링
        if countries:
            paper_ids_by_countries = PaperCountry.objects.filter(country__name__in=countries).values_list('paper_id', flat=True)
            papers = papers.filter(id__in=paper_ids_by_countries)

        selected_countries = countries
        
        # 필터링된 논문 개수
        total_results = papers.count()

        if total_results == 0:
            # 검색 결과가 없을 경우 error.html로 이동
            return render(request, 'error.html', {'error_message': '검색 결과가 없습니다.'})

        # 페이징 처리
        paginator = Paginator(papers, items_per_page)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        # 현재 페이지 그룹 계산 (10개 단위)
        current_page = page_obj.number
        current_group_start = (current_page - 1) // 10 * 10 + 1
        current_group_end = min(current_group_start + 9, paginator.num_pages)

        # 페이지 그룹 리스트 생성
        current_group_pages = list(range(current_group_start, current_group_end + 1))

        # 저자 및 키워드 추가
        papers_with_authors_and_keywords = []
        for paper in page_obj:
            authors = Author.objects.filter(paperauthor__paper_id=paper.id)
            keywords = Keyword.objects.filter(paperkeyword__paper_id=paper.id)
            affiliations = Affiliation.objects.filter(paperaffiliation__paper_id=paper.id)

            # 논문 내 국가 정보 수집 및 중복 제거
            countries = PaperCountry.objects.filter(paper_id=paper.id).select_related('country')
            unique_countries = list(set([country.country.name for country in countries]))

            # 논문이 저장된 상태인지 확인 (로그인한 경우에만 확인)
            is_saved = paper.id in saved_paper_ids if request.user.is_authenticated else False

            save_count = paper.saved_count
            
            papers_with_authors_and_keywords.append({
                'paper': paper,
                'save-count': save_count,
                'authors': authors,
                'keywords': keywords,
                'affiliations': affiliations,
                'countries': unique_countries,  # 나라 중복 제거
                'is_saved': is_saved,  # 저장 여부 추가
            })

        # 연도 및 발행처별 논문 수 계산
        year = range(2019, datetime.now().year + 1)
        paper_counts_by_year = {str(y): Paper.objects.filter(id__in=paper_ids, date__year=y).count() for y in year}

        publisher = ['ACM', 'IEEE', 'Springer']
        paper_counts_by_publisher = {p: Paper.objects.filter(id__in=paper_ids, publisher=p).count() for p in publisher}

        paper_countries = PaperCountry.objects.filter(paper_id__in=paper_ids).values_list('country__name', 'paper_id')
        country_paper_map = {}
        for country, paper in paper_countries:
            if country not in country_paper_map:
                country_paper_map[country] = set()
            country_paper_map[country].add(paper)

        paper_counts_by_country = {country: len(papers) for country, papers in country_paper_map.items()}
        paper_counts_by_country = dict(sorted(paper_counts_by_country.items(), key=lambda item: item[1], reverse=True))     # 국가별 논문 수 내림차순

        # 뉴스 검색 부분
        api_key = '2f963493ee124210ac91a3b54ebb3c5c'
        articles = []

        if filter_type == 'author':
            if search_query:
                if news_type == 'domestic':
                    url = f'https://newsapi.org/v2/everything?q={most_related_word}&language=ko&apiKey={api_key}'
                else:
                    url = f'https://newsapi.org/v2/everything?q={most_related_word}&language=en&apiKey={api_key}'

                response = requests.get(url)
                news_data = response.json()
                articles = news_data.get('articles', [])
                articles = [article for article in articles if article.get('title') != "[Removed]"]
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'articles': articles})
        else:
            if search_query:
                if news_type == 'domestic':
                    url = f'https://newsapi.org/v2/everything?q={search_query}&language=ko&apiKey={api_key}'
                else:
                    url = f'https://newsapi.org/v2/everything?q={search_query}&language=en&apiKey={api_key}'

                response = requests.get(url)
                news_data = response.json()
                articles = news_data.get('articles', [])
                articles = [article for article in articles if article.get('title') != "[Removed]"]

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'articles': articles})

        context = {
            'query': query,
            'papers_with_authors_and_keywords': papers_with_authors_and_keywords,
            'paper_counts_by_year': paper_counts_by_year,
            'paper_counts_by_publisher': paper_counts_by_publisher,
            'paper_counts_by_country': paper_counts_by_country,
            'related_terms': related_terms,
            'articles': articles,
            'news_type': news_type,
            'search_query': search_query,
            'page_obj': page_obj,
            'selected_years': years,
            'selected_publishers': publishers,
            'selected_countries': selected_countries,
            'total_results': total_results,
            'author': author,
            'affiliation': affiliation,
            'order': order,
            'sort_by': sort_by,
            'items_per_page': items_per_page,
            'current_group_pages': current_group_pages,
            'filter': filter_type,
        }

        # 검색 결과가 있는 경우에만 렌더 후에 키워드 저장
        response = render(request, 'search.html', context)

        # 검색 결과가 성공적으로 렌더링된 후에 키워드 카운트 업데이트
        if total_results > 0 and query and filter_type == 'paper':
            keyword, created = SearchKeyword.objects.get_or_create(keyword=query)
            if not created:
                keyword.count += 1
            keyword.save()

        return response
    except Exception as e:
        # 오류 발생 시 error.html로 리디렉션
        return render(request, 'error.html', {'error_message': str(e)})


# 저자 이름으로 검색하는 엔진
def get_author_paper_ids(user_keyword, start_year=2019, end_year=2024):
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    # 공백을 제거하고 비교하는 SQL 쿼리
    query_conditions = []
    query_params = []

    # 저자 이름 검색 조건 추가
    query_conditions.append("""
        REPLACE(a.name, ' ', '') LIKE REPLACE(%s, ' ', '')
    """)
    query_params.append(f'%{user_keyword}%')

    # 연도 필터링 조건 추가 (기본값: 2019 ~ 2024)
    query_conditions.append("YEAR(p.date) BETWEEN %s AND %s")
    query_params.extend([start_year, end_year])

    # 쿼리 실행
    query = f"""
        SELECT p.id
        FROM paper p
        JOIN paper_author pa ON p.id = pa.paper_id
        JOIN author a ON pa.author_id = a.id
        WHERE {' AND '.join(query_conditions)};
    """
    cursor.execute(query, query_params)

    # 쿼리 결과 가져오기
    paper_ids = [row['id'] for row in cursor.fetchall()]

    cursor.close()
    db.close()
    
    return paper_ids

def get_affiliation_paper_ids(user_keyword, start_year=2019, end_year=2024):
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    # 공백을 제거하고 비교하는 SQL 쿼리
    query_conditions = []
    query_params = []

    # 소속 이름 검색 조건 추가
    query_conditions.append("""
        REPLACE(a.name, ' ', '') LIKE REPLACE(%s, ' ', '')
    """)
    query_params.append(f'%{user_keyword}%')

    # 연도 필터링 조건 추가 (기본값: 2019 ~ 2024)
    query_conditions.append("YEAR(p.date) BETWEEN %s AND %s")
    query_params.extend([start_year, end_year])

    # 쿼리 실행
    query = f"""
        SELECT p.id
        FROM paper p
        JOIN paper_affiliation pa ON p.id = pa.paper_id
        JOIN affiliation a ON pa.affiliation_id = a.id
        WHERE {' AND '.join(query_conditions)};
    """
    cursor.execute(query, query_params)

    # 쿼리 결과 가져오기
    paper_ids = [row['id'] for row in cursor.fetchall()]

    cursor.close()
    db.close()
    
    return paper_ids


def get_country_paper_ids(user_keyword, start_year=2019, end_year=2024):
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    # 조건문을 통해 입력된 키워드 길이에 따라 검색 필드를 결정
    query_conditions = []
    query_params = []

    if len(user_keyword) == 2:
        # alpha_2로 검색
        query_conditions.append("c.alpha_2 = %s")
        query_params.append(user_keyword)
    elif len(user_keyword) == 3:
        # alpha_3로 검색
        query_conditions.append("c.alpha_3 = %s")
        query_params.append(user_keyword)
    else:
        # name으로 검색 (공백을 무시하고 검색)
        query_conditions.append("REPLACE(c.name, ' ', '') LIKE REPLACE(%s, ' ', '')")
        query_params.append(f'%{user_keyword}%')

    # 연도 필터링 조건 추가 (기본값: 2019 ~ 2024)
    query_conditions.append("YEAR(p.date) BETWEEN %s AND %s")
    query_params.extend([start_year, end_year])

    # 쿼리 실행
    query = f"""
        SELECT p.id
        FROM paper p
        JOIN paper_country pc ON p.id = pc.paper_id
        JOIN country c ON pc.country_id = c.id
        WHERE {' AND '.join(query_conditions)};
    """
    cursor.execute(query, query_params)

    # 결과 가져오기
    paper_ids = [row['id'] for row in cursor.fetchall()]

    cursor.close()
    db.close()
    
    return paper_ids

    
# 논문 제목으로 검색하는 엔진
def get_paper_ids(user_keyword, start_year=2019, end_year=2024):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 기본 쿼리
    query_conditions = []
    query_params = []

    # 검색 키워드 조건 추가
    if len(user_keyword) <= 2:  # 검색 키워드 길이가 2 이하일 때
        query_conditions.append("""
            (search LIKE %s OR title LIKE %s OR abstract LIKE %s
             OR EXISTS (SELECT 1 FROM paper_keyword pk
                        JOIN keyword k ON pk.keyword_id = k.id
                        WHERE pk.paper_id = p.id AND k.keyword_name LIKE %s))
        """)
        query_params.extend([f'%{user_keyword}%', f'%{user_keyword}%', f'%{user_keyword}%', f'%{user_keyword}%'])
    else:  # 검색 키워드 길이가 3 이상일 때
        # 띄어쓰기가 포함된 경우 boolean 모드와 큰따옴표 사용
        if " " in user_keyword:
            query_conditions.append("""
                (MATCH(search, title, abstract) AGAINST(%s IN BOOLEAN MODE)
                 OR EXISTS (SELECT 1 FROM paper_keyword pk
                            JOIN keyword k ON pk.keyword_id = k.id
                            WHERE pk.paper_id = p.id AND MATCH(k.keyword_name) AGAINST(%s IN BOOLEAN MODE)))
            """)
            query_params.extend([f'"{user_keyword}"', f'"{user_keyword}"'])
        else:  # 띄어쓰기가 없는 경우 일반적인 MATCH AGAINST 사용
            query_conditions.append("""
                (MATCH(search, title, abstract) AGAINST(%s)
                 OR EXISTS (SELECT 1 FROM paper_keyword pk
                            JOIN keyword k ON pk.keyword_id = k.id
                            WHERE pk.paper_id = p.id AND MATCH(k.keyword_name) AGAINST(%s)))
            """)
            query_params.extend([user_keyword, user_keyword])

    # 연도 필터링 조건 추가 (기본값: 2019 ~ 2024)
    query_conditions.append("YEAR(p.date) BETWEEN %s AND %s")
    query_params.extend([start_year, end_year])

    # 쿼리 실행
    query = f"""
        SELECT p.id
        FROM paper p
        WHERE {' AND '.join(query_conditions)};
    """
    cursor.execute(query, query_params)

    # 쿼리 결과 가져오기
    paper_ids = [row['id'] for row in cursor.fetchall()]

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return paper_ids


# 주어진 논문 ID 목록에 대한 abstract를 가져오는 함수
def get_abstracts(paper_ids):

    if not paper_ids:
        return pd.DataFrame(columns=['abstract'])
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 논문 ID 목록에 해당하는 abstract 가져오기
    format_strings = ','.join(['%s'] * len(paper_ids))
    cursor.execute(f"SELECT abstract FROM paper WHERE id IN ({format_strings})", tuple(paper_ids))

    # 결과를 데이터프레임으로 변환
    data = pd.DataFrame(cursor.fetchall(), columns=['abstract'])

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return data

def top_5_related_words(query, paper_ids):
    # 검색 엔진을 통해 논문 ID 가져오기
    # paper_ids = get_paper_ids(query)

    # 논문 ID를 사용하여 abstract 가져오기
    abstract_data = get_abstracts(paper_ids)

    # abstract 컬럼 내용 추출
    abstracts = abstract_data["abstract"]

    # TF-IDF 벡터화 (불용어 처리)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

    # 각 단어의 인덱스 확인
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 데이터프레임으로 변환
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # 각 단어의 총 TF-IDF 값을 계산
    tfidf_sums = tfidf_df.sum(axis=0)

    # TF-IDF 값으로 정렬하여 상위 20개 단어 선택
    top_20_words = tfidf_sums.sort_values(ascending=False).head(20).index.tolist()

    # BERT 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 단어 임베딩 계산 함수
    def get_word_embedding(word, tokenizer, model):
        inputs = tokenizer(word, return_tensors='pt')
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # 사용자 입력 키워드의 임베딩 계산
    user_keyword_embedding = get_word_embedding(query, tokenizer, model)

    # 상위 단어들의 임베딩 계산 및 유사도 측정
    word_embeddings = {word: get_word_embedding(word, tokenizer, model) for word in top_20_words}
    similarities = {word: cosine_similarity(user_keyword_embedding, word_embeddings[word])[0][0] for word in top_20_words if word != query}

    # 유사도가 높은 상위 5개 단어 선택
    top_5_related_words = sorted(similarities, key=similarities.get, reverse=True)[:5]

    # 유사도가 가장 높은 단어 선택
    most_related_word = max(similarities, key=similarities.get)

    return top_5_related_words, most_related_word


# 분석 페이지 그래프
def analyze(request):
    start_year = int(request.GET.get('start_year', 2019))  # 기본값은 2019
    end_year = int(request.GET.get('end_year', 2024))  # 기본값은 2024
    if 'query' in request.GET:
        user_keyword = request.GET['query']
        filter_type = request.GET.get('filter', 'paper')

        # MariaDB 데이터베이스 연결
        db = mariadb.connect(**db_config)
        # 커서 생성
        cursor = db.cursor(dictionary=True)

        if filter_type == 'paper':
            paper_ids = get_paper_ids(user_keyword, start_year=start_year, end_year=end_year)
        elif filter_type == 'author':
            paper_ids = get_author_paper_ids(user_keyword, start_year=start_year, end_year=end_year)
        elif filter_type == 'country':
            paper_ids = get_country_paper_ids(user_keyword, start_year=start_year, end_year=end_year)
        elif filter_type == 'affiliation':
            paper_ids = get_affiliation_paper_ids(user_keyword, start_year=start_year, end_year=end_year)

        papers = Paper.objects.filter(id__in=paper_ids)
        total_results = papers.count()

        # 연도별로 그룹화하고 각 연도의 논문 수 구하기
        if paper_ids:
            placeholders = ', '.join(['%s'] * len(paper_ids))
            cursor.execute(f"""
            SELECT YEAR(p.date) AS year, COUNT(*) AS count
            FROM paper p
            WHERE p.id IN ({placeholders})
            GROUP BY year
            ORDER BY year;
            """, paper_ids)
            
            papers_count = cursor.fetchall()
        else:
            papers_count = []

        # 데이터베이스 연결 종료
        cursor.close()
        db.close()

        # 저자별 논문 수를 계산
        author_counts = []
        if paper_ids:
            placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
            with connection.cursor() as cursor:
                cursor.execute(f"""
                SELECT a.name AS author_name, COUNT(distinct(pa.paper_id)) AS publications_count
                FROM author a
                JOIN paper_author pa ON a.id = pa.author_id
                JOIN paper p ON pa.paper_id = p.id
                WHERE p.id IN ({placeholders})
                GROUP BY a.name
                ORDER BY publications_count DESC
                LIMIT 10;
                """, paper_ids)

                rows = cursor.fetchall()
                author_counts = [(row[0].encode('latin1').decode('utf8'), row[1]) for row in rows]

        # 저자 이름과 논문 수를 결합한 리스트 생성
        author_data = list(zip([item[0] for item in author_counts], [item[1] for item in author_counts]))

        # 소속별 논문 수를 계산
        affiliation_counts = []
        if paper_ids:
            placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
            with connection.cursor() as cursor:
                cursor.execute(f"""
                SELECT af.name AS affiliation_name, COUNT(distinct(pa.paper_id)) AS publications_count
                FROM affiliation af
                JOIN paper_affiliation pa ON af.id = pa.affiliation_id
                JOIN paper p ON pa.paper_id = p.id
                WHERE p.id IN ({placeholders})
                GROUP BY af.name
                ORDER BY publications_count DESC
                LIMIT 10;
                """, paper_ids)

                # 데이터를 가져오기
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        # 인코딩 및 디코딩을 생략하고 데이터를 바로 사용
                        affiliation_name = row[0]
                        affiliation_counts.append((affiliation_name, row[1]))
                    except Exception as e:
                        # 문제가 발생한 경우 해당 데이터를 로그로 남김
                        logging.error(f"Error processing affiliation name: {row[0]} - {str(e)}")

        # 소속 이름과 논문 수를 결합한 리스트 생성
        affiliation_data = list(zip([item[0] for item in affiliation_counts], [item[1] for item in affiliation_counts]))

        # 국가별 논문 수 계산
        country_counts = []
        if paper_ids:
            placeholders = ', '.join(['%s'] * len(paper_ids))
            with connection.cursor() as cursor:
                cursor.execute(f"""
                SELECT c.name AS country_name, COUNT(distinct(pc.paper_id)) AS publications_count
                FROM country c
                JOIN paper_country pc ON c.id = pc.country_id
                JOIN paper p ON pc.paper_id = p.id
                WHERE p.id IN ({placeholders})
                GROUP BY c.name
                ORDER BY publications_count DESC
                LIMIT 10;
                """, paper_ids)
                
                country_counts = cursor.fetchall()

        # 국가 이름과 논문 수를 결합한 리스트 생성
        country_data = list(zip([item[0] for item in country_counts], [item[1] for item in country_counts]))

        # 키워드 카운트 초기화
        keyword_counts = Counter()

        if paper_ids:
            placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
            with connection.cursor() as cursor:
                cursor.execute(f"""
                SELECT k.keyword_name
                FROM paper_keyword pk
                JOIN keyword k ON pk.keyword_id = k.id
                WHERE pk.paper_id IN ({placeholders});
                """, paper_ids)

                # 쿼리 결과 가져오기
                keywords = [row[0] for row in cursor.fetchall()]
                keyword_counts.update(keywords)

        # 빈도수가 높은 top 10 키워드 출력하기
        top_keywords = keyword_counts.most_common(10)


        # 그래프에 사용할 데이터 준비
        context = {
            'papers_count': papers_count,
            'author_data': author_data,
            'affiliation_data': affiliation_data,
            'country_data': country_data,
            'top_keywords': top_keywords,
            'keyword': user_keyword,
            'total_results': total_results,
            'start_year': start_year,
            'end_year': end_year,
            'filter': filter_type,
        }

        return render(request, 'total_graph.html', context)
    

def get_paper_ids_country(country):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 파라미터화된 쿼리 사용
    query = """
        SELECT DISTINCT p.id
        FROM paper p
        JOIN paper_country pc ON p.id = pc.paper_id
        JOIN country c ON pc.country_id = c.id
        WHERE c.name = %s;
    """
    # 쿼리 실행
    cursor.execute(query, (country,))

    # 쿼리 결과 가져오기
    paper_ids_country = [row['id'] for row in cursor.fetchall()]
    print(len(paper_ids_country))

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return paper_ids_country


def get_paper_ids_author(author):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 파라미터화된 쿼리 사용
    query = """
        SELECT DISTINCT p.id
        FROM paper p
        JOIN paper_author pa ON p.id = pa.paper_id
        JOIN author a ON pa.author_id = a.id
        WHERE a.name = %s;
    """
    # 쿼리 실행
    cursor.execute(query, (author,))

    # 쿼리 결과 가져오기
    paper_ids_author = [row['id'] for row in cursor.fetchall()]
    print(paper_ids_author)
    print(len(paper_ids_author))

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return paper_ids_author



def author_wordcloud_html(request):
    return render(request, 'author_wordcloud.html')


def get_paper_ids_affiliation(affiliation):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 파라미터화된 쿼리 사용
    query = """
        SELECT DISTINCT p.id
        FROM paper p
        JOIN paper_affiliation pa ON p.id = pa.paper_id
        JOIN affiliation a ON pa.affiliation_id = a.id
        WHERE a.name = %s;
    """
    # 쿼리 실행
    cursor.execute(query, (affiliation,))

    # 쿼리 결과 가져오기
    paper_ids_affiliation = [row['id'] for row in cursor.fetchall()]
    print(paper_ids_affiliation)
    print(len(paper_ids_affiliation))

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return paper_ids_affiliation


def get_authors(paper_id):
    return Author.objects.filter(paperauthor__paper_id=paper_id)

def get_keywords(paper_id):
    return Keyword.objects.filter(paperkeyword__paper_id=paper_id)

def get_affiliations(paper_id):
    return Affiliation.objects.filter(paperaffiliation__paper_id=paper_id)

def get_countries(paper_id):
    return Country.objects.filter(papercountry__paper_id=paper_id)

# 논문 상세 페이지 출력
def paper_detail(request, paper_id):
    paper = get_object_or_404(Paper, id=paper_id)
    authors = get_authors(paper_id)
    keywords = get_keywords(paper_id)
    affiliations = get_affiliations(paper_id)
    countries = get_countries(paper_id)
    
    # 중복 제거
    unique_countries = list(set(countries.values_list('name', flat=True)))
    
    # 논문이 사용자의 저장된 논문인지 확인 (로그인 상태에서만)
    is_saved = False
    if request.user.is_authenticated:
        is_saved = SavedPaper.objects.filter(user=request.user, paper=paper).exists()
        
        # 최근 본 논문 추가 또는 업데이트
        recent_paper, created = RecentPaper.objects.update_or_create(
            user=request.user, paper=paper,
            defaults={'viewed_at': timezone.now()}
        )

        # 최근 본 논문이 20개를 초과한 경우 오래된 논문 삭제
        recent_papers_count = RecentPaper.objects.filter(user=request.user).count()
        if recent_papers_count > 20:
            # 오래된 논문을 가져오고 하나씩 삭제
            oldest_papers = RecentPaper.objects.filter(user=request.user).order_by('viewed_at')[:recent_papers_count - 20]
            for paper_to_delete in oldest_papers:
                paper_to_delete.delete()
            
    context = {
        'paper': paper,
        'authors': authors,
        'keywords': keywords,
        'affiliations': affiliations,
        'countries': unique_countries,
        'is_saved': is_saved,
    }
    return render(request, 'paper_detail.html', context)

# 국가 분석 페이지 html 출력 함수
def country_analyze_html(request):
    return render(request, 'country_analyze.html')

# 국가 네트워크 시각화
def country_network(request):
    try:
        original_country_name = request.GET.get('name')  # 중심 국가의 이름을 지정
        user_keyword = request.GET.get('keyword')  # 검색어를 가져옵니다

        # paper_ids를 가져오기 위한 조건 추가
        if user_keyword:
            paper_ids = country_get_filtered_paper_ids(user_keyword, original_country_name)
            # 검색 결과가 없을 경우 빈 네트워크 데이터 반환
            if not paper_ids:
                return JsonResponse({"nodes": [], "links": [], "center_node": original_country_name})
        else:
            paper_ids = get_paper_ids_country(original_country_name)

        with connection.cursor() as cursor:
            # paper_ids가 있을 경우 쿼리에서 필터링에 추가합니다
            paper_ids_condition = ""
            paper_ids_params = []
            if paper_ids:
                placeholders = ', '.join(['%s'] * len(paper_ids))
                paper_ids_condition = f"AND pa1.paper_id IN ({placeholders}) AND pa2.paper_id IN ({placeholders})"
                paper_ids_params = paper_ids + paper_ids

            query = f"""
            WITH country_paper_count AS (
                SELECT 
                    c.name AS country_name, 
                    COUNT(DISTINCT pa.paper_id) AS total_papers
                FROM 
                    country c
                JOIN 
                    paper_country pa ON c.id = pa.country_id
                GROUP BY 
                    c.name
            )

            SELECT 
                c1.name AS original_country, 
                c2.name AS co_country,
                COUNT(DISTINCT pa1.paper_id) AS num_papers,
                cpc1.total_papers AS original_country_total_papers,
                cpc2.total_papers AS co_country_total_papers
            FROM 
                country c1
            JOIN 
                paper_country pa1 ON c1.id = pa1.country_id
            JOIN 
                paper_country pa2 ON pa1.paper_id = pa2.paper_id AND pa1.country_id != pa2.country_id
            JOIN 
                country c2 ON pa2.country_id = c2.id
            JOIN 
                country_paper_count cpc1 ON c1.name = cpc1.country_name
            JOIN 
                country_paper_count cpc2 ON c2.name = cpc2.country_name
            WHERE 
                c1.name = %s AND c2.name != %s
                {paper_ids_condition}  -- 논문 ID 필터 추가
            GROUP BY 
                c1.name, c2.name, cpc1.total_papers, cpc2.total_papers
            ORDER BY 
                num_papers DESC
            """
            
            cursor.execute(query, [original_country_name, original_country_name] + paper_ids_params)
            rows = cursor.fetchall()

        nodes = []
        links = []
        node_set = set()

        for row in rows:
            original_country = row[0]
            co_country = row[1]
            num_papers = row[2]
            original_country_total_papers = row[3]
            co_country_total_papers = row[4]

            if original_country not in node_set:
                nodes.append({"id": original_country, "total_papers": original_country_total_papers})
                node_set.add(original_country)
            if co_country not in node_set:
                nodes.append({"id": co_country, "total_papers": co_country_total_papers})
                node_set.add(co_country)
            
            links.append({"source": original_country, "target": co_country, "value": num_papers})

        network_data = {"nodes": nodes, "links": links, "center_node": original_country_name}
        return JsonResponse(network_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    
# 워드클라우드 키워드 필터링 적용할 때 사용할 논문 id 가져오는 함수    
def country_get_filtered_paper_ids(user_keyword, country, start_year=2019, end_year=2024):
    # Step 1: 키워드 기반 논문 ID 가져오기
    paper_ids = get_paper_ids(user_keyword, start_year, end_year)
    
    if not paper_ids:
        # 키워드에 맞는 논문이 없으면 빈 결과 반환
        return []

    # Step 2: 나라 이름 기반으로 필터링된 논문 ID 가져오기
    # 나라에 해당하는 논문 ID만 paper_ids에서 필터링
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    # 나라 필터를 적용하여 paper_ids 중에서 해당 국가의 논문만 가져오기
    placeholders = ', '.join(['%s'] * len(paper_ids))
    query = f"""
        SELECT DISTINCT p.id
        FROM paper p
        JOIN paper_country pc ON p.id = pc.paper_id
        JOIN country c ON pc.country_id = c.id
        WHERE p.id IN ({placeholders}) AND c.name = %s;
    """
    params = paper_ids + [country]
    cursor.execute(query, params)

    # 최종 필터링된 논문 ID 가져오기
    filtered_paper_ids = [row['id'] for row in cursor.fetchall()]

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return filtered_paper_ids
    
# 국가 워드클라우드
def country_wordcloud(request):
    country = request.GET.get('name')
    user_keyword = request.GET.get('keyword', '').strip()  # 사용자가 입력한 키워드

    # 키워드 카운트 초기화
    keyword_counts = Counter()

    # 검색어가 있으면 필터링된 논문 ID를 가져오고, 없으면 기본 국가 기준으로 논문 ID를 가져옴
    if user_keyword:
        paper_ids = country_get_filtered_paper_ids(user_keyword, country)
    else:
        paper_ids = get_paper_ids_country(country)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
        # MariaDB 연결
        db = mariadb.connect(**db_config)
        cursor = db.cursor()

        # 논문 ID에 해당하는 키워드를 가져오기
        cursor.execute(f"""
            SELECT k.keyword_name
            FROM paper_keyword pk
            JOIN keyword k ON pk.keyword_id = k.id
            WHERE pk.paper_id IN ({placeholders});
        """, paper_ids)

        # 쿼리 결과 가져오기
        keywords = [row[0] for row in cursor.fetchall()]
        keyword_counts.update(keywords)

        # 데이터베이스 연결 종료
        cursor.close()
        db.close()

    # 빈도수가 높은 top 20 키워드 출력하기
    top_keywords = keyword_counts.most_common(20)
    top_keywords_list = [[keyword, count] for keyword, count in top_keywords]

    return JsonResponse(top_keywords_list, safe=False)

# 국가의 연도별 논문 수 함수
def country_get_paper_counts_by_year(request):
    country_name = request.GET.get('name')
    user_keyword = request.GET.get('keyword', '').strip()  # 사용자가 입력한 키워드

    # 키워드가 있으면 필터링된 논문 ID 가져오기
    if user_keyword:
        paper_ids = country_get_filtered_paper_ids(user_keyword, country_name)
    else:
        paper_ids = get_paper_ids_country(country_name)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT YEAR(p.date) AS year, COUNT(DISTINCT p.id) AS paper_count
                FROM paper p
                JOIN paper_country pc ON p.id = pc.paper_id
                JOIN country c ON pc.country_id = c.id
                WHERE c.name = %s AND p.id IN ({placeholders})
                GROUP BY YEAR(p.date)
                ORDER BY YEAR(p.date);
            """, [country_name] + paper_ids)
            
            rows = cursor.fetchall()
    else:
        rows = []

    # 데이터를 JSON 형식으로 변환
    data = [{'year': row[0], 'paper_count': row[1]} for row in rows]
    
    return JsonResponse(data, safe=False)

# 국가 논문 리스트
def country_get_recent_papers(request):
    country_name = request.GET.get('name')
    keyword = request.GET.get('keyword', '')

    # 논문 ID 필터링 (키워드에 따라 논문 ID 가져오기)
    paper_ids_by_keyword = get_paper_ids(keyword) if keyword else None

    # 데이터베이스 쿼리
    with connection.cursor() as cursor:
        query = """
            SELECT DISTINCT p.id, p.title, p.url, p.date, p.citations, p.publisher, p.abstract, p.saved_count
            FROM paper p
            JOIN paper_country pc ON p.id = pc.paper_id
            JOIN country c ON pc.country_id = c.id
            WHERE c.name = %s
        """
        params = [country_name]

        # 키워드 필터가 있는 경우
        if paper_ids_by_keyword:
            query += " AND p.id IN %s"
            params.append(tuple(paper_ids_by_keyword))

        # 좋아요 수로 정렬
        query += " ORDER BY p.saved_count DESC LIMIT 5"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()

    # JSON 변환
    data = [
        {
            'id': row[0],
            'title': row[1],
            'url': row[2],
            'date': row[3],
            'citations': row[4],
            'publisher': row[5],
            'abstract': row[6],
            'saved_count': row[7]
        }
        for row in rows
    ]
    return JsonResponse(data, safe=False)  

# 국가 발표 논문 수
def country_get_total_papers(request):
    country_name = request.GET.get('name')
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(DISTINCT pc.paper_id) AS total_papers
            FROM paper_country pc
            JOIN country c ON pc.country_id = c.id
            WHERE c.name = %s;
        """, [country_name])
        
        row = cursor.fetchone()
    
    # 데이터를 JSON 형식으로 변환
    data = {'total_papers': row[0]}
    
    return JsonResponse(data)

#GPT API 사용 함수
class AnalyzeNetworkData(CsrfExemptMixin, APIView):
    authentication_classes = []

    def post(self, request, format=None):
        network_data = request.data.get('network_data', '')

        if network_data:
            prompt = f"이 네트워크 데이터에서 노드 간의 상호작용 패턴을 식별하고, 주요 관계 및 특징적인 연결을 요약해 주세요: {network_data}"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # 또는 gpt-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            analysis_result = response.choices[0].message['content'].strip()

            return Response({'analysis_result': analysis_result})
        return Response({'error': 'Invalid request'}, status=400)

#GPT API 사용 함수
class AnalyzeKeywordData(CsrfExemptMixin, APIView):
    authentication_classes = []

    def post(self, request, format=None):
        keyword_data = request.data.get('keyword_data', '')

        if keyword_data:
            prompt = f"다음 키워드 데이터를 바탕으로 주요 주제와 잠재적인 하위 주제를 설명하고, 각 키워드가 어떤 역할을 하는지 분석해 주세요: {keyword_data}"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # 또는 gpt-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            analysis_result = response.choices[0].message['content'].strip()

            return Response({'analysis_result': analysis_result})
        
        return Response({'error': 'Invalid request'}, status=400)

# 소속 분석 페이지 html 출력 함수
def affiliation_analyze_html(request):
    return render(request, 'affiliation_analyze.html')

def affiliation_get_filtered_paper_ids(user_keyword, affiliation):
    # Step 1: 키워드 기반 논문 ID 가져오기
    paper_ids = get_paper_ids(user_keyword)

    if not paper_ids:
        # 키워드에 맞는 논문이 없으면 빈 결과 반환
        return []

    # Step 2: 소속 필터를 적용하여 논문 ID 필터링
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    placeholders = ', '.join(['%s'] * len(paper_ids))
    query = f"""
        SELECT DISTINCT p.id
        FROM paper p
        JOIN paper_affiliation pa ON p.id = pa.paper_id
        JOIN affiliation a ON pa.affiliation_id = a.id
        WHERE p.id IN ({placeholders}) AND a.name = %s;
    """
    params = paper_ids + [affiliation]
    cursor.execute(query, params)

    # 최종 필터링된 논문 ID 가져오기
    filtered_paper_ids = [row['id'] for row in cursor.fetchall()]

    cursor.close()
    db.close()

    return filtered_paper_ids

# 소속 네트워크 시각화
def affiliation_network(request):
    try:
        original_affiliation_name = request.GET.get('name')
        user_keyword = request.GET.get('keyword', '').strip()

        # 논문 ID 필터링 적용
        if user_keyword:
            paper_ids = affiliation_get_filtered_paper_ids(user_keyword, original_affiliation_name)
        else:
            paper_ids = get_paper_ids_affiliation(original_affiliation_name)

        if not paper_ids:
            return JsonResponse({"nodes": [], "links": []})

        placeholders = ', '.join(['%s'] * len(paper_ids))
        
        with connection.cursor() as cursor:
            query = f"""
            WITH affiliation_paper_count AS (
                SELECT 
                    a.name AS affiliation_name, 
                    COUNT(DISTINCT pa.paper_id) AS total_papers
                FROM 
                    affiliation a
                JOIN 
                    paper_affiliation pa ON a.id = pa.affiliation_id
                GROUP BY 
                    a.name
            )
            SELECT 
                a1.name AS original_affiliation, 
                a2.name AS co_affiliation,
                COUNT(DISTINCT pa1.paper_id) AS num_papers,
                apc1.total_papers AS original_affiliation_total_papers,
                apc2.total_papers AS co_affiliation_total_papers
            FROM 
                affiliation a1
            JOIN 
                paper_affiliation pa1 ON a1.id = pa1.affiliation_id
            JOIN 
                paper_affiliation pa2 ON pa1.paper_id = pa2.paper_id AND pa1.affiliation_id != pa2.affiliation_id
            JOIN 
                affiliation a2 ON pa2.affiliation_id = a2.id
            JOIN 
                affiliation_paper_count apc1 ON a1.name = apc1.affiliation_name
            JOIN 
                affiliation_paper_count apc2 ON a2.name = apc2.affiliation_name
            WHERE 
                a1.name = %s AND a2.name != %s AND pa1.paper_id IN ({placeholders})
            GROUP BY 
                a1.name, a2.name, apc1.total_papers, apc2.total_papers
            ORDER BY 
                num_papers DESC;
            """
            
            cursor.execute(query, [original_affiliation_name, original_affiliation_name] + paper_ids)
            rows = cursor.fetchall()

        nodes, links, node_set = [], [], set()

        for row in rows:
            original_affiliation, co_affiliation, num_papers = row[0], row[1], row[2]
            original_total_papers, co_total_papers = row[3] , row[4]

            if original_affiliation not in node_set:
                nodes.append({"id": original_affiliation, "total_papers": original_total_papers})
                node_set.add(original_affiliation)
            if co_affiliation not in node_set:
                nodes.append({"id": co_affiliation, "total_papers": co_total_papers})
                node_set.add(co_affiliation)
            
            links.append({"source": original_affiliation, "target": co_affiliation, "value": num_papers})

        return JsonResponse({"nodes": nodes, "links": links, "center_node": original_affiliation_name})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
# 소속 워드클라우드
def affiliation_wordcloud(request):
    affiliation = request.GET.get('name')
    user_keyword = request.GET.get('keyword', '').strip()  # 키워드 파라미터 추가
    keyword_counts = Counter()

    # 키워드가 있을 경우 필터링된 논문 ID를 가져옴
    if user_keyword:
        paper_ids = affiliation_get_filtered_paper_ids(user_keyword, affiliation)
    else:
        paper_ids = get_paper_ids_affiliation(affiliation)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))
        
        # 데이터베이스 연결
        db = mariadb.connect(**db_config)
        cursor = db.cursor()
        
        cursor.execute(f"""
            SELECT k.keyword_name
            FROM paper_keyword pk
            JOIN keyword k ON pk.keyword_id = k.id
            WHERE pk.paper_id IN ({placeholders});
        """, paper_ids)

        # 쿼리 결과를 가져와서 키워드 카운트 업데이트
        keywords = [row[0] for row in cursor.fetchall()]
        keyword_counts.update(keywords)

        # 데이터베이스 연결 종료
        cursor.close()
        db.close()

    # 상위 20개 키워드만 반환
    top_keywords = keyword_counts.most_common(20)
    top_keywords_list = [[keyword, count] for keyword, count in top_keywords]

    return JsonResponse(top_keywords_list, safe=False)

# 소속 연도별 논문 수
def affiliation_get_paper_counts_by_year(request):
    affiliation_name = request.GET.get('name')
    user_keyword = request.GET.get('keyword', '').strip()

    # 검색어가 있는 경우 필터링된 paper_ids 가져오기
    if user_keyword:
        paper_ids = affiliation_get_filtered_paper_ids(user_keyword, affiliation_name)
        # 검색 결과가 없을 경우 빈 리스트 반환
        if not paper_ids:
            return JsonResponse([], safe=False)
        
        placeholders = ', '.join(['%s'] * len(paper_ids))
        query_filter = f"AND p.id IN ({placeholders})"
        query_params = [affiliation_name] + paper_ids
    else:
        # 검색어가 없을 경우 소속 이름만으로 필터링
        query_filter = ""
        query_params = [affiliation_name]

    with connection.cursor() as cursor:
        cursor.execute(f"""
            SELECT YEAR(p.date) AS year, COUNT(DISTINCT p.id) AS paper_count
            FROM paper p
            JOIN paper_affiliation pa ON p.id = pa.paper_id
            WHERE pa.affiliation_id IN (SELECT id FROM affiliation WHERE name = %s)
            {query_filter}
            GROUP BY YEAR(p.date)
            ORDER BY YEAR(p.date);
        """, query_params)
        
        rows = cursor.fetchall()
    
    data = [{'year': row[0], 'paper_count': row[1]} for row in rows]
    return JsonResponse(data, safe=False)

#소속 논문 리스트 함수
def affiliation_get_recent_papers(request):
    affiliation_name = request.GET.get('name')
    keyword = request.GET.get('keyword', '')

    # 논문 ID 필터링 (키워드에 따라 논문 ID 가져오기)
    paper_ids_by_keyword = get_paper_ids(keyword) if keyword else None

    # 데이터베이스 쿼리
    with connection.cursor() as cursor:
        query = """
            SELECT DISTINCT p.id, p.title, p.url, p.date, p.citations, p.publisher, p.abstract, p.saved_count
            FROM paper p
            JOIN paper_affiliation pa ON p.id = pa.paper_id
            JOIN affiliation a ON pa.affiliation_id = a.id
            WHERE a.name = %s
        """
        params = [affiliation_name]

        # 키워드 필터가 있는 경우
        if paper_ids_by_keyword:
            query += " AND p.id IN %s"
            params.append(tuple(paper_ids_by_keyword))

        # 좋아요 수로 정렬
        query += " ORDER BY p.saved_count DESC LIMIT 5"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()

    # JSON 변환
    data = [
        {
            'id': row[0],
            'title': row[1],
            'url': row[2],
            'date': row[3],
            'citations': row[4],
            'publisher': row[5],
            'abstract': row[6],
            'saved_count': row[7]
        }
        for row in rows
    ]
    return JsonResponse(data, safe=False)

#소속 발표 논문 수
def affiliation_get_total_papers(request):
    affiliation_name = request.GET.get('name')
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(DISTINCT pa.paper_id) AS total_papers
            FROM paper_affiliation pa
            JOIN affiliation a ON pa.affiliation_id = a.id
            WHERE a.name = %s;
        """, [affiliation_name])
        
        row = cursor.fetchone()
    
    data = {'total_papers': row[0]}
    
    return JsonResponse(data)

# 저자 분석 페이지 html 출력 함수
def author_analyze_html(request):
    return render(request, 'author_analyze.html')


# 저자 워드클라우드 키워드 필터링 적용할 때 사용할 논문 id 가져오는 함수
def author_get_filtered_paper_ids(user_keyword, author, start_year=2019, end_year=2024):
    # Step 1: 키워드와 연도 범위 기반 논문 ID 가져오기
    paper_ids = get_paper_ids(user_keyword, start_year, end_year)
    
    if not paper_ids:
        # 키워드에 맞는 논문이 없으면 빈 결과 반환
        return []

    # Step 2: 저자 이름 기반으로 필터링된 논문 ID 가져오기
    # 저자에 해당하는 논문 ID만 paper_ids에서 필터링
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    # 저자 필터를 적용하여 paper_ids 중에서 해당 저자의 논문만 가져오기
    placeholders = ', '.join(['%s'] * len(paper_ids))
    query = f"""
        SELECT DISTINCT p.id
        FROM paper p
        JOIN paper_author pa ON p.id = pa.paper_id
        JOIN author a ON pa.author_id = a.id
        WHERE p.id IN ({placeholders}) AND a.name = %s;
    """
    params = paper_ids + [author]
    cursor.execute(query, params)

    # 최종 필터링된 논문 ID 가져오기
    filtered_paper_ids = [row['id'] for row in cursor.fetchall()]

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return filtered_paper_ids


# 저자 네트워크 시각화
def author_network(request):
    original_author_name = request.GET.get('name')
    user_keyword = request.GET.get('keyword')

    # 키워드에 따라 필터링된 논문 ID 가져오기
    if user_keyword:
        paper_ids = author_get_filtered_paper_ids(user_keyword, original_author_name)
        if not paper_ids:
            return JsonResponse({"nodes": [], "links": [], "center_node": original_author_name})
    else:
        paper_ids = get_paper_ids_author(original_author_name)

    with connection.cursor() as cursor:
        placeholders = ', '.join(['%s'] * len(paper_ids))
        query = f"""
            WITH author_paper_count AS (
                SELECT 
                    a.name AS author_name, 
                    COUNT(pa.paper_id) AS total_papers
                FROM 
                    author a
                JOIN 
                    paper_author pa ON a.id = pa.author_id
                GROUP BY 
                    a.name
            )
            SELECT 
                a1.name AS original_author, 
                a2.name AS co_author,
                COUNT(p.id) AS num_papers,
                apc1.total_papers AS original_author_total_papers,
                apc2.total_papers AS co_author_total_papers
            FROM 
                author a1
            JOIN 
                paper_author pa1 ON a1.id = pa1.author_id
            JOIN 
                paper_author pa2 ON pa1.paper_id = pa2.paper_id
            JOIN 
                author a2 ON pa2.author_id = a2.id
            JOIN 
                paper p ON pa1.paper_id = p.id
            JOIN 
                author_paper_count apc1 ON a1.name = apc1.author_name
            JOIN 
                author_paper_count apc2 ON a2.name = apc2.author_name
            WHERE 
                a1.name = %s AND a2.name != %s
                AND pa1.paper_id IN ({placeholders}) AND pa2.paper_id IN ({placeholders})
            GROUP BY 
                a1.name, a2.name, apc1.total_papers, apc2.total_papers
            ORDER BY 
                num_papers DESC;
        """
        cursor.execute(query, [original_author_name, original_author_name] + paper_ids * 2)
        rows = cursor.fetchall()

    nodes = []
    links = []
    node_set = set()

    for row in rows:
        original_author = row[0]
        co_author = row[1]
        num_papers = row[2]
        original_author_total_papers = row[3]
        co_author_total_papers = row[4]

        if original_author not in node_set:
            nodes.append({"id": original_author, "total_papers": original_author_total_papers})
            node_set.add(original_author)
        if co_author not in node_set:
            nodes.append({"id": co_author, "total_papers": co_author_total_papers})
            node_set.add(co_author)
        
        links.append({"source": original_author, "target": co_author, "value": num_papers})

    network_data = {"nodes": nodes, "links": links, "center_node": original_author_name}
    return JsonResponse(network_data)
    
# 저자 워드클라우드 시각화
def author_wordcloud(request):
    author = request.GET.get('name')
    user_keyword = request.GET.get('keyword', '').strip()
    keyword_counts = Counter()

    if user_keyword:
        paper_ids = author_get_filtered_paper_ids(user_keyword, author)
    else:
        paper_ids = get_paper_ids_author(author)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))
        db = mariadb.connect(**db_config)
        cursor = db.cursor()
        cursor.execute(f"""
            SELECT k.keyword_name
            FROM paper_keyword pk
            JOIN keyword k ON pk.keyword_id = k.id
            WHERE pk.paper_id IN ({placeholders});
        """, paper_ids)

        keywords = [row[0] for row in cursor.fetchall()]
        keyword_counts.update(keywords)

        cursor.close()
        db.close()

    top_keywords = keyword_counts.most_common(20)
    top_keywords_list = [[keyword, count] for keyword, count in top_keywords]

    return JsonResponse(top_keywords_list, safe=False)

# 저자 연도별 논문 수
def author_get_paper_counts_by_year(request):
    author_name = request.GET.get('name')
    user_keyword = request.GET.get('keyword', '').strip()

    if user_keyword:
        paper_ids = author_get_filtered_paper_ids(user_keyword, author_name)
    else:
        paper_ids = get_paper_ids_author(author_name)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT YEAR(p.date) AS year, COUNT(DISTINCT p.id) AS paper_count
                FROM paper p
                JOIN paper_author pa ON p.id = pa.paper_id
                WHERE pa.author_id IN (SELECT id FROM author WHERE name = %s)
                AND p.id IN ({placeholders})
                GROUP BY YEAR(p.date)
                ORDER BY YEAR(p.date);
            """, [author_name] + paper_ids)
            rows = cursor.fetchall()
    else:
        rows = []

    data = [{'year': row[0], 'paper_count': row[1]} for row in rows]
    return JsonResponse(data, safe=False)

# 저자 논문 리스트 함수
def author_get_recent_papers(request):
    author_name = request.GET.get('name')
    keyword = request.GET.get('keyword', '')

    # 논문 ID 필터링 (키워드에 따라 논문 ID 가져오기)
    paper_ids_by_keyword = get_paper_ids(keyword) if keyword else None

    # 데이터베이스 쿼리
    with connection.cursor() as cursor:
        query = """
            SELECT DISTINCT p.id, p.title, p.url, p.date, p.citations, p.publisher, p.abstract, p.saved_count
            FROM paper p
            JOIN paper_author pa ON p.id = pa.paper_id
            JOIN author a ON pa.author_id = a.id
            WHERE a.name = %s
        """
        params = [author_name]

        # 키워드 필터가 있는 경우
        if paper_ids_by_keyword:
            query += " AND p.id IN %s"
            params.append(tuple(paper_ids_by_keyword))

        # 좋아요 수로 정렬
        query += " ORDER BY p.saved_count DESC LIMIT 5"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()

    # JSON 변환
    data = [
        {
            'id': row[0],
            'title': row[1],
            'url': row[2],
            'date': row[3],
            'citations': row[4],
            'publisher': row[5],
            'abstract': row[6],
            'saved_count': row[7]
        }
        for row in rows
    ]
    return JsonResponse(data, safe=False)

#저자 발표 논문 수
def author_get_total_papers(request):
    author_name = request.GET.get('name')
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(DISTINCT pa.paper_id) AS total_papers
            FROM paper_author pa
            JOIN author a ON pa.author_id = a.id
            WHERE a.name = %s;
        """, [author_name])
        
        row = cursor.fetchone()
    
    data = {'total_papers': row[0]}
    
    return JsonResponse(data)

def author_get_affiliation(request):
    author_name = request.GET.get('name')  # 저자의 이름을 요청 파라미터로 받음
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT a.affiliation
            FROM author a
            WHERE a.name = %s;
        """, [author_name])
        
        row = cursor.fetchone()
    
    # 저자가 존재할 경우, 소속 정보 반환
    if row:
        affiliation = row[0]
        return JsonResponse({'affiliation': affiliation})
    else:
        return JsonResponse({'error': '저자를 찾을 수 없습니다.'}, status=404)
    
# 로그인 html 출력 함수
def login_html(request):
    return render(request, 'login.html')

# 회원가입 html 출력 함수
def signup_html(request):
    return render(request, 'signup.html')

# @login_required
# def mypage_html(request):
#     return render(request, 'mypage.html')

##################################################################################################################################

# 유저가 저장한 논문들의 아이디 가져오기
def get_user_savedpaper_ids(username):
    # 데이터베이스 연결
    conn = mariadb.connect(**db_config)
    cur = conn.cursor()
    
    # SQL 쿼리 실행
    query = """
        SELECT sp.paper_id
        FROM main_savedpaper sp
        JOIN auth_user au ON sp.user_id = au.id
        WHERE au.username = %s
    """
    
    cur.execute(query, (username,))
    
    # 결과 가져오기: 각 튜플의 첫 번째 요소인 paper_id만 추출
    paper_ids = [row[0] for row in cur.fetchall()]
    
    cur.close()
    conn.close()

    return paper_ids

# 유저가 저장한 논문들 중에서 tf-idf 수치(가장 많이 사용된 단어)가 높은 단어 3개 추출
def top3_word(username):
    # 검색 엔진을 통해 논문 ID 가져오기
    paper_ids = get_user_savedpaper_ids(username)

    # 논문 ID를 사용하여 abstract 가져오기
    abstract_data = get_abstracts(paper_ids)

    # abstract 컬럼 내용 추출
    abstracts = abstract_data["abstract"].tolist()

    if not abstracts or all(not abstract.strip() for abstract in abstracts):
        return []

    # TF-IDF 벡터화 (불용어 처리)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)
    except ValueError:
        # TF-IDF에 유효한 단어가 없을 때 빈 리스트 반환
        return []


    # 각 단어의 인덱스 확인
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 데이터프레임으로 변환
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # 각 단어의 총 TF-IDF 값을 계산
    tfidf_sums = tfidf_df.sum(axis=0)

    # TF-IDF 값으로 정렬하여 상위 20개 단어 선택
    top_3_words = tfidf_sums.sort_values(ascending=False).head(3).index.tolist()

    return top_3_words

# 논문 제목으로 검색하는 엔진 (매치율 * (saved_count + 1) 계산 추가)
def get_paper_ids_with_weighted_score(user_keyword, start_year=2019, end_year=2024):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    cursor = db.cursor(dictionary=True)

    # 기본 쿼리
    query_conditions = []
    query_params = []

    # 검색 키워드 조건 추가
    if len(user_keyword) <= 2:  # 검색 키워드 길이가 2 이하일 때
        query_conditions.append("""
            (search LIKE %s OR title LIKE %s OR abstract LIKE %s
             OR EXISTS (SELECT 1 FROM paper_keyword pk
                        JOIN keyword k ON pk.keyword_id = k.id
                        WHERE pk.paper_id = p.id AND k.keyword_name LIKE %s))
        """)
        query_params.extend([f'% {user_keyword} %', f'% {user_keyword} %', f'% {user_keyword} %', f'% {user_keyword} %'])
    else:  # 검색 키워드 길이가 3 이상일 때
        query_conditions.append("""
            (MATCH(search, title, abstract) AGAINST(%s)
             OR EXISTS (SELECT 1 FROM paper_keyword pk
                        JOIN keyword k ON pk.keyword_id = k.id
                        WHERE pk.paper_id = p.id AND MATCH(k.keyword_name) AGAINST(%s)))
        """)
        query_params.extend([user_keyword, user_keyword])

    # 연도 필터링 조건 추가 (기본값: 2019 ~ 2024)
    query_conditions.append("YEAR(p.date) BETWEEN %s AND %s")
    query_params.extend([start_year, end_year])

    # 쿼리 실행 (매치율과 saved_count 함께 계산)
    query = f"""
        SELECT p.id, 
               MATCH(search, title, abstract) AGAINST(%s IN NATURAL LANGUAGE MODE) AS relevance,
               (MATCH(search, title, abstract) AGAINST(%s IN NATURAL LANGUAGE MODE) * (p.saved_count + 1)) AS weighted_score
        FROM paper p
        WHERE {' AND '.join(query_conditions)}
        ORDER BY weighted_score DESC;
    """
    
    # 쿼리 실행 및 파라미터 전달
    cursor.execute(query, [user_keyword, user_keyword] + query_params)

    # 쿼리 결과 가져오기
    results = [{'id': row['id'], 'relevance': row['relevance'], 'weighted_score': row['weighted_score']} for row in cursor.fetchall()]

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return results

# 추천 논문 로직을 추가하는 함수
def get_recommended_papers_for_user(username, start_year=2019, end_year=2024):
    # 사용자에 대한 상위 3개의 키워드 추출
    top_keywords = top3_word(username)
    
    # 각 키워드에 대한 추천 논문 ID와 점수 저장
    recommended_papers = []

    for keyword in top_keywords:
        # 각 키워드에 대해 논문 ID를 가져오고, 점수로 상위 5개 선택
        paper_ids_with_scores = get_paper_ids_with_weighted_score(keyword, start_year, end_year)
        top_5_papers = sorted(paper_ids_with_scores, key=lambda x: x['weighted_score'], reverse=True)[:5]
        
        # 키워드와 논문 ID 리스트 (상위 5개)를 함께 저장
        recommended_papers.append({
            'keyword': keyword,
            'top_papers': [{'id': paper['id'], 'score': paper['weighted_score']} for paper in top_5_papers]
        })

    return recommended_papers

##################################################################################################################################

def mypage_html(request):
    if not request.user.is_authenticated:
        messages.warning(request, '로그인 후 접근 가능한 페이지입니다.')
        return redirect(f"{reverse('accounts:login')}?next={request.path}")
    
    # 로그인된 사용자에 맞는 데이터를 불러옴 (저장 논문, 최근 본 논문 등)
    saved_papers = SavedPaper.objects.filter(user=request.user)  # 저장된 논문 불러오기
    recent_papers = RecentPaper.objects.filter(user=request.user)  # 최근 본 논문 불러오기
    recommended_papers = get_recommended_papers_for_user(request.user)  # 예시: 추천 논문 불러오기

    context = {
        'saved_papers': saved_papers,
        'recent_papers': recent_papers,
        'recommended_papers': recommended_papers,
    }

    return render(request, 'mypage.html', context)

@login_required
def recommended_papers(request):
    # 상위 3개 키워드 추출
    top_keywords = top3_word(request.user.username)
    
    # 사용자가 저장한 논문 ID 가져오기
    saved_paper_ids = SavedPaper.objects.filter(user=request.user).values_list('paper_id', flat=True)
    
    # 각 키워드마다 저장되지 않은 상위 5개의 논문을 추천하여 수집
    recommended_papers = []
    for keyword in top_keywords:
        paper_ids = get_paper_ids_with_weighted_score(keyword)
        
        # 추천 논문 중에서 저장되지 않은 논문만 필터링하여 상위 5개 선택
        filtered_paper_ids = [paper['id'] for paper in paper_ids if paper['id'] not in saved_paper_ids][:5]
        papers = Paper.objects.filter(id__in=filtered_paper_ids)

        papers_with_details = []
        for paper in papers:
            authors = Author.objects.filter(paperauthor__paper_id=paper.id)
            keywords = Keyword.objects.filter(paperkeyword__paper_id=paper.id)
            affiliations = Affiliation.objects.filter(paperaffiliation__paper_id=paper.id)

            # 국가 정보 중복 제거
            countries = PaperCountry.objects.filter(paper_id=paper.id).select_related('country')
            unique_countries = list(set([country.country.name for country in countries]))

            papers_with_details.append({
                'paper': paper,
                'authors': authors,
                'keywords': keywords,
                'affiliations': affiliations,
                'countries': unique_countries,
                'is_saved': False,  # 추천 논문이므로 저장되지 않음으로 설정
            })

        recommended_papers.append({
            'keyword': keyword,
            'papers': papers_with_details,
        })

    context = {
        'recommended_papers': recommended_papers,
    }

    return render(request, 'recommended_papers.html', context)


# 논문 저장하기
@login_required
@require_POST
def save_paper(request):
    paper_id = request.POST.get('paper_id')

    if not paper_id:
        return JsonResponse({'success': False, 'message': '논문 ID가 제공되지 않았습니다.'})

    try:
        paper = Paper.objects.get(id=paper_id)
        saved_paper, created = SavedPaper.objects.get_or_create(user=request.user, paper=paper)
        if created:
            # 논문 저장 시 saved_count 증가
            paper.saved_count += 1
            paper.save()
            return JsonResponse({'success': True, 'message': '논문이 저장되었습니다.'})
        else:
            return JsonResponse({'success': False, 'message': '이미 저장된 논문입니다.'})
    except Paper.DoesNotExist:
        return JsonResponse({'success': False, 'message': '논문을 찾을 수 없습니다.'})
    
# 논문 저장하기 - 체크박스 (여러개)
@login_required
@require_POST
def save_selected_papers(request):
    try:
        # POST 요청에서 선택된 논문들의 ID 리스트를 받음
        selected_papers = request.POST.getlist('selected_papers[]')  # name이 selected_papers[]인 데이터를 가져옴
        
        # 선택된 논문들이 있는지 확인
        if not selected_papers:
            return JsonResponse({'success': False, 'message': '저장할 논문이 선택되지 않았습니다.'})

        # 각 논문을 저장
        saved_count = 0
        already_saved_count = 0
        for paper_id in selected_papers:
            try:
                paper = Paper.objects.get(id=paper_id)
                # 이미 저장된 논문이 있는지 확인 후 없으면 저장
                saved_paper, created = SavedPaper.objects.get_or_create(user=request.user, paper=paper)
                if created:
                    saved_count += 1
                    paper.saved_count += 1
                    paper.save()
                else:
                    already_saved_count += 1
            except Paper.DoesNotExist:
                # 논문이 존재하지 않는 경우 건너뜀
                continue

        # 결과 반환
        message = f'{saved_count}개의 논문이 저장되었습니다.'
        if already_saved_count > 0:
            message += f' {already_saved_count}개의 논문은 이미 저장되었습니다.'

        return JsonResponse({'success': True, 'message': message})

    except Exception as e:
        return JsonResponse({'success': False, 'message': f'오류가 발생했습니다: {str(e)}'})
    
# 논문 저장 삭제하기
@login_required
@require_POST
def remove_paper(request):
    paper_id = request.POST.get('paper_id')

    if not paper_id:
        return JsonResponse({'success': False, 'message': '논문 ID가 제공되지 않았습니다.'})

    try:
        paper = Paper.objects.get(id=paper_id)
        saved_paper = SavedPaper.objects.filter(user=request.user, paper=paper)
        if saved_paper.exists():
            # 논문 삭제 시 saved_count 감소
            saved_paper.delete()
            paper.saved_count = max(0, paper.saved_count - 1)  # saved_count가 음수가 되지 않도록
            paper.save()
            return JsonResponse({'success': True, 'message': '논문이 삭제되었습니다.'})
        else:
            return JsonResponse({'success': False, 'message': '저장되지 않은 논문입니다.'})
    except Paper.DoesNotExist:
        return JsonResponse({'success': False, 'message': '논문을 찾을 수 없습니다.'})
    
# 논문 삭제하기 - 체크박스 (여러개)
@login_required
@require_POST
def remove_selected_papers(request):
    try:
        # POST 요청에서 선택된 논문들의 ID 리스트를 받음
        selected_papers = request.POST.getlist('selected_papers[]')  # name이 selected_papers[]인 데이터를 가져옴

        # 선택된 논문들이 있는지 확인
        if not selected_papers:
            return JsonResponse({'success': False, 'message': '삭제할 논문이 선택되지 않았습니다.'})

        # 각 논문을 삭제
        removed_count = 0
        not_saved_count = 0
        for paper_id in selected_papers:
            try:
                paper = Paper.objects.get(id=paper_id)
                saved_paper = SavedPaper.objects.filter(user=request.user, paper=paper)
                if saved_paper.exists():
                    # 논문 삭제 시 saved_count 감소
                    saved_paper.delete()
                    paper.saved_count = max(0, paper.saved_count - 1)  # saved_count가 음수가 되지 않도록
                    paper.save()
                    removed_count += 1
                else:
                    not_saved_count += 1
            except Paper.DoesNotExist:
                # 논문이 존재하지 않는 경우 건너뜀
                continue

        # 결과 반환
        message = f'{removed_count}개의 논문이 삭제되었습니다.'
        if not_saved_count > 0:
            message += f' {not_saved_count}개의 논문은 저장된 기록이 없습니다.'

        return JsonResponse({'success': True, 'message': message})

    except Exception as e:
        return JsonResponse({'success': False, 'message': f'오류가 발생했습니다: {str(e)}'})
    
# 마이페이지 - 저장된 논문
@login_required
def saved_papers(request):
    query = request.GET.get('query', '')
    order = request.GET.get('order', 'desc')  # 정렬 순서 ('desc' 또는 'asc')
    items_per_page = int(request.GET.get('items_per_page', 10))  # 기본값 10
    filter_type = request.GET.get('filter', 'paper')

    # 로그인된 사용자의 저장된 논문을 저장된 시간 기준으로 가져오기
    saved_papers = SavedPaper.objects.filter(user=request.user).select_related('paper').order_by(
        '-saved_at' if order == 'desc' else 'saved_at'
    )

    paper_ids = []
    
    # 검색어가 있으면 필터(검색창-콤보박스)에 따라 논문 검색
    if query:
        if filter_type == 'paper':
            paper_ids = get_paper_ids(query)
        elif filter_type == 'author':
            paper_ids = get_author_paper_ids(query)
        elif filter_type == 'country':
            paper_ids = get_country_paper_ids(query)
        saved_papers = saved_papers.filter(paper_id__in=paper_ids)
        
    # 페이징 처리
    paginator = Paginator(saved_papers, items_per_page)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 페이지 그룹 계산 (10개 단위)
    current_page = page_obj.number
    current_group_start = (current_page - 1) // 10 * 10 + 1
    current_group_end = min(current_group_start + 9, paginator.num_pages)

    # 페이지 그룹 리스트 생성
    current_group_pages = list(range(current_group_start, current_group_end + 1))

    # 저자 및 키워드 추가
    papers_with_authors_and_keywords = []
    for saved_paper in page_obj:
        paper = saved_paper.paper
        saved_at = saved_paper.saved_at
        authors = Author.objects.filter(paperauthor__paper_id=paper.id)
        keywords = Keyword.objects.filter(paperkeyword__paper_id=paper.id)
        affiliations = Affiliation.objects.filter(paperaffiliation__paper_id=paper.id)

        # 논문 내 국가 정보 수집 및 중복 제거
        countries = PaperCountry.objects.filter(paper_id=paper.id).select_related('country')
        unique_countries = list(set([country.country.name for country in countries]))

        papers_with_authors_and_keywords.append({
            'paper': paper,
            'saved_at': saved_at,
            'authors': authors,
            'keywords': keywords,
            'affiliations': affiliations,
            'countries': unique_countries,
            'is_saved': True,
        })

    context = {
        'query': query,
        'filter': filter_type,
        'papers_with_authors_and_keywords': papers_with_authors_and_keywords,
        'page_obj': page_obj,
        'order': order,
        'items_per_page': items_per_page,
        'current_group_pages': current_group_pages,
    }

    # AJAX 요청일 경우 검색된 논문 부분만 렌더링
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        html = render_to_string('saved_papers.html', context, request=request)
        return HttpResponse(html)
    
    return render(request, 'saved_papers.html', context)

# 최근 본 논문
def recent_papers(request):
    query = request.GET.get('query', '')
    order = request.GET.get('order', 'desc')
    sort_by = request.GET.get('sort_by', 'viewed_at')
    items_per_page = int(request.GET.get('items_per_page', 10))  # 기본값 10

    # 로그인된 유저의 최근 본 논문을 가져오기
    if request.user.is_authenticated:
        recent_papers_qs = RecentPaper.objects.filter(user=request.user).select_related('paper')
        saved_paper_ids = SavedPaper.objects.filter(user=request.user).values_list('paper_id', flat=True)
    else:
        recent_papers_qs = RecentPaper.objects.none()  # 로그인되지 않은 경우 빈 QuerySet
        saved_paper_ids = []  # 저장된 논문 ID 리스트를 빈 리스트로 설정

    # Paper 객체만 추출
    papers = [recent_paper.paper for recent_paper in recent_papers_qs]

    ## 정렬 처리
    if sort_by == 'title':
        papers = sorted(papers, key=lambda paper: paper.title, reverse=(order == 'desc'))
    elif sort_by == 'latest':
        papers = sorted(papers, key=lambda paper: paper.date, reverse=(order == 'desc'))
    else:  # 기본적으로 viewed_at 기준으로 내림차순 정렬 (최근 본 순서대로)
        recent_papers_qs = recent_papers_qs.order_by('-viewed_at')
        papers = [recent_paper.paper for recent_paper in recent_papers_qs]

    # 페이징 처리
    paginator = Paginator(papers, items_per_page)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 페이지 그룹 계산 (10개 단위)
    current_page = page_obj.number
    current_group_start = (current_page - 1) // 10 * 10 + 1
    current_group_end = min(current_group_start + 9, paginator.num_pages)

    # 페이지 그룹 리스트 생성
    current_group_pages = list(range(current_group_start, current_group_end + 1))

    # 저자 및 키워드 추가 + 저장 여부 확인
    papers_with_authors_and_keywords = []
    for paper in page_obj:
        authors = Author.objects.filter(paperauthor__paper_id=paper.id)
        keywords = Keyword.objects.filter(paperkeyword__paper_id=paper.id)
        affiliations = Affiliation.objects.filter(paperaffiliation__paper_id=paper.id)

        # 논문 내 국가 정보 수집 및 중복 제거
        countries = PaperCountry.objects.filter(paper_id=paper.id).select_related('country')
        unique_countries = list(set([country.country.name for country in countries]))

        # 각 논문이 저장된 상태인지 확인하여 is_saved 추가
        is_saved = paper.id in saved_paper_ids

        papers_with_authors_and_keywords.append({
            'paper': paper,
            'authors': authors,
            'keywords': keywords,
            'affiliations': affiliations,
            'countries': unique_countries,  # 중복 제거된 국가 목록 추가
            'is_saved': is_saved,  # 저장 여부
        })

    context = {
        'query': query,
        'papers_with_authors_and_keywords': papers_with_authors_and_keywords,
        'page_obj': page_obj,
        'order': order,
        'sort_by': sort_by,
        'items_per_page': items_per_page,
        'current_group_pages': current_group_pages,
    }

    return render(request, 'recent_papers.html', context)

# 분석 저장소
def analysis_file(request):
    # 여기에 추천 논문 데이터를 불러오는 로직을 추가합니다.
    return render(request, 'analysis_file.html')

# 이미지 저장
@login_required
def save_image(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data['imageData']
        folder_name = data['folderName']
        image_name = data['imageName']
        
        user_folder = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username, folder_name)

        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        format, imgstr = image_data.split(';base64,') 
        ext = format.split('/')[-1] 
        image_data = ContentFile(base64.b64decode(imgstr), name=f'{image_name}.{ext}')

        file_path = os.path.join(user_folder, image_data.name)
        with open(file_path, 'wb') as f:
            f.write(image_data.read())

        return JsonResponse({'status': 'success', 'message': '이미지가 저장되었습니다.'})
    
    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'}, status=400)

# 폴더 생성
@login_required
def create_folder(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            folder_name = data.get('folder_name')
            if folder_name is None:
                return JsonResponse({'status': 'error', 'message': '폴더 이름이 제공되지 않았습니다.'})

            user_folder_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username)
            new_folder_path = os.path.join(user_folder_path, folder_name)

            os.makedirs(new_folder_path, exist_ok=True)
            return JsonResponse({'status': 'success', 'message': '폴더가 생성되었습니다.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'폴더 생성 중 오류 발생: {str(e)}'})

# 폴더명 수정
@login_required
def edit_folder(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            old_name = data.get('old_name')
            new_name = data.get('new_name')

            if old_name is None or new_name is None:
                return JsonResponse({'status': 'error', 'message': '올바른 폴더 이름이 제공되지 않았습니다.'})

            user_folder_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username)
            old_path = os.path.join(user_folder_path, old_name)
            new_path = os.path.join(user_folder_path, new_name)

            os.rename(old_path, new_path)
            return JsonResponse({'status': 'success', 'message': '폴더 이름이 수정되었습니다.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'폴더 이름 수정 중 오류 발생: {str(e)}'})

# 폴더 삭제
@login_required
def delete_folder(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            folder_name = data.get('folder_name')
            if folder_name is None:
                return JsonResponse({'status': 'error', 'message': '폴더 이름이 제공되지 않았습니다.'})

            user_folder_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username, folder_name)
            if os.path.isdir(user_folder_path):
                os.rmdir(user_folder_path)
                return JsonResponse({'status': 'success', 'message': '폴더가 삭제되었습니다.'})
            else:
                return JsonResponse({'status': 'error', 'message': '폴더가 존재하지 않습니다.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'폴더 삭제 중 오류 발생: {str(e)}'})
        
# 분석저장소 내 폴더 보여주기
@login_required
def analysis_storage(request):
    # 사용자 폴더 경로 설정
    user_folder_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username)
    folder_list = []

    # 사용자 폴더 내의 폴더 목록과 생성 날짜 가져오기
    if os.path.exists(user_folder_path):
        for folder in os.scandir(user_folder_path):
            if folder.is_dir():
                folder_info = {
                    'name': folder.name,
                    'creation_date': timezone.datetime.fromtimestamp(folder.stat().st_ctime),  # 생성 날짜
                    'count': len([f for f in os.scandir(folder.path) if f.is_file()])  # 폴더 내 파일 수
                }
                folder_list.append(folder_info)

    # 폴더 목록을 생성 날짜 순으로 정렬
    folder_list.sort(key=lambda x: x['creation_date'], reverse=True)

    return render(request, 'analysis_storage.html', {'folders': folder_list})

# 폴더 내 이미지 보여주기
def get_folder_images(request):
    folder_name = request.GET.get('folder_name')
    if not folder_name:
        return JsonResponse({'status': 'error', 'message': '폴더 이름이 제공되지 않았습니다.'})

    folder_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username, folder_name)
    
    if not os.path.isdir(folder_path):
        return JsonResponse({'status': 'error', 'message': '폴더가 존재하지 않습니다.'})

    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            images.append({
                'name': filename,
                'url': os.path.join(settings.MEDIA_URL, 'accounts', request.user.username, folder_name, filename)
            })

    return JsonResponse({'status': 'success', 'images': images})

# 폴더 내 이미지 목록 반환
def get_images(request):
    folder_name = request.GET.get('folder')
    folder_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username, folder_name)
    
    if not os.path.isdir(folder_path):
        return JsonResponse([], safe=False)
    
    images = []
    for filename in os.listdir(folder_path):
        images.append({
            'url': f'/media/accounts/{request.user.username}/{folder_name}/{filename}',
            'name': filename
        })
    return JsonResponse(images, safe=False)

# 이미지 삭제 뷰
def delete_image(request):
    if request.method == "DELETE":
        folder_name = request.GET.get('folder')  # 폴더 이름
        image_name = request.GET.get('image')    # 이미지 파일 이름
        
        # 이미지 파일의 전체 경로 구성
        image_path = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username, folder_name, image_name)

        # 파일 존재 여부 확인 후 삭제
        if os.path.isfile(image_path):
            os.remove(image_path)
            return JsonResponse({'status': 'success', 'message': '이미지가 삭제되었습니다.'})
        else:
            return JsonResponse({'status': 'error', 'message': '이미지를 찾을 수 없습니다.'}, status=404)

    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'}, status=400)

# 리포트 제출 및 저장
@csrf_exempt
@login_required
def submit_report(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        folder_name = data.get('folder')
        title = data.get('title')  # 추가된 제목 필드
        selected_images = data.get('images')
        content = data.get('content')
        
        if not title:
            return JsonResponse({'status': 'fail', 'message': '제목이 필요합니다.'}, status=400)
        
        # 저장 폴더 설정
        reports_folder = os.path.join(settings.MEDIA_ROOT, 'reports', request.user.username)
        os.makedirs(reports_folder, exist_ok=True)
        
        # 제목을 파일명으로 사용하여 저장
        safe_title = title.replace(" ", "_")  # 공백을 _로 대체하여 안전하게 파일명에 사용
        report_path = os.path.join(reports_folder, f"{safe_title}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(f"리포트 제목: {title}\n")
            report_file.write(f"리포트 내용:\n{content}\n\n")
            report_file.write("선택된 이미지 목록:\n")
            for img_url in selected_images:
                report_file.write(f"{img_url}\n")
        
        return JsonResponse({'status': 'success', 'message': '리포트 저장 완료'})
    
    return JsonResponse({'status': 'fail', 'message': '리포트 저장 실패'}, status=400)

# 리포팅 html 매핑
@login_required
def reporting(request):
    folder_name = request.GET.get('folder')
    return render(request, 'reporting.html', {'folder_name': folder_name})


from django.utils.dateformat import format

@login_required
def report_list(request):
    user_folder = os.path.join(settings.MEDIA_ROOT, 'reports', request.user.username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    reports = []
    for file in os.listdir(user_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(user_folder, file)
            folder_name = file.rsplit('.txt', 1)[0]

            # 생성 날짜를 datetime 객체로 변환
            creation_timestamp = os.path.getctime(file_path)
            creation_date = datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            reports.append({
                'folder_name': folder_name,
                'file_path': file_path,
                'creation_date': creation_date,
            })

    return render(request, 'report_list.html', {'reports': reports})


@login_required
def report_detail(request, folder_name):
    user_folder = os.path.join(settings.MEDIA_ROOT, 'reports', request.user.username)
    report_path = os.path.join(user_folder, f"{folder_name}.txt")

    if not os.path.exists(report_path):
        return render(request, 'report_detail.html', {'error': '리포트를 찾을 수 없습니다.'})

    title = ""
    content = ""
    image_urls = []

    # 파일을 읽어서 제목, 내용, 이미지 URL을 추출
    with open(report_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        is_content = False
        for line in lines:
            line = line.strip()
            if line.startswith("리포트 제목:"):
                title = line.replace("리포트 제목:", "").strip()
            elif line.startswith("리포트 내용:"):
                is_content = True
            elif line.startswith("선택된 이미지 목록:"):
                is_content = False
            elif is_content:
                content += line + "\n"
            elif line.startswith("/media/"):
                image_urls.append(line)

    return render(request, 'report_detail.html', {
        'folder_name': folder_name,
        'title': title,
        'content': content.strip(),
        'images': image_urls,
    })

    #         return JsonResponse({'status': 'error', 'message': str(e)})
    # return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'})

def analyze_chart(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            chart_id = data.get('chart_id', '')
            chart_data = data.get('chart_data', '')

            # 각 차트에 대한 GPT 프롬프트 생성
            if chart_id == 'papersChart':
                prompt = f"연도별 논문 수 분포를 분석하고, 중요한 경향이나 패턴이 있는지 설명해 주세요: {chart_data}"
            elif chart_id == 'authorsChart':
                prompt = f"저자별 논문 수 분포를 분석하고, 주요 기여자나 두드러진 패턴이 있는지 설명해 주세요: {chart_data}"
            elif chart_id == 'affiliationChart':
                prompt = f"소속 기관별 논문 수를 분석하고, 두드러진 기관이나 기여 수준의 경향이 있는지 설명해 주세요: {chart_data}"
            elif chart_id == 'countryChart':
                prompt = f"국가별 논문 수 분포를 분석하고, 주요 국가나 논문 수의 경향을 설명해 주세요: {chart_data}"
            elif chart_id == 'keywordChart':
                prompt = f"논문에서 사용된 주요 키워드를 분석하고, 주요 키워드나 어떤 주제의 논문 수가 많은지 경향을 설명해 주세요: {chart_data}"
            else:
                return JsonResponse({'error': 'Invalid chart ID'}, status=400)

            # GPT API 호출
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # 또는 gpt-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # GPT 분석 결과 추출
            analysis_result = response.choices[0].message['content'].strip()

            return JsonResponse({'analysis': analysis_result})

        except Exception as e:
            return JsonResponse({'error': f'Analysis failed: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def analysis_page(request):
    # 상위 10개 나라와 논문 수를 가져오기
    top_countries = (
        PaperCountry.objects
        .values('country__name')  # Country 이름으로 그룹화
        .annotate(total_papers=Count('paper_id', distinct=True))  # 논문 수 계산
        .order_by('-total_papers')[:10]  # 논문 수로 정렬 후 상위 10개
    )

    # 상위 10개 저자와 논문 수를 가져오기
    top_authors = (
        PaperAuthor.objects
        .values('author__name')
        .annotate(total_papers=Count('paper_id', distinct=True))
        .order_by('-total_papers')[:10]
    )

    # 상위 10개 소속과 논문 수를 가져오기
    top_affiliations = (
        PaperAffiliation.objects
        .values('affiliation__name')
        .annotate(total_papers=Count('paper_id', distinct=True))  # 중복 제거
        .order_by('-total_papers')[:10]
    )

    # 데이터를 템플릿으로 전달
    context = {
        'top_countries': top_countries,
        'top_authors': top_authors,
        'top_affiliations': top_affiliations,
    }

    return render(request, 'analysis_page.html', context)


def country_search(request):
    search_query = request.GET.get('name', '').strip()

    with connection.cursor() as cursor:
        # 'name', 'alpha_2', 'alpha_3' 중 하나와 매칭되는 국가 검색 및 논문 수 가져오기
        cursor.execute("""
            SELECT c.name, COUNT(DISTINCT pc.paper_id) AS total_papers
            FROM country c
            LEFT JOIN paper_country pc ON c.id = pc.country_id
            WHERE c.name LIKE %s OR c.alpha_2 LIKE %s OR c.alpha_3 LIKE %s
            GROUP BY c.name
            ORDER BY total_papers DESC;
        """, [f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])

        rows = cursor.fetchall()

    # 검색 결과를 JSON 형식으로 변환
    results = [{'name': row[0], 'total_papers': row[1]} for row in rows]
    return JsonResponse(results, safe=False)


def author_search(request):
    search_query = request.GET.get('name', '').strip()

    with connection.cursor() as cursor:
        # 저자 이름에 따라 매칭되는 저자와 해당 논문 수를 검색
        cursor.execute("""
            SELECT a.name, COUNT(DISTINCT pa.paper_id) AS total_papers
            FROM author a
            LEFT JOIN paper_author pa ON a.id = pa.author_id
            WHERE REPLACE(a.name, ' ', '') LIKE REPLACE(%s, ' ', '')
            GROUP BY a.name
            ORDER BY total_papers DESC;
        """, [f'%{search_query}%'])

        rows = cursor.fetchall()

    # 검색 결과를 JSON 형식으로 변환
    results = [{'name': row[0], 'total_papers': row[1]} for row in rows]
    return JsonResponse(results, safe=False)


def affiliation_search(request):
    search_query = request.GET.get('name', '').strip()
    start_year = int(request.GET.get('start_year', 2019))
    end_year = int(request.GET.get('end_year', 2024))

    with connection.cursor() as cursor:
        # 공백을 제거하여 검색하는 SQL 쿼리
        query_conditions = []
        query_params = []

        # 소속 이름 검색 조건 추가
        query_conditions.append("""
            REPLACE(a.name, ' ', '') LIKE REPLACE(%s, ' ', '')
        """)
        query_params.append(f'%{search_query}%')

        # 연도 필터링 조건 추가
        query_conditions.append("YEAR(p.date) BETWEEN %s AND %s")
        query_params.extend([start_year, end_year])

        # 쿼리 실행
        query = f"""
            SELECT a.name, COUNT(DISTINCT p.id) AS total_papers
            FROM paper p
            JOIN paper_affiliation pa ON p.id = pa.paper_id
            JOIN affiliation a ON pa.affiliation_id = a.id
            WHERE {' AND '.join(query_conditions)}
            GROUP BY a.name
            ORDER BY total_papers DESC;
        """
        cursor.execute(query, query_params)

        # 결과 가져오기
        rows = cursor.fetchall()

    # 검색 결과를 JSON 형식으로 변환
    results = [{'name': row[0], 'total_papers': row[1]} for row in rows]
    return JsonResponse(results, safe=False)

def get_folders(request):
    user_folder = os.path.join(settings.MEDIA_ROOT, 'accounts', request.user.username)
    if not os.path.exists(user_folder):
        return JsonResponse({'folders': []})
    
    folders = [f for f in os.listdir(user_folder) if os.path.isdir(os.path.join(user_folder, f))]
    return JsonResponse({'folders': folders})

