import mariadb
from django.shortcuts import render
from .models import Paper, Affiliation
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


db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'capstone',
    'port': 3307
}


# 메인 화면 논문 수, 소속 수
def home(request):
    papers = Paper.objects.all()
    paper_count = len(papers)
    affiliation_count = Affiliation.objects.values('name').distinct().count()
    return render(request, 'home.html', {
        'paper_count': paper_count,
        'affiliation_count': affiliation_count
    })

# 검색 시 논문 출력 및 필터링
def search(request):
    query = request.GET.get('query', '')
    year = request.GET.get('year', '')
    search_query = query
    news_type = request.GET.get('news_type', 'international')  # 기본값을 'international'로 설정

    related_terms = []
    if query:
        related_terms = top_5_related_words(query)
    
    paper_ids = get_paper_ids(query)
    
    if year:
        papers = Paper.objects.filter(id__in=paper_ids, date__year=year)
    else:
        papers = Paper.objects.filter(id__in=paper_ids)

    ordered_papers = sorted(papers, key=lambda paper: paper_ids.index(paper.id))

    paginator = Paginator(ordered_papers, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    years = range(2020, datetime.now().year + 1)
    paper_counts_by_year = {str(y): Paper.objects.filter(id__in=paper_ids, date__year=y).count() for y in years}

    # 뉴스 검색 부분
    api_key = '2f963493ee124210ac91a3b54ebb3c5c'
    articles = []
    if search_query:
        if news_type == 'domestic':
            url = f'https://newsapi.org/v2/everything?q={search_query}&language=ko&apiKey={api_key}'
        else:
            url = f'https://newsapi.org/v2/everything?q={search_query}&language=en&apiKey={api_key}'

        response = requests.get(url)
        news_data = response.json()
        articles = news_data.get('articles', [])
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'articles': articles})
    
    context = {
        'query': query,
        'papers': page_obj,
        'paper_counts_by_year': paper_counts_by_year,
        'related_terms': related_terms,
        'articles': articles,
        'news_type': news_type,
        'search_query': search_query,
    }
    return render(request, 'search.html', context)


# 검색 엔진
def get_paper_ids(user_keyword):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 쿼리 실행 / 검색 방식
    if len(user_keyword) <= 2:      # 검색 키워드 길이가 2 이하일 때
        cursor.execute("""
        SELECT id
        FROM paper
        WHERE search LIKE %s
           OR title LIKE %s
           OR abstract LIKE %s
        UNION
        SELECT p.id
        FROM paper p
        JOIN paper_keyword pk ON p.id = pk.paper_id
        JOIN keyword k ON pk.keyword_id = k.id
        WHERE k.keyword_name LIKE %s;
        """, (f'% {user_keyword} %', f'% {user_keyword} %', f'% {user_keyword} %', f' %{user_keyword} %'))
    else:       # 검색 키워드 길이가 3 이상일 때
        cursor.execute("""
        SELECT id
        FROM paper
        WHERE MATCH(search, title, abstract) AGAINST(%s)
        UNION
        SELECT p.id
        FROM paper p
        JOIN paper_keyword pk ON p.id = pk.paper_id
        JOIN keyword k ON pk.keyword_id = k.id
        WHERE MATCH(k.keyword_name) AGAINST(%s);
        """, (user_keyword, user_keyword))

    # 쿼리 결과 가져오기
    paper_ids = [row['id'] for row in cursor.fetchall()]

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return paper_ids

# 주어진 논문 ID 목록에 대한 abstract를 가져오는 함수
def get_abstracts(paper_ids):
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

def top_5_related_words(query):
    # 검색 엔진을 통해 논문 ID 가져오기
    paper_ids = get_paper_ids(query)

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

    return top_5_related_words

# 분석 페이지 그래프
def analyze(request):
    if 'query' in request.GET:
        user_keyword = request.GET['query']
        paper_ids = get_paper_ids(user_keyword)

        # MariaDB 데이터베이스 연결
        db = mariadb.connect(**db_config)
        # 커서 생성
        cursor = db.cursor(dictionary=True)

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

                # 데이터를 UTF-8로 인코딩하여 가져오기
                rows = cursor.fetchall()
                affiliation_counts = [(row[0].encode('latin1').decode('utf8'), row[1]) for row in rows]

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
            'keyword': user_keyword
        }

        return render(request, 'total_graph.html', context)
    

# Django orm 사용
# from collections import Counter
# from django.db.models import Count
# from django.shortcuts import render
# from .models import Paper, Author, Affiliation, Country, Keyword

# def analyze(request):
#     if 'query' in request.GET:
#         user_keyword = request.GET['query']
#         paper_ids = get_paper_ids(user_keyword)

#         papers_count = []
#         if paper_ids:
#             papers_count = (Paper.objects
#                             .filter(id__in=paper_ids)
#                             .values('date__year')
#                             .annotate(count=Count('id'))
#                             .order_by('date__year'))

#         author_counts = []
#         if paper_ids:
#             author_counts = (Author.objects
#                              .filter(paperauthor__paper_id__in=paper_ids)
#                              .values('name')
#                              .annotate(publications_count=Count('paperauthor__paper'))
#                              .order_by('-publications_count')[:10])

#         affiliation_counts = []
#         if paper_ids:
#             affiliation_counts = (Affiliation.objects
#                                   .filter(paperaffiliation__paper_id__in=paper_ids)
#                                   .values('name')
#                                   .annotate(publications_count=Count('paperaffiliation__paper'))
#                                   .order_by('-publications_count')[:10])

#         country_counts = []
#         if paper_ids:
#             country_counts = (Country.objects
#                               .filter(papercountry__paper_id__in=paper_ids)
#                               .values('name')
#                               .annotate(publications_count=Count('papercountry__paper'))
#                               .order_by('-publications_count')[:10])

#         keyword_counts = Counter()
#         if paper_ids:
#             keywords = (Keyword.objects
#                         .filter(paperkeyword__paper_id__in=paper_ids)
#                         .values_list('keyword_name', flat=True))
#             keyword_counts.update(keywords)

#         top_keywords = keyword_counts.most_common(10)

#         context = {
#             'papers_count': list(papers_count),
#             'author_data': [(item['name'], item['publications_count']) for item in author_counts],
#             'affiliation_data': [(item['name'], item['publications_count']) for item in affiliation_counts],
#             'country_data': [(item['name'], item['publications_count']) for item in country_counts],
#             'top_keywords': top_keywords,
#             'keyword': user_keyword
#         }

#         return render(request, 'total_graph.html', context)

# 저자 네트워크 시각화
def author_network(request):
    try:
        original_author_name = 'A. Aguado'  # 중심 노드의 이름을 지정

        with connection.cursor() as cursor:
            query = """
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
            GROUP BY 
                a1.name, a2.name, apc1.total_papers, apc2.total_papers
            ORDER BY 
                num_papers DESC;
            """
            
            cursor.execute(query, [original_author_name, original_author_name])
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

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def author_html(request):
    return render(request, 'author_network.html')


# 소속 네트워크 시각화
def affiliation_network(request):
    try:
        original_affiliation_name = 'University of Florida, Gainesville, FL'  # 중심 노드의 이름을 지정

        with connection.cursor() as cursor:
            query = """
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
                a1.name = %s AND a2.name != %s
            GROUP BY 
                a1.name, a2.name, apc1.total_papers, apc2.total_papers
            ORDER BY 
                num_papers DESC;
            """
            
            cursor.execute(query, [original_affiliation_name, original_affiliation_name])
            rows = cursor.fetchall()

        nodes = []
        links = []
        node_set = set()

        for row in rows:
            original_affiliation = row[0]
            co_affiliation = row[1]
            num_papers = row[2]
            original_affiliation_total_papers = row[3]
            co_affiliation_total_papers = row[4]

            if original_affiliation not in node_set:
                nodes.append({"id": original_affiliation, "total_papers": original_affiliation_total_papers})
                node_set.add(original_affiliation)
            if co_affiliation not in node_set:
                nodes.append({"id": co_affiliation, "total_papers": co_affiliation_total_papers})
                node_set.add(co_affiliation)
            
            links.append({"source": original_affiliation, "target": co_affiliation, "value": num_papers})

        network_data = {"nodes": nodes, "links": links, "center_node": original_affiliation_name}
        return JsonResponse(network_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def affiliation_html(request):
    return render(request, 'affiliation_network.html')

# 국가 네트워크 시각화
def country_network(request):
    try:
        original_country_name = 'USA'  # 중심 국가의 이름을 지정

        with connection.cursor() as cursor:
            query = """
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
            GROUP BY 
                c1.name, c2.name, cpc1.total_papers, cpc2.total_papers
            ORDER BY 
                num_papers DESC
            LIMIT 10;
            """
            
            cursor.execute(query, [original_country_name, original_country_name])
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

def country_html(request):
    return render(request, 'country_network.html')

# 나라 워드클라우드 시각화
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

def country_wordcloud(request):
    country='USA'

    # 키워드 카운트 초기화
    keyword_counts = Counter()

    # 여기서 get_paper_ids_country 함수는 country에 따라 paper_ids를 가져오는 사용자 정의 함수입니다.
    paper_ids = get_paper_ids_country(country)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
        # MariaDB 연결
        db = mariadb.connect(**db_config)
        cursor = db.cursor()

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

def country_wordcloud_html(request):
    return render(request, 'country_wordcloud.html')


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

def author_wordcloud(request):
    author = 'Florian Kerschbaum'
    # 키워드 카운트 초기화
    keyword_counts = Counter()

    paper_ids = get_paper_ids_author(author)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
        # MariaDB 연결
        db = mariadb.connect(**db_config)
        cursor = db.cursor()

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

    # 빈도수가 높은 top 10 키워드 출력하기
    top_keywords = keyword_counts.most_common(20)
    top_keywords_list = [[keyword, count] for keyword, count in top_keywords]

    return JsonResponse(top_keywords_list, safe=False)

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

def affiliation_wordcloud(request):
    affiliation = 'Technion, Haifa'
    # 키워드 카운트 초기화
    keyword_counts = Counter()

    paper_ids = get_paper_ids_affiliation(affiliation)

    if paper_ids:
        placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
        # MariaDB 연결
        db = mariadb.connect(**db_config)
        cursor = db.cursor()

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

    # 빈도수가 높은 top 10 키워드 출력하기
    top_keywords = keyword_counts.most_common(20)
    top_keywords_list = [[keyword, count] for keyword, count in top_keywords]

    return JsonResponse(top_keywords_list, safe=False)

def affiliation_wordcloud_html(request):
    return render(request, 'affiliation_wordcloud.html')