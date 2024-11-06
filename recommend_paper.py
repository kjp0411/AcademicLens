import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import mariadb

db_config = {
    'host': '127.0.0.1',
    'user': 'goorm',
    'password': '123456',
    'database': 'capstone',
    'port':3307
}

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

# 유저가 저장한 논문들 중에서 tf-idf 수치(가장 많이 사용된 단어)가 높은 단어 3개 추출
def top3_word(username):
    # 검색 엔진을 통해 논문 ID 가져오기
    paper_ids = get_user_savedpaper_ids(username)

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

def get_recommended_papers_for_user(username, start_year=2019, end_year=2024):
    # 상위 3개의 키워드를 추출
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

# 예시 실행
username = 'kaihojun'
recommended_papers = get_recommended_papers_for_user(username)

# 결과 출력
for item in recommended_papers:
    print(f"Keyword: {item['keyword']}")
    for paper in item['top_papers']:
        print(f"  - Paper ID: {paper['id']}, Score: {paper['score']}")