import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import mariadb

# 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',      # 호스트 주소
    'user': 'kaihojun',  # 사용자 이름
    'password': '1234',  # 비밀번호
    'database': 'capstone',  # 데이터베이스 이름
    'port': 3306              # 포트 번호
}

# 검색 엔진 함수
def get_paper_ids(user_keyword):
    # MariaDB 데이터베이스 연결
    db = mariadb.connect(**db_config)
    # 커서 생성
    cursor = db.cursor(dictionary=True)

    # 쿼리 실행 / 검색 방식
    if len(user_keyword) <= 2:  # 검색 키워드 길이가 2 이하일 때
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
        """, (f'%{user_keyword}%', f'%{user_keyword}%', f'%{user_keyword}%', f'%{user_keyword}%'))
    else:  # 검색 키워드 길이가 3 이상일 때
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

# 사용자 입력 키워드
user_keyword = "Artificial Intelligence"

# 검색 엔진을 통해 논문 ID 가져오기
paper_ids = get_paper_ids(user_keyword)

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

print(top_20_words)