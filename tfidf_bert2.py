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
        """, (f'% {user_keyword} %', f'% {user_keyword} %', f'% {user_keyword} %', f'% {user_keyword} %'))
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
    cursor.execute(f"SELECT id, abstract FROM paper WHERE id IN ({format_strings})", tuple(paper_ids))

    # 결과를 데이터프레임으로 변환
    data = pd.DataFrame(cursor.fetchall(), columns=['id', 'abstract'])

    # 데이터베이스 연결 종료
    cursor.close()
    db.close()

    return data

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

def top_5_keywords(user_keyword):
    # BERT 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 검색 엔진을 통해 논문 ID 가져오기
    paper_ids = get_paper_ids(user_keyword)

    # 논문 ID를 사용하여 abstract 가져오기
    abstract_data = get_abstracts(paper_ids)

    # 가져온 논문의 수 확인
    num_papers = len(abstract_data)
    print(f"가져온 논문의 수: {num_papers}")

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
    top_20_words = tfidf_sums.sort_values(ascending=False).head(5).index.tolist()
    print(f"TF-IDF 상위 20개 단어: {top_20_words}")

    # 상위 단어가 포함된 논문 ID 저장
    containing_paper_ids = []
    for idx, row in abstract_data.iterrows():
        if top_20_words[0] in row['abstract']:
            containing_paper_ids.append(row['id'])
            print(f"논문 ID {row['id']}에는 상위 단어 '{top_20_words[0]}'가 포함되어 있습니다.")

    print(f"상위 단어 '{top_20_words[0]}'가 포함된 논문 ID 목록: {containing_paper_ids}")

    # 사용자 입력 키워드와 상위 단어의 임베딩 계산
    user_keyword_embedding = get_embedding(user_keyword, tokenizer, model)
    top_word_embedding = get_embedding(top_20_words[0], tokenizer, model)

    # 논문들의 abstract 임베딩 계산 및 유사도 측정
    similarities = []
    for paper_id in containing_paper_ids:
        abstract = abstract_data.loc[abstract_data['id'] == paper_id, 'abstract'].values[0]
        abstract_embedding = get_embedding(abstract, tokenizer, model)
        similarity = cosine_similarity([user_keyword_embedding], [abstract_embedding])[0][0]
        similarities.append((paper_id, similarity))

    # 유사도 기준으로 정렬
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    print(f"유사도가 높은 논문 ID와 유사도: {similarities}")

    return top_20_words, abstracts, similarities

# 사용자 입력 키워드
user_keyword = "anti"

# 상위 5개의 연관 검색어 선택
related_words, abstracts, similarities = top_5_keywords(user_keyword)

# 결과 출력
print("연관 검색어로 사용할 상위 20개 단어:")
for word in related_words:
    print(word)