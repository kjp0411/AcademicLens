import mariadb
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# MariaDB 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'capstone',
    'port': 3307
}

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

def make_affiliation_WordCloud(affiliation):
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
    print(top_keywords)

    # 워드 클라우드 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_keywords))

    # 워드 클라우드 출력
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top 10 Keywords for Affiliation: {affiliation}")
    plt.show()

# 예시 실행
make_affiliation_WordCloud('Technion, Haifa')