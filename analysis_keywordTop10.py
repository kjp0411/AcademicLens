import mariadb
from collections import Counter
import matplotlib.pyplot as plt

# MariaDB 데이터베이스 연결 정보
db_config = {
    'host': 'localhost',
    'user': 'kaihojun',
    'password': '1234',
    'database': 'capstone'
}

user_keyword = input("검색하고싶은 키워드 입력 >>> ")

# MariaDB 데이터베이스 연결
db = mariadb.connect(**db_config)

# 커서 생성
cursor = db.cursor(dictionary=True)

# 쿼리 실행 / 검색 방식
if len(user_keyword) <= 2:
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
else:
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

# 키워드 카운트 초기화
keyword_counts = Counter()

# 각 논문의 키워드 카운트
for paper_id in paper_ids:
    cursor.execute("""
    SELECT keyword_name
    FROM paper_keyword pk
    JOIN keyword k ON pk.keyword_id = k.id
    WHERE pk.paper_id = %s;
    """, (paper_id,))
    keywords = [row['keyword_name'] for row in cursor.fetchall()]
    keyword_counts.update(keywords)

# 빈도수가 높은 top 10 키워드 출력하기
top_keywords = keyword_counts.most_common(10)
print("Top 10 Keywords:")
for keyword, count in top_keywords:
    print(f"- {keyword}: {count}")

# 시각화
keywords, counts = zip(*top_keywords)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(keywords, counts)

ax.set_title('Top 10 Keywords')
ax.set_xlabel('Frequency')
ax.set_ylabel('Keyword')

plt.show()

# 데이터베이스 연결 종료
db.close()
