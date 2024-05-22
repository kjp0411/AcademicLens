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
cursor.execute("""
SELECT p.id, GROUP_CONCAT(k.keyword_name) AS keywords
FROM paper p
JOIN paper_keyword pk ON p.id = pk.paper_id
JOIN keyword k ON pk.keyword_id = k.id
WHERE MATCH(p.search, p.title, p.abstract) AGAINST(%s)
   OR MATCH(k.keyword_name) AGAINST(%s)
GROUP BY p.id;
""", (user_keyword, user_keyword))

# 쿼리 결과 가져오기
results = cursor.fetchall()
print("가져온 총 논문 수 >>> ", len(results))

# 키워드 데이터를 하나의 리스트로 만들기
all_keywords = []
for row in results:
    all_keywords.extend(row['keywords'].split(','))

# 키워드 빈도수 계산하기
keyword_counts = Counter(all_keywords)

# 빈도수가 높은 top 10 키워드 출력하기
top_keywords = keyword_counts.most_common(10)
print("Top 10 Keywords:")
for keyword, count in top_keywords:
    print(f"- {keyword.strip()}: {count}")

keywords = [keyword.strip() for keyword, count in top_keywords]
counts = [count for keyword, count in top_keywords]

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(keywords, counts)

ax.set_title('Top 10 Keywords')
ax.set_xlabel('Frequency')
ax.set_ylabel('Keyword')

ax.set_xticks(range(0, max(counts)+1, 500))
plt.show()

# 데이터베이스 연결 종료
db.close()