import pandas as pd
import mariadb
import ast  # 문자열을 리스트나 딕셔너리로 변환하기 위한 모듈
import time
from tqdm import tqdm  # 진행도를 표시하는 라이브러리

start_time = time.time()

# mariadb 연결 설정(이건 개인별로 다르게 설정하면 됨)
conn = mariadb.connect(
    host="127.0.0.1",
    port=3306,
    user="goorm",
    password="1234",
    database="capstone",
)
cursor = conn.cursor()

# CSV 파일 읽기
csv_file = r"cleaned_file.csv"  # 전처리 완료된 파일 경로
selected_columns = ["search", "title", "url", "author", "date", "citations", "publisher", "abstract", "affiliation", "keywords"]
data = pd.read_csv(csv_file, usecols=selected_columns, encoding='utf-8')

# 첫 5000개 행만 선택
data = data.head(5000)

# MySQL에 데이터 삽입 (진행도 표시)
for index, row in tqdm(data.iterrows(), total=len(data), desc="Inserting data into MariaDB"):
    try:
        # 데이터 파싱
        search, title, url, authors, date, citations, publisher, abstract, affiliations, keywords = row
        authors = ast.literal_eval(authors)  # 문자열을 리스트로 변환
        keywords = ast.literal_eval(keywords)  # 키워드 문자열을 리스트로 변환
        affiliations = ast.literal_eval(affiliations)  # affiliation 문자열을 딕셔너리로 변환

        # paper 데이터 삽입
        sql = "INSERT INTO paper (search, title, url, date, citations, publisher, abstract) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (search, title, url, date, citations, publisher, abstract)
        cursor.execute(sql, values)
        paper_id = cursor.lastrowid  # 삽입된 paper의 id를 가져옴

        # affiliations 데이터 처리
        for author, affiliation in affiliations.items():
            if affiliation == 'none':
                continue

            affiliation_list = affiliation.split(',')

            # 각 항목의 양쪽 공백을 제거하고 소문자로 변환
            affiliation_list = [a.strip().lower() for a in affiliation_list]

            country = None

            # affiliation_list의 각 항목을 맨 뒤에서부터 country 테이블의 name, alpha_2, alpha_3와 비교
            for item in reversed(affiliation_list):
                if 'korea' in item:
                    cursor.execute("SELECT id FROM country WHERE name = %s", ('South Korea',))
                else:
                    cursor.execute("SELECT id FROM country WHERE name = %s OR alpha_2 = %s OR alpha_3 = %s", (item, item, item))
                country_result = cursor.fetchone()
                if country_result:
                    country = item
                    country_id = country_result[0]
                    break

            affiliation = ', '.join(affiliation_list)

            # 소속 삽입 코드
            check_sql = "SELECT id FROM affiliation WHERE name = %s"
            cursor.execute(check_sql, (affiliation,))
            result = cursor.fetchone()

            if result:
                affiliation_id = result[0]
            else:
                insert_sql = "INSERT INTO affiliation (name) VALUES (%s)"
                cursor.execute(insert_sql, (affiliation,))
                affiliation_id = cursor.lastrowid

            # author 데이터 삽입
            check_sql = "SELECT id FROM author WHERE name = %s AND affiliation = %s"
            cursor.execute(check_sql, (author, affiliation))
            result = cursor.fetchone()

            if result:
                author_id = result[0]
            else:
                sql = "INSERT INTO author (name, affiliation) VALUES (%s, %s)"
                cursor.execute(sql, (author, affiliation))
                author_id = cursor.lastrowid

            # paper_author 및 paper_affiliation 관계 테이블 삽입
            sql = "INSERT INTO paper_author (paper_id, author_id) VALUES (%s, %s)"
            cursor.execute(sql, (paper_id, author_id))

            sql = "INSERT INTO paper_affiliation (paper_id, affiliation_id) VALUES (%s, %s)"
            cursor.execute(sql, (paper_id, affiliation_id))

            if country:  # country가 설정된 경우에만 country 연결
                sql = "INSERT INTO paper_country (paper_id, country_id) VALUES (%s, %s)"
                cursor.execute(sql, (paper_id, country_id))

        # keyword 데이터 처리
        for keyword in keywords:
            check_sql = "SELECT id FROM keyword WHERE keyword_name = %s"
            cursor.execute(check_sql, (keyword,))
            result = cursor.fetchone()

            if result:
                keyword_id = result[0]
            else:
                sql = "INSERT INTO keyword (keyword_name) VALUES (%s)"
                cursor.execute(sql, (keyword,))
                keyword_id = cursor.lastrowid

            sql = "INSERT INTO paper_keyword (paper_id, keyword_id) VALUES (%s, %s)"
            cursor.execute(sql, (paper_id, keyword_id))

    except Exception as e:  # 오류 발생
        print(f"Error on row {index}: {e}")
        continue  # 다음 행으로 넘어갑니다

# 변경사항 커밋 및 연결 닫기
conn.commit()
conn.close()

end_time = time.time()
execution_time = end_time - start_time
print(f"Code execution time: {execution_time:.2f} seconds")
