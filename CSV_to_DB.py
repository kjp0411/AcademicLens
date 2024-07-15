import pandas as pd
import mariadb
import ast  # 문자열을 리스트나 딕셔너리로 변환하기 위한 모듈

# mariadb 연결 설정(이건 개인별로 다르게 설정하면 됨)
conn = mariadb.connect(
    host="127.0.0.1",
    port=3307,
    user="root",
    password="123456",
    database="capstone",
)
cursor = conn.cursor()

# CSV 파일 읽기
csv_file = r"C:\Users\wlgus\OneDrive\바탕 화면\대학\4-1\캡스톤디자인\csv파일\total_result.csv"
selected_columns = ["search", "title", "url", "author", "date", "citations", "publisher", "abstract", "affiliation", "keywords"]
data = pd.read_csv(csv_file, usecols=selected_columns, encoding='utf-8')

# "none" 값을 None으로 변경
data.loc[data['date'] == "none", 'date'] = None


# 'nan' 값을 NULL로 대체
data.fillna(value=pd.NA, inplace=True)

# MySQL에 데이터 삽입
for index, row in data.iterrows():
    # 'nan' 값을 포함하지 않는 행만 삽입
    if not row.isnull().any():
        try:
            # 데이터 파싱
            search, title, url, authors, date, citations, publisher, abstract, affiliations, keywords = row
            authors = ast.literal_eval(authors)  # 문자열을 리스트로 변환

            # 'none'이라는 값이 있다면 빈 리스트로 변환
            if keywords == 'none':
                keywords = []
            else:
                keywords = ast.literal_eval(keywords)

            # 만약 affiliations가 단일 값만 가지고 있다면 이렇게 처리할 수 있습니다.
            if isinstance(affiliations, str) and '{' not in affiliations and '}' not in affiliations:
                affiliations = {"affiliation": affiliations}
            else:
                affiliations = ast.literal_eval(affiliations)

            # 이미 존재하는 URL인지 확인
            cursor.execute("SELECT COUNT(*) FROM paper WHERE url = %s", (url,))
            if cursor.fetchone()[0] > 0:
                print(f"Skipping insertion for duplicate URL: {url}")
                continue  # 이미 존재하는 URL이면 다음 행으로 넘어감

            # 존재하지 않는 url이면 삽입 진행
            sql = "INSERT INTO paper (search, title, url, date, citations, publisher, abstract) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            values = (search, title, url, date, citations, publisher, abstract)
            cursor.execute(sql, values)
            paper_id = cursor.lastrowid  # 삽입된 article의 id를 가져옴

            for author, affiliation in affiliations.items():
                if affiliation == 'none':
                    continue

                affiliation_list=affiliation.split(',')

                # 각 항목의 양쪽 공백을 제거
                affiliation_list=[a.strip() for a in affiliation_list]

                country = None

                if len(affiliation_list) >= 3:
                    affiliation = ', '.join(affiliation_list[:-1])
                    if ' ' not in affiliation_list[-1]:
                        country = affiliation_list[-1]
                else:
                    affiliation = ', '.join(affiliation_list[:])

                
                sql = "INSERT INTO affiliation (name) VALUES (%s)"
                cursor.execute(sql, (affiliation,))
                affiliation_id = cursor.lastrowid  

                sql = "INSERT INTO author (name, affiliation) VALUES (%s, %s)"
                cursor.execute(sql, (author, affiliation))
                author_id = cursor.lastrowid  

                if country:  # country가 설정된 경우에만 country 삽입
                    sql = "INSERT INTO country (name) VALUES (%s)"
                    cursor.execute(sql, (country,))
                    country_id = cursor.lastrowid

                    sql = "INSERT INTO paper_country (paper_id, country_id) VALUES (%s, %s)"
                    cursor.execute(sql, (paper_id, country_id))

                sql = "INSERT INTO paper_author (paper_id, author_id) VALUES (%s, %s)"
                cursor.execute(sql, (paper_id, author_id))
                
                sql = "INSERT INTO paper_affiliation (paper_id, affiliation_id) VALUES (%s, %s)"
                cursor.execute(sql, (paper_id, affiliation_id))

            
            # keyword 테이블과 paper_keyword 테이블에 데이터 삽입
            for keyword in keywords:
                # 먼저 keyword 테이블에 키워드가 존재하는지 확인
                sql = "SELECT id FROM keyword WHERE keyword_name = %s"
                cursor.execute(sql, (keyword,))
                result = cursor.fetchone()
                
                if result:
                    keyword_id = result[0]
                else:
                    # 키워드가 없으면 삽입
                    sql = "INSERT INTO keyword (keyword_name) VALUES (%s)"
                    cursor.execute(sql, (keyword,))
                    keyword_id = cursor.lastrowid
                
                # paper_keyword 테이블에 paper_id와 keyword_id를 삽입
                sql = "INSERT INTO paper_keyword (paper_id, keyword_id) VALUES (%s, %s)"
                cursor.execute(sql, (paper_id, keyword_id))
        except SyntaxError:  # affiliations 파싱에서 SyntaxError가 발생하면
            continue  # 다음 행으로 넘어갑니다.

# 변경사항 커밋 및 연결 닫기
conn.commit()
conn.close()