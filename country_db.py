import pandas as pd
import mariadb

# mariadb 연결 설정
conn = mariadb.connect(
    host="127.0.0.1",
    port=3306,
    user="goorm",
    password="1234",
    database="capstone",
)

cursor = conn.cursor()

# CSV 파일 읽기
csv_file = r"country.csv"
data = pd.read_csv(csv_file, encoding='utf-8')

# NaN 값을 빈 문자열로 대체하고 문자열로 변환
data['alpha_2'] = data['alpha_2'].fillna('').astype(str).str[:2]
data['alpha_3'] = data['alpha_3'].fillna('').astype(str).str[:3]

# 정제된 데이터 확인
print(data.head())
print(data['alpha_2'].apply(len).max())  # alpha_2 열의 최대 길이 출력

# 테이블에 데이터 삽입
for index, row in data.iterrows():
    name = row['name']
    alpha_2 = row['alpha_2']
    alpha_3 = row['alpha_3']

    try:
        sql = "INSERT INTO country (name, alpha_2, alpha_3) VALUES (%s, %s, %s)"
        cursor.execute(sql, (name, alpha_2, alpha_3))
    except mariadb.Error as e:
        print(f"데이터 삽입 중 오류가 발생했습니다 (행 {index}): {e}")

# 변경사항 커밋 및 연결 닫기
conn.commit()
conn.close()
