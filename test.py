# 국가 그래프 출력
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

def extract_column_by_index(filename, column_index):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(filename)
        
        # 지정된 열의 값만 추출하여 출력
        pd.set_option('display.max_colwidth', None)  # 열의 최대 너비를 설정하지 않음
        column_data = df.iloc[:, column_index]
        
        # 각 국가 이름과 등장 횟수를 저장할 딕셔너리 생성
        country_counts = Counter()
        
        for text in column_data:
            country = extract_country(text)
            if country:
                country_counts[country] += 1
                
        # 가장 많이 등장한 상위 5개의 국가 선택
        top_countries = dict(country_counts.most_common(10))
        
        # 내림차순으로 정렬
        top_countries = dict(sorted(top_countries.items(), key=lambda item: item[1], reverse=False))
        
        # 한글 폰트 설정
        font_path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 경로
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        
        # 막대 그래프로 시각화
        plt.figure(figsize=(10, 6))
        plt.barh(list(top_countries.keys()), list(top_countries.values()), color='skyblue')
        plt.xlabel('등장 횟수')
        plt.ylabel('국가')
        plt.title('상위 10개 국가별 등장 횟수')
        plt.tight_layout()  # 그래프를 깔끔하게 배치
        plt.show()
        
    except FileNotFoundError:
        print("해당 파일을 찾을 수 없습니다.")
    except IndexError:
        print("지정된 열이 존재하지 않습니다.")
    except Exception as e:
        print("오류 발생:", e)

def extract_country(text):
    # 국가 이름 추출을 위한 정규식 패턴
    pattern = r',\s*([^,]+)(?=\s*$)'  # 쉼표(,) 이후의 국가 이름을 찾기 위한 패턴
    
    # 정규식을 사용하여 국가 이름 추출
    matches = re.findall(pattern, text)
    
    # 만약에 결과가 있다면, 결과의 문자열에서 작은 따옴표(')와 중괄호({})를 제거하고 반환
    if matches:
        country = matches[0].strip(" '}")  # 작은 따옴표(')와 중괄호({})를 제거
        return country

# CSV 파일 경로 지정
filename = 'dummy.csv'

# 추출할 열의 인덱스 지정
column_index = 8  # 0부터 시작하므로 8번째 열의 인덱스는 7입니다.

# 함수 호출
extract_column_by_index(filename, column_index)