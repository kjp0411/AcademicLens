import pandas as pd
import re

# CSV 파일 읽기
csv_file = "total_result.csv"  # 파일 경로를 입력하세요
data = pd.read_csv(csv_file, encoding='utf-8')

# 1. 각 컬럼에서 비어 있거나 NaN인 값을 가진 행 삭제 (제일 먼저 실행)
columns_to_check = ['search', 'title', 'url', 'author', 'date', 'citations', 'publisher', 'abstract', 'affiliation', 'keywords']
data = data.dropna(subset=columns_to_check)

print("\n비어 있거나 NaN 값을 가진 행이 삭제되었습니다.")

# 2. citations 컬럼에서 숫자가 아닌 값 확인 및 삭제
non_numeric_citations = data[pd.to_numeric(data['citations'], errors='coerce').isnull()]

if not non_numeric_citations.empty:
    print("\ncitations 컬럼에 숫자가 아닌 값이 포함된 행 번호와 값:")
    for index, value in non_numeric_citations['citations'].items():
        print(f"행 번호: {index}, 값: {value}")

    # 숫자가 아닌 값을 가진 행 삭제
    data = data.drop(non_numeric_citations.index)
    print("\n숫자가 아닌 값을 가진 행이 삭제되었습니다.")

# 3. url 컬럼에서 'https'로 시작하지 않는 행 확인 및 삭제
non_https_urls = data[~data['url'].str.startswith('https', na=False)]

if not non_https_urls.empty:
    print("\nurl 컬럼에서 'https'로 시작하지 않는 값이 포함된 행 번호와 값:")
    for index, value in non_https_urls['url'].items():
        print(f"행 번호: {index}, 값: {value}")

    # 'https'로 시작하지 않는 값을 가진 행 삭제
    data = data.drop(non_https_urls.index)
    print("\n'url' 컬럼에서 'https'로 시작하지 않는 행이 삭제되었습니다.")

# 4. author 컬럼에서 '['로 시작하지 않는 행 확인 및 삭제
non_bracket_authors = data[~data['author'].str.startswith('[', na=False)]

if not non_bracket_authors.empty:
    print("\nauthor 컬럼에서 '['로 시작하지 않는 값이 포함된 행 번호와 값:")
    for index, value in non_bracket_authors['author'].items():
        print(f"행 번호: {index}, 값: {value}")

    # '['로 시작하지 않는 값을 가진 행 삭제
    data = data.drop(non_bracket_authors.index)
    print("\n'author' 컬럼에서 '['로 시작하지 않는 행이 삭제되었습니다.")

# 5. date 컬럼에서 형식이 맞지 않는 값 확인 및 삭제
date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # YYYY-MM-DD 형식
invalid_dates = data[~data['date'].astype(str).str.match(date_pattern)]

if not invalid_dates.empty:
    print("\ndate 컬럼에서 형식이 맞지 않는 값이 포함된 행 번호와 값:")
    for index, value in invalid_dates['date'].items():
        print(f"행 번호: {index}, 값: {value}")

    # 형식이 맞지 않는 값을 가진 행 삭제
    data = data.drop(invalid_dates.index)
    print("\n'date' 컬럼에서 형식이 맞지 않는 행이 삭제되었습니다.")

# 6. publisher 컬럼에서 'ACM', 'Springer', 'IEEE'가 아닌 값 확인 및 삭제
invalid_publishers = data[~data['publisher'].isin(['ACM', 'Springer', 'IEEE'])]

if not invalid_publishers.empty:
    print("\npublisher 컬럼에서 'ACM', 'Springer', 'IEEE' 외의 값이 포함된 행 번호와 값:")
    for index, value in invalid_publishers['publisher'].items():
        print(f"행 번호: {index}, 값: {value}")

    # 'ACM', 'Springer', 'IEEE'가 아닌 값을 가진 행 삭제
    data = data.drop(invalid_publishers.index)
    print("\n'publisher' 컬럼에서 'ACM', 'Springer', 'IEEE' 외의 행이 삭제되었습니다.")

# 7. affiliation 컬럼에서 '{'로 시작하지 않는 값 확인 및 삭제
non_bracket_affiliations = data[~data['affiliation'].str.startswith('{', na=False)]

if not non_bracket_affiliations.empty:
    print("\naffiliation 컬럼에서 '{'로 시작하지 않는 값이 포함된 행 번호와 값:")
    for index, value in non_bracket_affiliations['affiliation'].items():
        print(f"행 번호: {index}, 값: {value}")

    # '{'로 시작하지 않는 값을 가진 행 삭제
    data = data.drop(non_bracket_affiliations.index)
    print("\n'affiliation' 컬럼에서 '{'로 시작하지 않는 행이 삭제되었습니다.")

# 8. keywords 컬럼에서 '['로 시작하지 않는 값 확인 및 수정
non_bracket_keywords = data[~data['keywords'].str.startswith('[', na=False)]

if not non_bracket_keywords.empty:
    print("\nkeywords 컬럼에서 '['로 시작하지 않는 값이 포함된 행 번호와 값:")
    for index, value in non_bracket_keywords['keywords'].items():
        print(f"행 번호: {index}, 값: {value}")

    # '['로 시작하지 않는 값을 빈 리스트로 대체
    data.loc[non_bracket_keywords.index, 'keywords'] = '[]'
    print("\n'['로 시작하지 않는 keywords 값이 빈 리스트('[]')로 대체되었습니다.")

# 9. url 컬럼에서 중복된 값 확인 및 제거
data_before = len(data)
data = data.drop_duplicates(subset=['url'], keep='first')
data_after = len(data)

print(f"\n중복된 URL 값이 제거되었습니다. {data_before - data_after}개의 중복된 행이 삭제되었습니다.")

# 10. 결과를 저장할 파일 경로
output_file = "preprocessing_total_result.csv"
data.to_csv(output_file, index=False)

print(f"\n수정된 데이터가 {output_file}에 저장되었습니다.")
