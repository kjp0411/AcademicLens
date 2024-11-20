import pandas as pd

# CSV 파일 열기
file_path = 'total_result.csv'  # CSV 파일 경로를 입력하세요
df = pd.read_csv(file_path)

# URL 변경 함수 정의
def update_url(url):
    if isinstance(url, str) and 'ieeexplore-ieee-org.libproxy.catholic.ac.kr' in url:
        return url.replace('ieeexplore-ieee-org.libproxy.catholic.ac.kr', 'ieeexplore.ieee.org')
    return url

# URL 컬럼 업데이트
df['url'] = df['url'].apply(update_url)

# 수정된 데이터를 새 CSV로 저장
output_path = 'change_IEEE_URL_total_result.csv'  # 저장할 CSV 파일 경로
df.to_csv(output_path, index=False)

print(f"URL 변경이 완료되었습니다. 수정된 파일이 {output_path}에 저장되었습니다.")