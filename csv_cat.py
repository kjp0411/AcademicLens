import pandas as pd
import glob

# 합칠 CSV 파일들이 위치한 디렉토리를 지정합니다.
csv_file_path = 'C:/python/goorm/one/*.csv'  
files = glob.glob(csv_file_path)

# 모든 CSV 파일을 읽어 데이터프레임 리스트를 생성합니다.
df_list = [pd.read_csv(file, encoding='utf-8-sig') for file in files]

# 모든 데이터프레임을 하나로 합칩니다.
merged_df = pd.concat(df_list)

# 결과를 새 CSV 파일로 저장합니다.
merged_df.to_csv('C:/python/goorm/one/total_result.csv', index=False)