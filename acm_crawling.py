from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

data=pd.DataFrame(columns=['search','title', 'url', 'author','date','citations','publisher','abstract','affiliation','keywords'])

# chromedriver 자동으로 설치
chromedriver_autoinstaller.install()

# 웹 드라이버 실행
driver = webdriver.Chrome()

def main_crawling(s, link):
    search=s
    url = "https://dl.acm.org" + link
    driver.get(url)
    time.sleep(10)
    wait = WebDriverWait(driver, 10)
    button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="skip-to-main-content"]/main/article/header/div/div[3]/a')))
    driver.execute_script("arguments[0].click();", button)

    time.sleep(10)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    #제목
    try:
        title = soup.find('h1', {'property': 'name'}).text
    except:
        title = 'none'

    #저자 리스트
    try:
        author = [tag.find('span', property='givenName').text.strip() + ' ' + tag.find('span', property='familyName').text.strip()
           for tag in soup.find_all('span', property='author')]
    except:
        author = 'none'

    #url
    try:
        link = soup.find('a', property='sameAs').text.strip()
    except:
        link = 'none'

    #출판일
    try:
        published_date = soup.find('span', class_='core-date-published').text.strip()
        published_date = datetime.strptime(published_date, '%d %B %Y')
        published_date=published_date.strftime('%Y-%m-%d')
    except:
        published_date = 'none'

    #인용 수
    try:
        citations = soup.find('span', class_='citation').find('span').text.strip()
    except:
        citations = '0'

    #초록
    try:
        paragraphs = soup.find_all('div', role='paragraph')
        filtered_paragraphs = [para.text.strip() for para in paragraphs if "core-copyright" not in para.get("class", [])]
        abstract = " ".join(filtered_paragraphs)
    except:
        abstract = 'none'

    #키워드
    try:
        # 최상위 ol 태그 찾기
        ol_tag = soup.find('ol', class_='rlist organizational-chart')

        # a 태그에서 텍스트 추출
        items = ol_tag.find_all('a')

        # 키워드 리스트 생성
        keyword = [item.text for item in items]
    except:
        keyword= 'none'

    #출판사
    publisher="ACM"

    #소속 기관
    try:
        institutions = soup.find_all('span', class_='info--text auth-institution')
        institution_list = [institution.text.strip() for institution in institutions]
    except:
        institution_list='none'
    
    #저자 별 소속 기관
    if author=='none' or institution_list=='none':
        affiliation='none'
    else:
        affiliation=dict(zip(author,institution_list))

    new_data=[search,title,link,author,published_date,citations,publisher,abstract,affiliation,keyword]
    data.loc[len(data)]=new_data

search=input("검색할 단어 : ")
year=input("검색할 년도 : ")
page=int(input("페이지 수(페이지당 50개, 0으로 입력 시 무한): "))

if page == 0:
    try:
        # 무한 크롤링
        i=0
        while True:
            url='https://dl.acm.org/action/doSearch?fillQuickSearch=false&target=advanced&expand=dl&EpubDate=%5B'+year+'0101+TO+'+year+'1231%5D&AllField='+search+'&pageSize=50&startPage='+str(i)

            driver.get(url)
            time.sleep(10)

            # 크롤링 진행 중인 페이지 출력
            print(f'현재 페이지 : {i+1}')

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            links = soup.find_all('h5', class_='issue-item__title')

            if not links: #검색된 url의 links가 비었을 경우 반복 종료
                break

            for item in tqdm(links):
                try:
                    link= item.find('a')['href']
                    main_crawling(search,link)
                    time.sleep(10)
                except Exception as e:
                    continue

            # 현재 데이터프레임의 개수 확인
            print(f'현재 데이터 개수 : {len(data)}')
            if len(data) > 200:
                break;
            i+=1
    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 결측값 0으로 채움
        data=data.fillna(0)

        # link 기준 중복값 제거
        data=data.drop_duplicates(subset=['url'])

        #숫자 데이터 문자열 변환
        data['citations']=data['citations'].astype(str)

        # 최종 결과 csv파일 변환
        data.to_csv('ACM_'+year+'_'+search+'_crawling.csv', encoding='utf-8-sig', index=False, float_format='%f')
        driver.quit()



else:
    try:
        for i in tqdm(range(page)):
            # 검색할 url
            url='https://dl.acm.org/action/doSearch?fillQuickSearch=false&target=advanced&expand=dl&EpubDate=%5B'+year+'0101+TO+'+year+'1231%5D&AllField='+search+'&pageSize=50&startPage='+str(i)
            driver.get(url)
            time.sleep(10)

            # 크롤링 진행 중인 페이지 출력
            print(f'현재 페이지 : {i+1}')

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            links = soup.find_all('h5', class_='issue-item__title')

            if not links: #검색된 url의 links가 비었을 경우 반복 종료
                    break

            for item in tqdm(links):
                try:
                    link= item.find('a')['href']
                    main_crawling(search,link)
                    time.sleep(10)
                except Exception as e:
                    continue
            
            # 현재 데이터프레임의 개수 확인
            print(f'현재 데이터 개수 : {len(data)}')
    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 결측값 0으로 채움
        data=data.fillna('0')

        # link 기준 중복값 제거
        data=data.drop_duplicates(subset=['url'])

        #숫자 데이터 문자열 변환
        data['citations']=data['citations'].astype(str)

        # 최종 결과 csv파일 변환
        data.to_csv('ACM_'+year+'_'+search+'_crawling.csv', encoding='utf-8-sig', index=False, float_format='%f')
        driver.quit()