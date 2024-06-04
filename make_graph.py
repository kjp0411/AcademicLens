# main/views.py에 analyze 함수 고장 시 활용
# # 나라별 논문 수 그래프
# def country_publications_graph(request):
#     if request.method == 'GET' and 'query' in request.GET:
#         user_keyword = request.GET['query']
#         paper_ids = get_paper_ids(user_keyword)

#         # 국가별 논문 수 계산
#         country_counts = []
#         if paper_ids:
#             placeholders = ', '.join(['%s'] * len(paper_ids))
#             with connection.cursor() as cursor:
#                 cursor.execute(f"""
#                 SELECT c.name AS country_name, COUNT(pc.paper_id) AS publications_count
#                 FROM country c
#                 JOIN paper_country pc ON c.id = pc.country_id
#                 JOIN paper p ON pc.paper_id = p.id
#                 WHERE p.id IN ({placeholders})
#                 GROUP BY c.name
#                 ORDER BY publications_count DESC
#                 LIMIT 10;
#                 """, paper_ids)
                
#                 country_counts = cursor.fetchall()

#         # 국가 이름과 논문 수를 결합한 리스트 생성
#         country_data = list(zip([item[0] for item in country_counts], [item[1] for item in country_counts]))

#         # 그래프에 사용할 데이터 준비
#         context = {
#             'country_data': country_data,  # 국가 이름과 논문 수를 결합한 리스트
#             'keyword': user_keyword  # 검색된 키워드
#         }

#         return render(request, 'country_graph.html', context)
#     else:
#         return render(request, 'error.html', {'message': 'Invalid request'})


# # 저자별 논문 수 그래프
# def author_publications_graph(request):
#     if 'query' in request.GET:
#         user_keyword = request.GET['query']
#         paper_ids = get_paper_ids(user_keyword)

#         # 저자별 논문 수를 계산
#         author_counts = []
#         if paper_ids:
#             placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
#             with connection.cursor() as cursor:
#                 cursor.execute(f"""
#                 SELECT a.name AS author_name, COUNT(pa.paper_id) AS publications_count
#                 FROM author a
#                 JOIN paper_author pa ON a.id = pa.author_id
#                 JOIN paper p ON pa.paper_id = p.id
#                 WHERE p.id IN ({placeholders})
#                 GROUP BY a.name
#                 ORDER BY publications_count DESC
#                 LIMIT 10;
#                 """, paper_ids)

#                 author_counts = cursor.fetchall()

#         # 저자 이름과 논문 수를 결합한 리스트 생성
#         author_data = list(zip([item[0] for item in author_counts], [item[1] for item in author_counts]))

#         # 그래프에 사용할 데이터 준비
#         context = {
#             'author_data': author_data,  # 저자 이름과 논문 수를 결합한 리스트
#             'keyword': user_keyword  # 검색된 키워드
#         }

#         return render(request, 'author_graph.html', context)
#     else:
#         return render(request, 'error.html', {'message': 'Invalid request'})

# # 연도별 논문 증감 그래프
# def year_publications_graph(request):
#     if 'query' in request.GET:
#         user_keyword = request.GET['query']
#         paper_ids = get_paper_ids(user_keyword)

#         # MariaDB 데이터베이스 연결
#         db = mariadb.connect(**db_config)
#         # 커서 생성
#         cursor = db.cursor(dictionary=True)

#         # 연도별로 그룹화하고 각 연도의 논문 수 구하기
#         if paper_ids:
#             placeholders = ', '.join(['%s'] * len(paper_ids))
#             cursor.execute(f"""
#             SELECT YEAR(p.date) AS year, COUNT(*) AS count
#             FROM paper p
#             WHERE p.id IN ({placeholders})
#             GROUP BY year
#             ORDER BY year;
#             """, paper_ids)
            
#             papers_count = cursor.fetchall()
#         else:
#             papers_count = []

#         # 데이터베이스 연결 종료
#         cursor.close()
#         db.close()

#         # 그래프에 사용할 데이터 준비
#         context = {'papers_count': papers_count}

#         return render(request, 'year_graph.html', context)
#     else:
#         return render(request, 'error.html', {'message': 'Invalid request'})


# # 소속별 논문 수 그래프
# def affiliation_publication_graph(request):
#     if 'query' in request.GET:
#         user_keyword = request.GET['query']
#         paper_ids = get_paper_ids(user_keyword)

#         # 소속별 논문 수를 계산
#         affiliation_counts = []
#         if paper_ids:
#             placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
#             with connection.cursor() as cursor:
#                 cursor.execute(f"""
#                 SELECT af.name AS affiliation_name, COUNT(pa.paper_id) AS publications_count
#                 FROM affiliation af
#                 JOIN paper_affiliation pa ON af.id = pa.affiliation_id
#                 JOIN paper p ON pa.paper_id = p.id
#                 WHERE p.id IN ({placeholders})
#                 GROUP BY af.name
#                 ORDER BY publications_count DESC
#                 LIMIT 10;
#                 """, paper_ids)

#                 affiliation_counts = cursor.fetchall()

#         # 소속 이름과 논문 수를 결합한 리스트 생성
#         affiliation_data = list(zip([item[0] for item in affiliation_counts], [item[1] for item in affiliation_counts]))

#         # 그래프에 사용할 데이터 준비
#         context = {
#             'affiliation_data': affiliation_data,  # 소속 이름과 논문 수를 결합한 리스트
#             'keyword': user_keyword  # 검색된 키워드
#         }

#         return render(request, 'affiliation_graph.html', context)
#     else:
#         return render(request, 'error.html', {'message': 'Invalid request'})


# # 검색어에 대한 TOP 10 키워드
# def keyword_counts_graph(request):
#     if 'query' in request.GET:
#         user_keyword = request.GET['query']
#         paper_ids = get_paper_ids(user_keyword)

#         # 키워드 카운트 초기화
#         keyword_counts = Counter()

#         if paper_ids:
#             placeholders = ', '.join(['%s'] * len(paper_ids))  # placeholders 생성
#             with connection.cursor() as cursor:
#                 cursor.execute(f"""
#                 SELECT k.keyword_name
#                 FROM paper_keyword pk
#                 JOIN keyword k ON pk.keyword_id = k.id
#                 WHERE pk.paper_id IN ({placeholders});
#                 """, paper_ids)

#                 # 쿼리 결과 가져오기
#                 keywords = [row[0] for row in cursor.fetchall()]
#                 keyword_counts.update(keywords)

#         # 빈도수가 높은 top 10 키워드 출력하기
#         top_keywords = keyword_counts.most_common(10)

#         # 그래프에 사용할 데이터 준비
#         context = {
#             'top_keywords': top_keywords,  # 상위 10개의 키워드와 그 빈도수
#             'keyword': user_keyword  # 검색된 키워드
#         }

#         return render(request, 'keyword_graph.html', context)
#     else:
#         return render(request, 'error.html', {'message': 'Invalid request'})