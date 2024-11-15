from django.urls import path
from . import views
from django.contrib import admin
from django.urls import include
from django.conf import settings
from django.conf.urls.static import static

from django.shortcuts import redirect

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('analyze/', views.analyze, name='total_graph'),
    path('paper/<int:paper_id>/', views.paper_detail, name='paper_detail'),
    
    path('country-analyze/', views.country_analyze_html, name='country_analyze'),
    path('country_network/',views.country_network,name='country-network'),
    path('country_wordcloud/', views.country_wordcloud, name='country-wordcloud'),
    path('api/country_paper_counts_by_year/', views.country_get_paper_counts_by_year, name='paper_counts_by_year'),
    path('api/country_recent_papers/', views.country_get_recent_papers, name='recent_papers'),
    path('api/country_total_papers/', views.country_get_total_papers, name='total_papers'),

    path('affiliation-analyze/', views.affiliation_analyze_html, name='affiliation_analyze'),
    path('affiliation_network/',views.affiliation_network,name='affiliation-network'),
    path('affiliation_wordcloud/', views.affiliation_wordcloud, name='affiliation-wordcloud'),
    path('api/affiliation_paper_counts_by_year/', views.affiliation_get_paper_counts_by_year, name='affiliation_paper_counts_by_year'),
    path('api/affiliation_recent_papers/', views.affiliation_get_recent_papers, name='affiliation_recent_papers'),
    path('api/affiliation_total_papers/', views.affiliation_get_total_papers, name='affiliation_total_papers'),

    path('author-analyze/', views.author_analyze_html, name='author_analyze'),
    path('author_network/', views.author_network, name='author-network'),
    path('author_wordcloud/', views.author_wordcloud, name='author-wordcloud'),
    path('api/author_paper_counts_by_year/', views.author_get_paper_counts_by_year, name='author_paper_counts_by_year'),
    path('api/author_recent_papers/', views.author_get_recent_papers, name='author_recent_papers'),
    path('api/author_total_papers/', views.author_get_total_papers, name='author_total_papers'),
    path('author/affiliation/', views.author_get_affiliation, name='author_get_affiliation'),

    path('api/analyze_network_data/', views.AnalyzeNetworkData.as_view(), name='analyze_network_data'),
    path('api/analyze_keyword_data/', views.AnalyzeKeywordData.as_view(), name='analyze_keyword_data'),

    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('accounts/', include('allauth.urls')),
    path('post/', include('post.urls', namespace='post')),
    path('', lambda r: redirect('accounts:login'), name='root'),
    path('board/', include('board.urls')),

    path('login/', views.login_html, name='login'),
    path('signup/', views.signup_html, name='signup'),
    
    path('mypage/', views.mypage_html, name='mypage'),
    path('mypage/recommended-papers/', views.recommended_papers, name='recommended_papers'),
    path('mypage/recent-papers/', views.recent_papers, name='recent_papers'),
    path('mypage/saved-papers/', views.saved_papers, name='saved_papers'),
    path('mypage/analysis-storage/', views.analysis_storage, name='analysis_storage'),
    path('save-paper/', views.save_paper, name='save_paper'),
    path('remove-paper/', views.remove_paper, name='remove_paper'),
    path('save-selected-papers/', views.save_selected_papers, name='save_selected_papers'),

    # 이미지 저장
    path('save-image/', views.save_image, name='save_image'),

    # 폴더 생성/수정/삭제
    path('create-folder/', views.create_folder, name='create_folder'),
    path('edit-folder/', views.edit_folder, name='edit_folder'),
    path('delete-folder/', views.delete_folder, name='delete_folder'),

    # 폴더 이미지 보여주기
    path('folder-images/', views.get_folder_images, name='folder_images'),

    # 리포팅
    path('get-images/', views.get_images, name='get_images'),
    path('submit-report/', views.submit_report, name='submit_report'),
    path('reporting/', views.reporting, name='reporting'),

    path('mypage/reports/', views.report_list, name='report_list'),
    path('mypage/reports/<str:folder_name>/', views.report_detail, name='report_detail'),

    path('remove-selected-papers/', views.remove_selected_papers, name='remove_selected_papers'),

    # 분석페이지 gpt 분석
    path('analyze-chart/', views.analyze_chart, name='analyze_chart'),

    # 분석 페이지
    path('analysis/', views.analysis_page, name='analysis_page'),
    path('api/country_search/', views.country_search, name='country_search'),
    path('api/author_search/', views.author_search, name='author_search'),
    path('api/affiliation_search/', views.affiliation_search, name='affiliation_search'),

    path('get-folders/',views.get_folders, name='get_folders'),
    path('accounts/', include('accounts.urls')),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)