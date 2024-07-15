from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('analyze/', views.analyze, name='total_graph'),
    path('paper/<int:paper_id>/', views.paper_detail, name='paper_detail'),
    
    path('author_network/', views.author_network, name='author-network'),
    path('author-network-view/', views.author_html, name='author_html'),

    path('affiliation_network/',views.affiliation_network,name='affiliation-network'),
    path('affiliation-network-view/', views.affiliation_html, name='affiliation_html'),

    path('country_network/',views.country_network,name='country-network'),
    path('country-network-view/', views.country_html, name='country_html'),

    path('country_wordcloud/', views.country_wordcloud, name='country-wordcloud'),
    path('country-wordcloud-view/', views.country_wordcloud_html, name='country_wordcloud_html'),

    path('author_wordcloud/', views.author_wordcloud, name='author-wordcloud'),
    path('author-wordcloud-view/', views.author_wordcloud_html, name='author_wordcloud_html'),
    
    path('affiliation_wordcloud/', views.affiliation_wordcloud, name='affiliation-wordcloud'),
    path('affiliation-wordcloud-view/', views.affiliation_wordcloud_html, name='affiliation_wordcloud_html'),
]
