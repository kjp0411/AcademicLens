from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('analyze/', views.analyze, name='total_graph'),
    
    path('author_network/', views.author_network, name='author_network'),
    path('author-network-view/', views.author_html, name='author_html'),

    path('affiliation_network/',views.affiliation_network,name='affiliation-network'),
    path('affiliation-network-view/', views.affiliation_html, name='affiliation_html'),

    path('country_network/',views.country_network,name='country-network'),
    path('country-network-view/', views.country_html, name='country_html'),
]
