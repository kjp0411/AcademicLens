from django.urls import path
from . import views
from .views import country_publications_graph, author_publications_graph, year_publications_graph, affiliation_publication_graph, keyword_counts_graph

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('analyze/', views.analyze, name='total_graph'),
    path('country-graph/', country_publications_graph, name='country_graph'),
    path('author-graph/', author_publications_graph, name='author_graph'),
    path('year-graph/', year_publications_graph, name='year_graph' ),
    path('affiliation-graph/', affiliation_publication_graph, name='affiliation_graph' ),
    path('keyword-graph/', keyword_counts_graph, name='keyword_graph' ),
]
