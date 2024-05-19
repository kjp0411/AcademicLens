from django.db import models

class Affiliation(models.Model):
    name = models.TextField()
    country = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'affiliation'


class Author(models.Model):
    name = models.CharField(max_length=200)
    affiliation = models.TextField()

    class Meta:
        managed = False
        db_table = 'author'


class Keyword(models.Model):
    keyword_name = models.CharField(max_length=200)

    class Meta:
        managed = False
        db_table = 'keyword'


class Paper(models.Model):
    search = models.CharField(max_length=50)
    title = models.TextField()
    url = models.TextField(unique=True)
    date = models.DateField()
    citations = models.IntegerField()
    publisher = models.CharField(max_length=20)
    abstract = models.TextField()

    class Meta:
        managed = False
        db_table = 'paper'


class PaperAffiliation(models.Model):
    paper = models.ForeignKey(Paper, models.DO_NOTHING)
    affiliation = models.ForeignKey(Affiliation, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'paper_affiliation'


class PaperAuthor(models.Model):
    paper = models.ForeignKey(Paper, models.DO_NOTHING)
    author = models.ForeignKey(Author, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'paper_author'


class PaperKeyword(models.Model):
    paper = models.ForeignKey(Paper, models.DO_NOTHING)
    keyword = models.ForeignKey(Keyword, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'paper_keyword'