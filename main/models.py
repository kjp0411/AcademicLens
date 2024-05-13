from django.db import models

class Affiliation(models.Model):
    name = models.TextField()

    class Meta:
        managed = False
        db_table = 'affiliation'


class Author(models.Model):
    name = models.CharField(max_length=200)
    affiliation = models.TextField()

    class Meta:
        managed = False
        db_table = 'author'


class Paper(models.Model):
    search = models.CharField(max_length=50)
    title = models.TextField()
    url = models.TextField()
    date = models.DateField()
    citations = models.IntegerField()
    publisher = models.CharField(max_length=20)
    abstract = models.TextField()
    keywords = models.TextField()

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