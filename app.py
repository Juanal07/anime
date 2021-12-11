from os import write
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import sys
from google.cloud import storage
import pandas as pd
import requests
import time

spark = SparkSession.builder.master("local[*]").getOrCreate()
storage_client = storage.Client()
bucket = storage_client.bucket('anime-jarr')
user_id_target=666666

ratings = spark.read.csv("gs://anime-jarr/rating_complete.csv", header=True,inferSchema=True,sep=",")
anime = spark.read.csv("gs://anime-jarr/anime.csv", header=True,inferSchema=True,sep=",")

(training,test) = ratings.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model=als.fit(training)

users = ratings.filter(ratings["user_id"]==user_id_target)
userSubsetRecs = model.recommendForUserSubset(users, 100)

movies=userSubsetRecs.first()['recommendations']

recommendations=[]
for movie in movies:
    recommendations.append(movie['anime_id'])

result = anime.filter((anime.ID).isin(recommendations)).select('ID','Name','Japanese name','Type')

df = result.toPandas()

# names = ["peliculas.txt","series.txt"]
# types = ['Movie', 'TV']
df_tv = df.loc(df['Type']=="TV")
df_movie = df.loc(df['Type']=="Movie")

def addImageVideo(df):
    images = []
    videos = []
    for i in range(5):
        print(str(df.iloc[i].loc['ID']))
        r = requests.get('https://api.jikan.moe/v3/anime/'+str(df.iloc[i].loc['ID']))
        image=r.json()['image_url']
        print(r.json()['image_url'])
        video=str(r.json()['trailer_url'])
        print(r.json()['trailer_url'])
        images.append('<img src="'+image+'" />')
        videos.append('<iframe width="420" height="315" src="'+video+'"></iframe>')
        time.sleep(2)
    df['Image'] = images
    df['Trailer'] = videos
    return df

def save(df, name):
    df.to_csv("output/{}.csv".format(name))
    blob = bucket.blob("output/{}.csv".format(name))
    blob.upload_from_filename("output/{}.csv".format(name))

    df.to_html("output/{}.html".format(name),escape=False)

df = addImageVideo(df_movie)
save(df, 'movies')


