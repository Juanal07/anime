from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from google.cloud import storage
import requests
import time

spark = SparkSession.builder.master("local[*]").getOrCreate()

storage_client = storage.Client()
bucket = storage_client.bucket('anime-jarr')

user_id_target=666666

ratings = spark.read.csv("gs://anime-jarr/rating_complete.csv", header=True,inferSchema=True,sep=",")
anime = spark.read.csv("gs://anime-jarr/anime.csv", header=True,inferSchema=True,sep=",")

(training,test) = ratings.randomSplit([0.8, 0.2])
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model=als.fit(training)

users = ratings.filter(ratings["user_id"]==user_id_target)
userSubsetRecs = model.recommendForUserSubset(users, 100)

movies=userSubsetRecs.first()['recommendations']
recommendations=[]
for movie in movies:
    recommendations.append(movie['anime_id'])
result = anime.filter((anime.ID).isin(recommendations)).select('ID','Name','Japanese name','Type')

df_tv = result.filter(result['Type']=="TV").toPandas()
df_movie = result.filter(result['Type']=="Movie").toPandas()

name='Series'

local_df= df_tv[0:5]
images = []
videos = []
for i in range(5):
    # print(str(local_df.iloc[i].loc['ID']))
    r = requests.get('https://api.jikan.moe/v3/anime/'+str(local_df.iloc[i].loc['ID']))
    image=str(r.json()['image_url'])
    print(r.json()['image_url'])
    video=str(r.json()['trailer_url'])
    print(r.json()['trailer_url'])
    images.append('<img src="'+image+'" />')
    videos.append('<iframe width="420" height="315" src="'+video+'"></iframe>')
    time.sleep(2)
print(local_df)
print(images)
print(videos)
local_df['Image'] = images
local_df['Trailer'] = videos
local_df.to_csv("output/{}.csv".format(name))
blob = bucket.blob("output/{}.csv".format(name))
blob.upload_from_filename("output/{}.csv".format(name))
local_df.to_html("output/{}.html".format(name),escape=False)
blob = bucket.blob("output/{}.html".format(name))
blob.upload_from_filename("output/{}.html".format(name))

# save(df_tv, 'Series')
# save(df_movie, 'Películas')

