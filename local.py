from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from google.cloud import storage
import requests
import time
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

spark = SparkSession.builder.master("local[*]").getOrCreate()
user_id_target=0
# user_id_target=666666
ratings = spark.read.csv("dataset/rating_test.csv", header=True,inferSchema=True,sep=",")
anime = spark.read.csv("dataset/anime.csv", header=True,inferSchema=True,sep=",")
# ratings = spark.read.csv("gs://anime-jarr/rating_complete.csv", header=True,inferSchema=True,sep=",")
# anime = spark.read.csv("gs://anime-jarr/anime.csv", header=True,inferSchema=True,sep=",")
(training,test) = ratings.randomSplit([0.8, 0.2])
als = ALS(maxIter=10, regParam=0.1, implicitPrefs=True, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model=als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
users = ratings.filter(ratings["user_id"]==user_id_target)
userSubsetRecs = model.recommendForUserSubset(users, 30)
movies=userSubsetRecs.first()['recommendations']
recommendations=[]
for movie in movies:
    recommendations.append(movie['anime_id'])
result = anime.filter((anime.ID).isin(recommendations)).select('ID','Name','Japanese name','Type')
df_tv = result.filter(result['Type']=="TV").toPandas()
df_movie = result.filter(result['Type']=="Movie").toPandas()

def save(df,name):
    local_df= df[0:5]
    images = []
    videos = []
    for i in range(5):
        r = requests.get('https://api.jikan.moe/v3/anime/'+str(local_df.iloc[i].loc['ID']))
        image=str(r.json()['image_url'])
        print(image)
        video=str(r.json()['trailer_url'])
        print(video)
        images.append('<img src="'+image+'" />')
        videos.append('<iframe width="420" height="315" src="'+video+'"></iframe>')
        time.sleep(2)
    local_df['Image'] = images
    local_df['Trailer'] = videos
    local_df.to_csv("output/{}.txt".format(name))
    local_df.to_html("output/{}.html".format(name),escape=False)

save(df_tv, 'series')
save(df_movie, 'peliculas')
