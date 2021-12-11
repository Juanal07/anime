from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import sys
from google.cloud import storage

spark = SparkSession.builder.master("local[*]").getOrCreate()
storage_client = storage.Client()
bucket = storage_client.bucket('anime-jarr')
user_id_target=666666

ratings = spark.read.csv("gs://anime-jarr/rating_complete.csv", header=True,inferSchema=True,sep=",")
anime = spark.read.csv("gs://anime-jarr/anime.csv", header=True,inferSchema=True,sep=",",escape="\"")

ratings = ratings.join(anime,ratings.anime_id==anime.ID,"inner").select(ratings["*"],anime["Type"])
ratings_tv= ratings.filter(ratings['Type']=="TV")
ratings_movies= ratings.filter(ratings['Type']=="Movie")
ratings = [ratings_movies,ratings_tv]
names = ["peliculas.txt","series.txt"]

i=0
for r in ratings:
    (training,test) = r.randomSplit([0.8, 0.2])
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
    model=als.fit(training)
    users = r.filter(r["user_id"]==user_id_target)
    userSubsetRecs = model.recommendForUserSubset(users, 5)
    movies=userSubsetRecs.first()['recommendations']
    recommendations=[]
    for movie in movies:
        recommendations.append(movie['anime_id'])
    result = anime.filter((anime.ID).isin(recommendations)).select('ID','English name','Japanese name')
    sys.stdout = open("output/"+names[i], "w+")
    result.show(truncate=False)
    sys.stdout.close()
    blob = bucket.blob("output/"+names[i])
    blob.upload_from_filename("output/"+names[i])
    i+=1

