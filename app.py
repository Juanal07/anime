from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

spark = SparkSession.builder.master("local[*]").getOrCreate()

# user_id_target=666666
user_id_target=0

ratings = spark.read.csv("dataset/rating_test.csv", header=True,inferSchema=True,sep=",")
# ratings = spark.read.csv("gs://anime-jar/rating_complete.csv", header=True,inferSchema=True,sep=",")
anime = spark.read.csv("dataset/anime.csv", header=True,inferSchema=True,sep=",",escape="\"")
# anime = spark.read.csv("gs://anime-jarr/anime.csv", header=True,inferSchema=True,sep=",",escape="\"")

ratings = ratings.join(anime,ratings.anime_id==anime.ID,"inner").select(ratings["*"],anime["Type"])
# ratings.show()
ratings_tv= ratings.filter(ratings['Type']=="TV")
ratings_tv.show()
ratings_movies= ratings.filter(ratings['Type']=="Movie")
ratings_movies.show()
ratings = [ratings_movies,ratings_tv]

i=0
for r in ratings:
# Dividimos el dataset en train/test
    (training,test) = r.randomSplit([0.8, 0.2])
# Entrenamos el modelo. La estrategia coldstartcon 'drop' descarta valores NaN en evaluación
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
    model=als.fit(training)
# Evaluamos el modelo con RMSE
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    users = r.filter(r["user_id"]==user_id_target)
    userSubsetRecs = model.recommendForUserSubset(users, 5)
    movies=userSubsetRecs.first()['recommendations']
    recommendations=[]
    for movie in movies:
        recommendations.append(movie['anime_id'])
    anime.filter((anime.ID).isin(recommendations)).select('ID','English name','Japanese name').write.format("com.databricks.spark.csv").option("header", "true").save("output_{}".format(i))
    i+=1

