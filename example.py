from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

spark = SparkSession.builder.master("local[*]").getOrCreate()

lines = spark.read.text("gs://movielens-small/ratings.csv").rdd
parts=lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]),timestamp=int(p[3])))

ratings = spark.createDataFrame(ratingsRDD)
(training,test) = ratings.randomSplit([0.8, 0.2])
# Entrenamos el modelo. La estrategia coldstartcon 'drop' descaratavalores NaNen evaluación
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model=als.fit(training)
# Evaluamos el modelo con RMSE
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-squareerror = "+str(rmse))
# Generamos las 10 mejores recomendaciones para cada usuario
# userRecs = model.recommendForAllUsers(10)
# # Generamos los top 10 usuarios para cada película
# movieRecs = model.recommendForAllItems(10)
# Generar las 10 mejores recomendaciones para un subconjunto de usuarios
users = ratings.select(als.getUserCol()).distinct().limit(3)
# userSubsetRecs = model.recommendForUserSubset(users, 10)
# # Geenararla recomendación con el top 10 usuarios para el subconjunto de películas dado
# movies = ratings.select(als.getItemCol()).distinct().limit(3)
# movieSubSetRecs = model.recommendForItemSubset(movies, 10)
