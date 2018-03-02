from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit
from pyspark.sql.functions import *
from pyspark.sql import Row, SparkSession
import sys
rand_seed=sys.argv[1]

file=".\data\sample_movielens_ratings.txt"
spark=SparkSession.builder.master("local").appName("LAB2").getOrCreate()
lines=spark.read.text(file).rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit(weights=[0.8, 0.2],seed=rand_seed)
global_mean=training.agg({"rating":"avg"}).collect()[0][0]
Movie_table=(training.groupBy("movieId").agg({'rating':'mean'}))
user_table=training.groupBy("userId").agg({'rating':'mean'})
df=training.join(Movie_table,"movieId")
data=df.withColumnRenamed("avg(rating)","item-mean")
df2=data.join(user_table,"userId").withColumnRenamed("avg(rating)","user-mean")
newdf=df2.withColumn('user-item-interaction',df2['rating']-(df2['user-mean']+df2['item-mean']-global_mean))


Movie_test_table=(test.join(Movie_table,"movieId")).withColumnRenamed("avg(rating)","item-mean")
test_df=Movie_test_table.join(user_table,"userId").withColumnRenamed("avg(rating)","user-mean")

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user-item-interaction",
         coldStartStrategy="drop",rank=70, seed=int(rand_seed))
model = als.fit(newdf)
# Evaluate the model by computing the RMSE on the test data
##
predictions = model.transform(test_df)
predictions=predictions.withColumn('evaluater',predictions['prediction']+predictions['user-mean']+predictions['item-mean']-global_mean)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="evaluater")
rmse = evaluator.evaluate(predictions)
print(rmse)






