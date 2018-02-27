from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit
from pyspark.sql import Row, SparkSession
import sys
rand_seed=sys.argv[1]
file="..\data\sample_movielens_ratings.txt"
spark=SparkSession.builder.master("local").appName("LAB2").getOrCreate()
lines=spark.read.text(file).rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit(weights=[0.8, 0.2],seed=rand_seed)
avg=training.agg({"rating":"avg"}).collect()[0][0]
#print(avg)

Pred=test.withColumn('globalAVG', lit(avg))

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

# Evaluate the model by computing the RMSE on the test data

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="globalAVG")
rmse = evaluator.evaluate(Pred)
print("Root-mean-square error = " + str(rmse))

