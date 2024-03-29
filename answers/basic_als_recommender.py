from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
import sys
rand_seed=int(sys.argv[1])
file="./data/sample_movielens_ratings.txt"
spark=SparkSession.builder.master("local").appName("LAB2").getOrCreate()
lines=spark.read.text(file).rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit(weights=[0.8, 0.2],seed=rand_seed)

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, rank=70, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop", seed=rand_seed)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(rmse)

