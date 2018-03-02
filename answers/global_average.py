from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
#mport sys
#rand_seed=sys.argv[1]
file="./data/sample_movielens_ratings.txt"
spark=SparkSession.builder.master("local").appName("LAB2").getOrCreate()
lines=spark.read.text(file).rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])
print(training.agg({"rating":"avg"}).collect()[0][0])