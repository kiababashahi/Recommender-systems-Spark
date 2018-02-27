from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit
from pyspark.sql.functions import *
from pyspark.sql import Row, SparkSession
import sys
rand_seed=sys.argv[1]
p=int(sys.argv[2])
file="..\data\sample_movielens_ratings.txt"
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
dff=newdf.sort('userId','movieId',ascending=True)
dff=dff.select("userId","movieId","rating","user-mean","item-mean","user-item-interaction")
dff.show(n=p)









