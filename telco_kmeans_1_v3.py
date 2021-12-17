###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-137-222.ec2.internal
ssh -i ~/emr_key.pem $url

pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list --num-executors 25 --conf "spark.executor.memoryOverhead=2048" --executor-memory 9g --conf "spark.driver.memoryOverhead=6144" --driver-memory 50g --executor-cores 3 --driver-cores 5 --conf "spark.default.parallelism=150" --conf "spark.sql.shuffle.partitions=150" --conf "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2"

#Alt 10 instances
url=hadoop@ip-172-22-139-78.ec2.internal
ssh -i ~/emr_key.pem $url

pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list --num-executors 100 --conf "spark.executor.memoryOverhead=2048" --executor-memory 9g --conf "spark.driver.memoryOverhead=6144" --driver-memory 50g --executor-cores 3 --driver-cores 5 --conf "spark.default.parallelism=600" --conf "spark.sql.shuffle.partitions=600" --conf "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2"

##################################################
pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list


from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.sql.types as T
import csv
import pandas as pd
import numpy as np
import sys
from pyspark.sql import Window
from pyspark.sql.functions import rank, col
#import geohash2 as geohash
#import pygeohash as pgh
from functools import reduce
from pyspark.sql import *
from pyspark import StorageLevel

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

###############################################################################
############################### MODEL 1 #######################################
###############################################################################

##################
# Read model csv #
##################
path = 's3a://ada-dev/Natassha/202105/clustering_telco/M1_kmeans_incl_prediction_V3.csv/*.csv'
model = spark.read.csv(path, header = True).withColumnRenamed('prediction','cluster')
model = model.cache()
model.show(5,0)
model.printSchema()


###############################
# Agg all metrics by cluster #
###############################

aggregate = ['r','vf','m','usage_frequency','avg_spend','last_seen','device_price','overall_brq','null','Auto Vehicles','Beauty','Call and Chat','Career','Couple App','Dating App','Education',
'Finance','Food and Beverage','Games','Medical','Music','News','Parenting','Personal Productivity','Photo Video','Religious Apps','Social App Accessories','Sports and Fitness',
'Travel','Video Streaming','eCommerce','overall_brq_poi','AUT','EDU','FIN','FNB','HLT','HOT','REL','RET','STY','WOR']

exprs = [mean(x) for x in aggregate]
df = model.groupBy('cluster').agg(*exprs).sort(col('cluster'), descending = False)
df = df.cache()
df.show()
df.printSchema()

for i,var in zip(df.columns[1:],aggregate):
        df = df.withColumnRenamed(''+i+'',''+var+'')

df = df.cache()
df.show()
df.printSchema()

# Group & Join with ifa count
temp = model.groupBy('cluster').agg(count('ifa').alias('ifa_count')).sort(col('cluster'), descending = False)
temp = temp.cache()
temp.show()

df1 = df.join(temp, on='cluster')
df1 = df1.cache()
df1.show(20,0)
df1.printSchema()


####################################################################################
# convert app and poi mean brq counts to % over mean overall brq & overall brq poi #
####################################################################################
list_app = ['null','Auto Vehicles','Beauty','Call and Chat','Career','Couple App','Dating App','Education','Finance','Food and Beverage','Games',
'Medical','Music','News','Parenting','Personal Productivity','Photo Video','Religious Apps','Social App Accessories','Sports and Fitness','Travel','Video Streaming','eCommerce']

list_poi = ['AUT','EDU','FIN','FNB','HLT','HOT','REL','RET','STY','WOR']

for i in df1.columns:
    if i in list_app:
        print(''+i+'')
        df1 = df1.withColumn(''+i+'', (F.col(''+i+'')/F.col('overall_brq')) )
    else:
        if i in list_poi:
            print(''+i+'')
            df1 = df1.withColumn(''+i+'', (F.col(''+i+'')/F.col('overall_brq_poi')) )

df1 = df1.cache()
df1.show(20,0)

#df1.select('Auto Vehicles','Beauty','Call and Chat','Career','Couple App','Dating App','Education','Finance','Food and Beverage','Games',\
#'Medical','Music','News','Parenting','Personal Productivity','Photo Video','Religious Apps','Social App Accessories','Sports and Fitness','Travel','Video Streaming','eCommerce')\
#.show()

#df1.select('AUT','EDU','FIN','FNB','HLT','HOT','REL','RET','STY','WOR')\
#.show()

##################
#  AGE variable #
##################
age = model.select('cluster','age')
age = age.withColumn('age', F.when(col('age') == 0.0, F.lit('18-24')).when(col('age') == 1.0, F.lit('25-34')).when(col('age') == 2.0, F.lit('35-49')).when(col('age') == 3.0, F.lit('50+')))
age = age.cache()
age.show()

age.select('age').distinct().show(20,0)

# group and agg age by cluster
age_temp = age.groupBy('cluster').pivot('age').agg(count('age')).sort(col('cluster'), ascending = True)
age_temp = age_temp.cache()
age_temp.show()

# Join age with total ifa count
age_df = age_temp.join(temp, on='cluster')
age_df = age_df.cache()
age_df.show()




###############################################################
#####        Getting the IFA LIST for each cluster      ######
###############################################################
# For each cluster
list = ['0','1','2','3','4','5']

for i in list:
    print('CLUSTER_'+i+'')
    df = model.filter(col('cluster') == ''+i+'').select('ifa').distinct()
    df = df.cache()
    df.count() #
    df.write.parquet('s3a://ada-dev/segment_creation/ID_telco/output/ifa_list/parquet/cluster_'+i+'')
    df = df.unpersist()
    print('Done')

df = model.select('ifa').distinct()
df = df.cache()
df.count() #
df.write.parquet('s3a://ada-dev/segment_creation/ID_telco/output/ifa_list/parquet/all')

# Write CSV
# For each cluster
list = ['0','1','2','3','4','5']

for i in list:
    print('CLUSTER_'+i+'')
    df = spark.read.parquet('s3a://ada-dev/segment_creation/ID_telco/output/ifa_list/parquet/cluster_'+i+'/*.parquet')
    df = df.withColumnRenamed('ifa', 'Mobile Device ID')
    df = df.cache()
    df.count()
    df.coalesce(1).write.csv('s3a://ada-dev/segment_creation/ID_telco/output/ifa_list/csv/cluster_'+i+'', header=True)
    print('Done')




##################################################################################################################################
# Sanity Check Using MEDIAN Instead of AVG or MEAN to see if MEAN is heavily influenced by outliers or if it is close to MEDIAN #
##################################################################################################################################
# Getting Average Sum for M raw values #
aggregate = ['avg_spend','device_price']

test = model
for i in aggregate:
    test1 = test.groupBy('cluster').agg(F.percentile_approx(''+i+'', 0.5, lit(1000000)))
    test = test.join(test1, on='cluster')

for i,var in zip(test.columns[-2:],aggregate):
        test = test.withColumnRenamed(''+i+'','median_'+var+'')

test = test.cache()
test.show(2,0)
#test.printSchema()

exprs = [mean(x) for x in aggregate]
df = test.groupBy('cluster').agg(*exprs).sort(col('cluster'), descending = False)
df = df.cache()
df.show()
df.printSchema()


test1 = model.groupBy('cluster').agg(F.percentile_approx('avg_price', 0.5, lit(1000000)).alias('median_avg_spend')).sort(col('cluster'), ascending = True)
test1.show(20,0)

test2 = model.groupBy('cluster').agg(F.percentile_approx('device_price', 0.5, lit(1000000)).alias('median_device_price')).sort(col('cluster'), ascending = True)
test2.show(20,0)



##########
