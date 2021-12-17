###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-133-131.ec2.internal
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


############
#Parameters#
############
COUNTRY = 'ID'
YEAR = '2021{01,02,03,04}'

###########
# RECENCY #
###########
# Join userbase df with brq to get timestamp in the past month
path = 's3a://ada-dev/segment_creation/ID_telco/user_base/*/*.parquet'
ub = spark.read.parquet(path)
#ub.printSchema()
#ub.select('req_con_type_desc').distinct().show()

# brq
path = 's3a://ada-prod-data/etl/data/brq/agg/agg_brq/monthly/'+COUNTRY+'/'+YEAR+'/*.parquet'
brq = spark.read.parquet(path).select('ifa', explode('app')).select('ifa', 'col.last_seen', 'col.brq_count').cache()
brq.show(5,0)
#brq.printSchema()

# Join
join = ub.join(brq, on = 'ifa')
join = join.cache()
join.show(5,0)

## Adding columns
df = join.withColumn("date", F.to_date(F.col("last_seen")))
df = df.withColumn("month", F.trunc("date", "month"))
df = df.withColumn('period_months',F.months_between(F.lit('2021-04-01'),F.col('month')))
df = df.withColumn('days_since',F.datediff(F.lit('2021-04-30'),F.col('date')))

df = df.cache()
df.show(5,0)
save_path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/main_df'
df.write.parquet(save_path, mode = 'overwrite')

###########```````````````````````````````````````````````````````````````###########
###########                  Checking brq distribution                    ###########
###########,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,###########
bdf = join.select('ifa', 'brq_count').distinct()  #Remove duplicates caused by conn type col
bdf = bdf.groupBy('ifa').agg(F.sum('brq_count').alias('sum_brq'))
bdf = bdf.cache()
bdf.show(5,0)

res = bdf.groupBy('sum_brq').agg(F.count('ifa').alias('ifa')).sort('sum_brq')
res.count()
res_path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/brq_count'
res.coalesce(1).write.csv(res_path, header = True, mode = 'overwrite')


###############
## Monetary ##
###############
COUNTRY = 'ID'
YEAR = '2021{01,02,03,04}'
year_list = ['202101','202102','202103','202104']


# Reference
master_df = spark.read.csv('s3a://ada-prod-data/reference/app/master_all/all/all/all/app.csv', header=True)
level_df = spark.read.csv('s3a://ada-prod-data/reference/app/app_level/all/all/all/app_level.csv', header=True)
lifestage_df = spark.read.csv('s3a://ada-prod-data/reference/app/lifestage/all/all/all/app_lifestage.csv', header=True)

join_df1 = master_df.join(level_df, on='app_level_id', how='left').cache()
join_df2 = join_df1.join(lifestage_df, on='app_lifestage_id', how='left').cache()

select_columns = ['bundle','app_l1_name','app_l2_name','app_l3_name','lifestage_name']
finalapp_df = join_df2.select(*select_columns)
finalapp_df = finalapp_df.cache()
finalapp_df.show()

# Loop
for i in year_list:
    print('{}'.format(i))
    # Device
    path = 's3a://ada-prod-data/etl/data/brq/sub/device/monthly/'+COUNTRY+'/{}/*.parquet'.format(i)
    df_device = spark.read.parquet(path)
    df_device = df_device.withColumn("filename",F.input_file_name())
    df_device = df_device.withColumn("month", F.split(F.col("filename"), "/").getItem(10))
    window_spec = Window.partitionBy('ifa').orderBy(F.col("month").desc())
    df_device = df_device.withColumn('rank', F.row_number().over(window_spec))
    df_device = df_device.filter(F.col('rank')==1)
    df_device = df_device.drop(*['filename','rank'])
    device = df_device.select('ifa', 'device.device_name', 'device.device_model', 'device.price')\
            .filter(col('price') != '0.0')\
            .withColumn('myr_price', F.col('price')*4.13).drop('price', 'device_name', 'device_model') #.distinct()
    #app df
    brq = spark.read.parquet('s3a://ada-prod-data/etl/data/brq/agg/agg_brq/monthly/'+COUNTRY+'/{}/*.parquet'.format(i)) #change the country
    brq2 = brq.select('ifa',explode('app'),'connection').select('ifa','col.*',explode('connection')).select('ifa','bundle','col.req_con_type_desc','brq_count')
    app = brq2.join(finalapp_df, on='bundle', how='left').drop('bundle', 'ndays').cache()
    app.show(5,0)
    #app.printSchema()
    # Video streaming app usage while on data
    beh = app.filter(col('app_l1_name') == 'Video Streaming').select('ifa', 'brq_count')\
        .filter( (~col('req_con_type_desc').like('WIFI')) & (~col('req_con_type_desc').like('Unknown')) )
    beh_g = beh.groupBy('ifa').agg(sum('brq_count').alias('data_streaming'))
    beh_g = beh_g.cache()
    beh_g.show(10,0)
    #beh.select('app_l1_name').distinct().show(10,0)
    main_df3 = device.join(beh_g, on='ifa')
    main_df4 = main_df3.withColumn('spend', F.col('myr_price') + F.col('data_streaming')).select('ifa', 'spend') #.distinct()
    # Save
    save_path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/spend_df/{}'.format(i)
    main_df4.write.parquet(save_path, mode = 'overwrite')


###########```````````````````````````````````````````````````````````````###########
###########                      Daily Data Users                         ###########
###########,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,###########
#Read Main df
path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/main_df/*.parquet'
main_df = spark.read.parquet(path).cache()
main_df.show(5,0)

'''
+------------------------------------+-----------------+----------------+-------------------+---------+----------+----------+-------------+----------+
|ifa                                 |req_con_type_desc|req_carrier_name|last_seen          |brq_count|date      |month     |period_months|days_since|
+------------------------------------+-----------------+----------------+-------------------+---------+----------+----------+-------------+----------+
|00000acc-c34a-41a8-9f2f-839429b04d13|4G               |PT. Excelcom    |2021-01-26 07:12:21|2        |2021-01-26|2021-01-01|3.0          |94        |
|00000acc-c34a-41a8-9f2f-839429b04d13|4G               |PT. Excelcom    |2021-01-07 23:51:32|1        |2021-01-07|2021-01-01|3.0          |113       |
|00000acc-c34a-41a8-9f2f-839429b04d13|4G               |PT. Excelcom    |2021-01-29 09:53:05|44       |2021-01-29|2021-01-01|3.0          |91        |
|00000acc-c34a-41a8-9f2f-839429b04d13|4G               |PT. Excelcom    |2021-03-16 07:02:19|4        |2021-03-16|2021-03-01|1.0          |45        |
|00000acc-c34a-41a8-9f2f-839429b04d13|4G               |PT. Excelcom    |2021-03-05 13:16:51|1        |2021-03-05|2021-03-01|1.0          |56        |
+------------------------------------+-----------------+----------------+-------------------+---------+----------+----------+-------------+----------+
'''

udf = main_df.filter(~col('req_con_type_desc').like('WIFI'))
udf = udf.groupBy('ifa','date','period_months','days_since').agg(F.sum('brq_count').alias('sum_brq_daily'))
udf = udf.cache()
udf.show(5,0)

'''
udf:
+------------------------------------+----------+-------------+----------+-------------+
|ifa                                 |date      |period_months|days_since|sum_brq_daily|
+------------------------------------+----------+-------------+----------+-------------+
|0000fdd6-5ddf-4948-9f20-baafdcba8d2b|2021-04-24|0.0          |6         |9            |
|00096761-8d87-4043-9d2a-08773cd4b697|2021-03-05|1.0          |56        |95           |
|00139065-eb19-4ed3-9fdf-1687cdbbb2d4|2021-01-29|3.0          |91        |6            |
|0013dc24-6cf1-4200-bf55-84372dc2f98d|2021-01-31|3.0          |89        |35763        |
|0015ec38-15c1-42ed-832d-3cd704e5052c|2021-03-30|1.0          |31        |4050         |
+------------------------------------+----------+-------------+----------+-------------+
'''

###########```````````````````````````````````````````````````````````````###########
###########                              RFM                               ###########
###########,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,###########

rdf = udf.groupBy('ifa').agg(F.min('date').alias('earliest_date'),\
                                F.max('date').alias('latest_date'),\
                                F.max('days_since').alias('first_seen'),\
                                F.min('days_since').alias('last_seen'),\
                                F.min('period_months').alias('min_period'),\
                                F.max('period_months').alias('max_period'),\
                                F.count('date').alias('data_usage_freq'),\
                                F.sum('sum_brq_daily').alias('brq_count')\
                                )

rdf = rdf.withColumn('duration_months',(F.col('max_period') - F.col('min_period') + 1))
#rdf.show(5,0)
#rdf.select('last_seen').distinct().show(100,0)

#Join with spend df
path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/spend_df/*/*.parquet'
spend_df = spark.read.parquet(path)
window_spec = Window.partitionBy('ifa').orderBy(F.col("spend").desc())
spend_df = spend_df.withColumn('rank', F.row_number().over(window_spec))
spend_df = spend_df.filter(F.col('rank')==1)
spend_df = spend_df.drop(*['rank'])

rdf2 = rdf.join(spend_df, on='ifa').cache()
rdf2.show(5,0)


'''
rdf2:
+------------------------------------+-------------+-----------+----------+---------+----------+----------+---------------+---------+---------------+------------------+
|ifa                                 |earliest_date|latest_date|first_seen|last_seen|min_period|max_period|data_usage_freq|brq_count|duration_months|spend             |
+------------------------------------+-------------+-----------+----------+---------+----------+----------+---------------+---------+---------------+------------------+
|00004a35-f154-4da8-9b21-84860ee07019|2021-01-03   |2021-04-19 |117       |11       |0.0       |3.0       |14             |4482     |4.0            |1768.52           |
|0000cb3b-9112-4edb-a011-bfb92b1645b4|2021-01-06   |2021-04-30 |114       |0        |0.0       |3.0       |17             |22888    |4.0            |838.74            |
|00056774-f603-4618-bc04-3057e0b185ba|2021-01-07   |2021-04-28 |113       |2        |0.0       |3.0       |16             |4848     |4.0            |768.8             |
|00057727-1c56-43d7-864a-96907e6793b3|2021-01-16   |2021-04-28 |104       |2        |0.0       |3.0       |16             |16480    |4.0            |4006.7999999999997|
|00062797-088e-463a-9d35-166ceebe90b3|2021-03-08   |2021-03-08 |53        |53       |1.0       |1.0       |1              |13       |1.0            |661.8             |
+------------------------------------+-------------+-----------+----------+---------+----------+----------+---------------+---------+---------------+------------------+'''

## R,F,M variables
rfm = rdf2.withColumn('usage_frequency',F.col('data_usage_freq')/F.col('duration_months'))\
         .withColumn('brq_frequency',F.col('brq_count')/F.col('duration_months'))\
         .withColumn('avg_spend',F.round(F.col('spend')/F.col('duration_months')))\
.select('ifa','last_seen','usage_frequency','brq_frequency','data_usage_freq','avg_spend','brq_count').cache() #.drop_duplicates(['ifa'])

rfm.show(4,0)
#rfm.select('ifa').distinct().count() #163469735

'''
rfm:
+------------------------------------+---------+---------------+-------------+---------------+---------+---------+
|ifa                                 |last_seen|usage_frequency|brq_frequency|data_usage_freq|avg_spend|brq_count|
+------------------------------------+---------+---------------+-------------+---------------+---------+---------+
|00004a35-f154-4da8-9b21-84860ee07019|11       |3.5            |1120.5       |14             |442.0    |4482     |
|0000cb3b-9112-4edb-a011-bfb92b1645b4|0        |4.25           |5722.0       |17             |210.0    |22888    |
|00056774-f603-4618-bc04-3057e0b185ba|2        |4.0            |1212.0       |16             |192.0    |4848     |
|00057727-1c56-43d7-864a-96907e6793b3|2        |4.0            |4120.0       |16             |1002.0   |16480    |
+------------------------------------+---------+---------------+-------------+---------------+---------+---------+
'''
###############################################
### R Distribution (ifa_count vs last_seen) ##
###############################################
#r_dist = rfm.select('ifa', 'last_seen').withColumn('last_seen_weeks', (F.col('last_seen')/7)).distinct()
#r_dist = r_dist.groupBy('last_seen_weeks').agg(F.count('ifa').alias('ifa_count')).sort(col('last_seen_weeks'), ascending = True)
r_dist = rfm.groupBy('last_seen').agg(F.countDistinct('ifa').alias('ifa_count')).sort(col('last_seen'), ascending = True).cache()
r_dist.show(20,0)

r_dist.coalesce(1).write.csv('s3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/r_dist', header = True)

### F Distribution
f_dist = rfm.groupBy('usage_frequency').agg(F.countDistinct('ifa').alias('ifa_count')).sort(col('usage_frequency'), ascending = True).cache()
f_dist.show(20,0)

f_dist.coalesce(1).write.csv('s3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/f_dist', header = True)

### M Distribution
m_dist = rfm.groupBy('avg_spend').agg(F.countDistinct('ifa').alias('ifa_count')).sort(col('avg_spend'), ascending = True).cache()
m_dist.show(20,0)

m_dist.coalesce(1).write.csv('s3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/m_dist', header = True, mode = 'overwrite')


## Percentile ranking ##
r_window = Window.orderBy(F.col('last_seen').desc())
vf_window = Window.orderBy(F.col('usage_frequency'))
bf_window = Window.orderBy(F.col('brq_frequency'))
m_window = Window.orderBy(F.col('avg_spend'))
rfm_score = rfm.withColumn('r_percentile',F.percent_rank().over(r_window))\
         .withColumn('vf_percentile',F.percent_rank().over(vf_window))\
         .withColumn('bf_percentile',F.percent_rank().over(bf_window))\
         .withColumn('m_percentile',F.percent_rank().over(m_window))

## R,F,M scores
rfm_score = rfm_score.withColumn('r',F.col('r_percentile')*5.0)\
         .withColumn('vf',F.col('vf_percentile')*5.0)\
         .withColumn('bf',F.col('bf_percentile')*5.0)\
         .withColumn('m',F.col('m_percentile')*5.0).cache()

rfm_score.show(5,0)

'''
+------------------------------------+---------+---------------+-------------+---------------+---------+---------+-------------------+--------------------+-------------------+-------------------+------------------+------------------+------------------+-------------------+
|ifa                                 |last_seen|usage_frequency|brq_frequency|data_usage_freq|avg_spend|brq_count|r_percentile       |vf_percentile       |bf_percentile      |m_percentile       |r                 |vf                |bf                |m                  |
+------------------------------------+---------+---------------+-------------+---------------+---------+---------+-------------------+--------------------+-------------------+-------------------+------------------+------------------+------------------+-------------------+
|670759dd-5377-4754-9e8b-59bbc7c2f19c|21       |1.4            |5950.8       |7              |28.0     |29754    |0.19755155636823812|0.08695722784073054 |0.8451209849906176 |0.0                |0.9877577818411907|0.4347861392036527|4.225604924953088 |0.0                |
|89fe0e02-f4e4-4516-8fd2-a2c63583444a|4        |1.2            |100.8        |6              |31.0     |504      |0.40095631932790865|0.058235924161095944|0.15802084218602502|3.15346345679826E-8|2.0047815966395435|0.2911796208054797|0.7901042109301251|1.57673172839913E-7|
|cf215831-d575-4c5a-99dd-c1588a0db00f|2        |2.8            |345.0        |14             |31.0     |1725     |0.48767445156934475|0.41932480247887455 |0.336015154283988  |3.15346345679826E-8|2.4383722578467237|2.096624012394373 |1.68007577141994  |1.57673172839913E-7|
|ca9501eb-8cf8-423d-a8b1-5e5ce1b16752|1        |2.6            |442.8        |13             |31.0     |2214     |0.5666107189691706 |0.363413485439592   |0.3828685574938669 |3.15346345679826E-8|2.833053594845853 |1.81706742719796  |1.9143427874693344|1.57673172839913E-7|
|3e9831bf-a2a1-4132-82ac-85547d7e261b|2        |4.0            |2483.2       |20             |31.0     |12416    |0.48767445156934475|0.6498350659775508  |0.7188964833048549 |3.15346345679826E-8|2.4383722578467237|3.249175329887754 |3.5944824165242744|1.57673172839913E-7|
+------------------------------------+---------+---------------+-------------+---------------+---------+---------+-------------------+--------------------+-------------------+-------------------+------------------+------------------+------------------+-------------------+
'''

rfm_path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/output_rfm'
rfm_score.coalesce(1).write.csv(rfm_path, header = True, mode='overwrite')

# Sanity count
path = 's3a://ada-dev/segment_creation/ID_telco/data_prep/rfm/output_rfm/*.csv'
df = spark.read.csv(path, header=True)
df.select('ifa').count() #31,711,166
df.select('ifa').distinct().count() #31,711,166

###########
Caveat
1 brq_count = RM 1
spend = device price + data streaming usage count


###############
