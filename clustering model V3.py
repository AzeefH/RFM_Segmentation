# CLEAN CODES MODEL 1:

df = spark.read.option("header","true").csv('s3a://ada-dev/segment_creation/ID_telco/data_prep/combined/csv/part-00000-a528296f-0803-4ad8-90d7-1d31e90eff83-c000.csv')
df.head()
# remove null values from subset
df = df.na.drop(subset=['avg_spend','prediction','last_seen','device_price','usage_frequency','overall_brq'])
# cast to float
df2 = df.select([col(c).cast("float") for c in df.columns])


# choose subset
df2 = df2.select('avg_spend','prediction','last_seen','device_price','usage_frequency','overall_brq')
from pyspark.ml.feature import VectorAssembler
assemble=VectorAssembler(inputCols=[
    'avg_spend','prediction','last_seen','device_price','usage_frequency','overall_brq'
 ], outputCol='features')
assembled_data=assemble.transform(df2)
assembled_data.show(2)

# standardize
from pyspark.ml.feature import StandardScaler
scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)
data_scale_output.show(2)
data_scale_output = data_scale_output.withColumnRenamed('prediction','age')

# check number of clusters with prediction
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(2,10):
    KMeans_algo=KMeans(featuresCol='standardized', k=i)
    KMeans_fit=KMeans_algo.fit(data_scale_output)
    output=KMeans_fit.transform(data_scale_output)
    score=evaluator.evaluate(output)
    silhouette_score.append(score)
    print('Cluster ',i," Silhouette Score:",score)

# Cluster  2  Silhouette Score: 0.8632198341345765
# Cluster  3  Silhouette Score: 0.851593736817522
# Cluster  4  Silhouette Score: 0.3295495961978312
# Cluster  5  Silhouette Score: 0.4321076923019674
# Cluster  6  Silhouette Score: 0.44555380063480365
# Cluster  7  Silhouette Score: 0.39126076502059526
# Cluster  8  Silhouette Score: 0.3614499312104347
# Cluster  9  Silhouette Score: 0.38246180821773734

# fit kmeans to subset - 6 clusters
KMeans_algo=KMeans(featuresCol='standardized', k=6)
KMeans_fit=KMeans_algo.fit(data_scale_output)
preds=KMeans_fit.transform(data_scale_output)

# number of rows in both dataframes
print(preds.count())
print(df.count())

# add id
df = df.withColumn("id", monotonically_increasing_id())
preds = preds.withColumn("id", monotonically_increasing_id())
# join
df_pred = preds.join(df,on='id').cache()
df_pred.show(5)

# drop duplicate cols
cols_new = []
seen = set()
for c in df_pred.columns:
    cols_new.append('{}_dup'.format(c) if c in seen else c)
    seen.add(c)

df_pred = df_pred.toDF(*cols_new).select(*[c for c in cols_new if not c.endswith('_dup')])

# drop standardized and features
df_pred = df_pred.drop('standardized')
df_pred = df_pred.drop('features')

# save
df_pred.coalesce(1).write.format('com.databricks.spark.csv').save('s3://ada-dev/azeef/202105/clustering_telco/M1_kmeans_incl_prediction_V3.csv',header = 'true')
