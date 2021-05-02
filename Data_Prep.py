# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# import libraries
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
import pyspark.ml.feature as ftr
from pathlib import Path

import pandas as pd
import requests
import timezonefinder
import pandas_profiling
from pandas_profiling.utils.cache import cache_file

# import libraries
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
import pyspark.ml.feature as ftr
from pathlib import Path

import requests
import timezonefinder
import pandas_profiling
from pandas_profiling.utils.cache import cache_file

# Libraries for modeling
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import PipelineModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import PCA

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark


name_dict = {"JFK": "JFK INTERNATIONAL AIRPORT",
             "BOS": "BOSTON",
             "ATL": "ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT",
             "DFW": "DALLAS FORT WORTH",
             "ORD": "CHICAGO OHARE INTERNATIONAL AIRPORT",
             "CLT": "CHARLOTTE DOUGLAS AIRPORT",
             "DCA": "WASHINGTON REAGAN NATIONAL AIRPORT",
             "IAH": "HOUSTON INTERCONTINENTAL AIRPORT",
             "SEA": "SEATTLE TACOMA INTERNATIONAL AIRPORT",
             "LAX": "LOS ANGELES INTERNATIONAL AIRPORT",
             "SFO": "SAN FRANCISCO INTERNATIONAL AIRPORT",
             "DEN": "DENVER INTERNATIONAL AIRPORT",
             "CVG": "CINCINNATI NORTHERN KENTUCKY INTERNATIONAL AIRPORT"}



# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

#dbutils.widgets.removeAll()

dbutils.widgets.dropdown("00.Airport_Code", "JFK", ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH"])
dbutils.widgets.text('01.training_start_date', "2018-01-01")
dbutils.widgets.text('02.training_end_date', "2019-03-15")
dbutils.widgets.text('03.inference_date', (dt.strptime(str(dbutils.widgets.get('02.training_end_date')), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))

training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))
print(airport_code,training_start_date,training_end_date,inference_date)


# COMMAND ----------

from pyspark.sql.types import *

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

# COMMAND ----------

#airline data
bronze_airtraffic = spark.sql("select * from dscc202_db.bronze_air_traffic")
df_base = bronze_airtraffic.filter((bronze_airtraffic.DEST == airport_code) | (bronze_airtraffic.ORIGIN == airport_code)).filter(bronze_airtraffic.FL_DATE < training_end_date).filter(bronze_airtraffic.FL_DATE > training_start_date)


# COMMAND ----------

def airlines_transform(dataframe):
  # Selected Columns
  selected_col = [
  "YEAR",
  "QUARTER",
  "MONTH",
  "DAY_OF_MONTH",
  "DAY_OF_WEEK",
  "FL_DATE",
  "FL_PATH",
  "OP_UNIQUE_CARRIER",
  "TAIL_NUM",
  "OP_CARRIER_FL_NUM",
  "ORIGIN",
  "ORIGIN_CITY_NAME",
  "ORIGIN_STATE_ABR",
  "DEST",
  "DEST_CITY_NAME",
  "DEST_STATE_ABR",
  "CRS_DEP_TIME",
  "CRS_DEP_TIME_HOUR",
  "DEP_TIME_HOUR",
  "DEP_DELAY_NEW",
  "DEP_TIME_BLK",
  "CRS_ARR_TIME",
  "CRS_ARR_TIME_HOUR",
  "ARR_TIME_HOUR",
  "ARR_DELAY_NEW",
  "ARR_TIME_BLK",
  "DISTANCE",
  "DISTANCE_GROUP",
  "DEP_DEL15",
  "ARR_DEL15",
  "ORIGIN_AIRPORT_ID",
  "DEST_AIRPORT_ID",
  "CRS_DEP_TIMESTAMP",
  "CRS_ARR_TIMESTAMP",
  "PR_ARR_DEL15"]
  
  # Creating a window partition to extract prior arrival delay for each flight
  windowSpec = Window.partitionBy("TAIL_NUM").orderBy("CRS_DEP_TIMESTAMP")
  
  return (
    dataframe
    .filter("CANCELLED != 1 AND DIVERTED != 1")
    .withColumn("FL_DATE", f.col("FL_DATE").cast("date"))
    .withColumn("OP_CARRIER_FL_NUM", f.col("OP_CARRIER_FL_NUM").cast("string"))
    .withColumn("DEP_TIME_HOUR", dataframe.DEP_TIME_BLK.substr(1, 2).cast("int"))
    .withColumn("ARR_TIME_HOUR", dataframe.ARR_TIME_BLK.substr(1, 2).cast("int"))
    .withColumn("CRS_DEP_TIME_HOUR", f.round((f.col("CRS_DEP_TIME")/100)).cast("int"))
    .withColumn("CRS_ARR_TIME_HOUR", f.round((f.col("CRS_ARR_TIME")/100)).cast("int"))
    .withColumn("DISTANCE_GROUP", f.col("DISTANCE_GROUP").cast("string"))
    .withColumn("OP_CARRIER_FL_NUM", f.concat(f.col("OP_CARRIER"),f.lit("_"),f.col("OP_CARRIER_FL_NUM")))
    .withColumn("FL_PATH", f.concat(f.col("ORIGIN"),f.lit("-"),f.col("DEST")))
    .withColumn("DEP_DEL15", f.col("DEP_DEL15").cast("string"))
    .withColumn("ARR_DEL15", f.col("ARR_DEL15").cast("string"))
    .withColumn("FL_DATE_string", f.col("FL_DATE").cast("string"))
    .withColumn("YEAR", f.col("YEAR").cast("string"))
    .withColumn("QUARTER", f.col("QUARTER").cast("string"))
    .withColumn("MONTH", f.col("MONTH").cast("string"))
    .withColumn("DAY_OF_MONTH", f.col("DAY_OF_MONTH").cast("string"))
    .withColumn("DAY_OF_WEEK", f.col("DAY_OF_WEEK").cast("string"))
    .withColumn("CRS_DEP_TIME_string", f.col("CRS_DEP_TIME").cast("string"))
    .withColumn("CRS_ARR_TIME_string", f.col("CRS_ARR_TIME").cast("string"))
    .withColumn("CRS_DEP_TIME_HOUR_string", f.col("CRS_DEP_TIME_HOUR").cast("string"))
    .withColumn("CRS_ARR_TIME_HOUR_string", f.col("CRS_ARR_TIME_HOUR").cast("string"))
    .withColumn("CRS_DEP_TIME_HH", f.lpad("CRS_DEP_TIME_string", 4, '0').substr(1,2))
    .withColumn("CRS_DEP_TIME_MM", f.lpad("CRS_DEP_TIME_string", 4, '0').substr(3,2))
    .withColumn("CRS_ARR_TIME_HH", f.lpad("CRS_ARR_TIME_string", 4, '0').substr(1,2))
    .withColumn("CRS_ARR_TIME_MM", f.lpad("CRS_ARR_TIME_string", 4, '0').substr(3,2))
    .withColumn("CRS_DEP_TIMESTAMP", f.concat(f.col("FL_DATE_string"),f.lit(" "),f.col("CRS_DEP_TIME_HH"), f.lit(":"),f.col("CRS_DEP_TIME_MM")).cast("timestamp"))
    .withColumn("CRS_ARR_TIMESTAMP", f.concat(f.col("FL_DATE_string"),f.lit(" "),f.col("CRS_ARR_TIME_HH"), f.lit(":"),f.col("CRS_ARR_TIME_MM")).cast("timestamp"))
    .withColumn("CRS_ELAPSED_TIME", f.round((f.col("CRS_ELAPSED_TIME")/60)).cast("int"))
    .withColumn("PR_ARR_DEL15", f.lag(f.col("ARR_DEL15"), 1).over(windowSpec).cast("string"))
    .select(selected_col)
    )



# COMMAND ----------


df_airlines = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "airlines_"+airport_code + ".parquet")
df_airlines = df_airlines.filter(df_airlines.FL_DATE < inference_date).filter(df_airlines.FL_DATE > training_start_date)
df_airlines.createOrReplaceTempView('airlines')

from pyspark.sql.functions import *
from pyspark.sql.types import *
df_airlines = df_airlines.withColumn("dep_time", date_trunc('hour', "CRS_DEP_TIMESTAMP")).withColumn("arr_time", date_trunc('hour', "CRS_ARR_TIMESTAMP")).withColumn("FL_PATH", f.concat(f.col("ORIGIN"),f.lit("-"),f.col("DEST"))).withColumn('ORIGIN_CITY', split(col('ORIGIN_CITY_NAME'),",")[0]).withColumn('DEST_CITY', split(col('DEST_CITY_NAME'),",")[0])


display(df_airlines)

# COMMAND ----------

from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F

dest_ls = df_airlines.select(F.collect_set('DEST').alias('DEST')).first()['DEST']
orig_ls = df_airlines.select(F.collect_set('ORIGIN').alias('ORIGIN')).first()['ORIGIN']

dest_city = df_airlines.select(F.collect_set('DEST_CITY').alias('DEST_CITY')).first()['DEST_CITY']
orig_city = df_airlines.select(F.collect_set('ORIGIN_CITY').alias('ORIGIN_CITY')).first()['ORIGIN_CITY']

tmp = df_airlines.select(["DEST", "DEST_CITY"]).distinct().toPandas()
dest_dict = dict(zip(tmp.DEST,tmp.DEST_CITY))

tmp = df_airlines.select(["ORIGIN", "ORIGIN_CITY"]).distinct().toPandas()
orig_dict = dict(zip(tmp.ORIGIN,tmp.ORIGIN_CITY))

# COMMAND ----------

l = ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH"]

city_codes = {'DEN': 'Denver', 'CLT': 'Charlotte', 'SEA': 'Seattle', 'DFW': 'Dallas Fort Worth', 'ATL': 'Atlanta', 'LAX': 'Los Angeles', 'BOS': 'Boston', 'JFK': 'New York', 'IAH': 'Houston', 'DCA': 'Washington', 'ORD': 'Chicago', 'SFO': 'San Francisco', 'CVG': 'Cincinnati'}

down_col = ["STATION", "DATE", "LATITUDE", 'LONGITUDE','NAME', 'REPORT_TYPE', 'CALL_SIGN','TMP','WND','CIG','VIS','DEW','SLP','AA1','AJ1', 'AT1', 'GA1', 'IA1', 'MA1','MD1','OC1','REM']

weather_cols = ["STATION", "DATE", "LATITUDE", 'LONGITUDE', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AA1', 'AJ1', 'AT1', 'GA1', 'IA1', 'MA1']


# COMMAND ----------

# weather data
# read from parquet
from pyspark.sql import functions as f
def weather_raw(dataframe):
  return (
    dataframe
      .withColumn("COUNTRY", f.substring(f.col("NAME"), -2, 2))
      .filter("COUNTRY = 'US'")
      .filter("REPORT_TYPE LIKE '%FM-15%'")
      .filter("NAME LIKE '% AIRPORT%'")
      .select(down_col)
  )

#read the NYC weather data from 2020 and aggregate it by the hour
from datetime import datetime as dt
from datetime import timedelta
from pyspark.sql.functions import *
from pyspark.sql.types import *

def weather_trans(weather_airport):
  return (weather_airport
          .withColumn('temp_f', split(col('TMP'),",")[0]*9/50+32)
          .withColumn('temp_qual', split(col('TMP'),",")[1])
          .withColumn('wnd_deg', split(col('WND'),",")[0])
          .withColumn('wnd_1', split(col('WND'),",")[1])
          .withColumn('wnd_2', split(col('WND'),",")[2])
          .withColumn('wnd_mps', split(col('WND'),",")[3]/10)
          .withColumn('wnd_4', split(col('WND'),",")[4])
          .withColumn('vis_m', split(col('VIS'),",")[0])
          .withColumn('vis_1', split(col('VIS'),",")[1])
          .withColumn('vis_2', split(col('VIS'),",")[2])
          .withColumn('vis_3', split(col('VIS'),",")[3])
          .withColumn('dew_pt_f', split(col('DEW'),",")[0]*9/50+32)
          .withColumn('dew_1', split(col('DEW'),",")[1])
          .withColumn('slp_hpa', split(col('SLP'),",")[0]/10)
          .withColumn('slp_1', split(col('SLP'),",")[1])
          .withColumn('precip_hr_dur', split(col('AA1'),",")[0])
          .withColumn('precip_mm_intvl', split(col('AA1'),",")[1]/10)
          .withColumn('precip_cond', split(col('AA1'),",")[2])
          .withColumn('precip_qual', split(col('AA1'),",")[3])
          .withColumn('precip_mm', col('precip_mm_intvl')/col('precip_hr_dur'))
          .withColumn("time", date_trunc('hour', "DATE"))
          .where("REPORT_TYPE='FM-15'")
          .groupby("time")
          .agg(mean('temp_f').alias('avg_temp_f'), \
               sum('precip_mm').alias('tot_precip_mm'), \
               mean('wnd_mps').alias('avg_wnd_mps'), \
               mean('vis_m').alias('avg_vis_m'),  \
               mean('slp_hpa').alias('avg_slp_hpa'),  \
               mean('dew_pt_f').alias('avg_dewpt_f'), ))



# COMMAND ----------

 
bronze_weather = spark.sql("select * from dscc202_db.bronze_weather where REPORT_TYPE='FM-15' and NAME LIKE '% US'")
bronze_weather = weather_raw(bronze_weather)
from pyspark.sql.functions import col, split
bronze_weather = bronze_weather.filter(col("DATE")> training_start_date).filter(col("DATE")<inference_date).filter(col("NAME").like("% US")).withColumn('STATE', split(col('NAME'),",")[1])
display(bronze_weather)

# COMMAND ----------

stations = spark.read.option("header", "true").parquet(GROUP_DATA_PATH + "stations.parquet")
stations.createOrReplaceTempView('airports')
#display(stations)
call_sgn = stations.select(F.collect_set('CALL_SIGN').alias('CALL_SIGN')).first()['CALL_SIGN']
dest_prf = ["K"+sub  for sub in dest_ls]
diff = set(dest_prf) - set(call_sgn)


# COMMAND ----------

dep = df_airlines.filter(col("ORIGIN").contains(airport_code))

# COMMAND ----------

arr = df_airlines.filter(col("DEST").contains(airport_code))

# COMMAND ----------

dep = dep.toPandas()
local_weather = bronze_weather.filter(col("CALL_SIGN").contains('K'+airport_code))
weather_avg = weather_trans(local_weather).toPandas()
weather_avg = weather_avg.add_prefix('dep_')
dep = pd.merge(dep, weather_avg, on="dep_time", how="left")
#display(dep)



# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "pre_dep_"+ airport_code + ".parquet", recurse=True)
dep.to_parquet("/dbfs"+ GROUP_DATA_PATH + "pre_dep_"+ airport_code + ".parquet")
dep = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "pre_dep_"+ airport_code + ".parquet")

# COMMAND ----------

display(dep)

# COMMAND ----------

tripVertices = dep.withColumnRenamed("OP_UNIQUE_CARRIER", "id").select("id").distinct()
tripEdges = dep.select(col("OP_UNIQUE_CARRIER").alias("operator"),col("DEP_DELAY_NEW").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("operator")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))



# COMMAND ----------

tripVertices = dep.withColumnRenamed("DAY_OF_WEEK", "id").select("id").distinct()
tripEdges = dep.select(col("DAY_OF_WEEK").alias("day_of_week"),col("DEP_DELAY_NEW").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("day_of_week","dst")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))




# COMMAND ----------

from pyspark.sql.types import StringType, DoubleType
dep_num=[f.name for f in dep.schema.fields if isinstance(f.dataType, DoubleType)]
df_num = dep.select(dep_num).toPandas()


# COMMAND ----------

from pyspark.sql.types import StringType, DoubleType
dep_num=[f.name for f in dep.schema.fields if isinstance(f.dataType, DoubleType)]
df_num = dep.select(dep_num).toPandas()
plot_corr(df_num)

# COMMAND ----------

tripVertices = arr.withColumnRenamed("DAY_OF_WEEK", "id").select("id").distinct()
tripEdges = arr.select(col("DAY_OF_WEEK").alias("day_of_week"),col("ARR_DELAY_NEW").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("dst") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("day_of_week","src")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))




# COMMAND ----------

#dbutils.fs.rm(GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet", recurse=True)
#dep.to_csv(GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".txt")
#dbutils.fs.rm(GROUP_DATA_PATH + "airlines_arr_"+ airport_code + ".parquet", recurse=True)
#arr.to_csv(GROUP_DATA_PATH + "airlines_arr_"+ airport_code + ".txt")

#dep = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet")
#arr = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "airlines_arr_"+ airport_code + ".parquet")

#city = dest_dict.get(dest_ls[0])
#city = city.replace(" City","").upper()
#city

# COMMAND ----------

#ARRIVALS
weather_avg = weather_trans(local_weather).toPandas()
weather_avg = weather_avg.add_prefix('arr_')
arr= arr.toPandas()
arr = pd.merge(arr, weather_avg, on="arr_time", how="left")

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "pre_arr_"+ airport_code + ".parquet", recurse=True)
arr.to_parquet("/dbfs"+ GROUP_DATA_PATH + "pre_arr_"+ airport_code + ".parquet")
arr = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "pre_arr_"+ airport_code + ".parquet")

# COMMAND ----------

display(arr)

# COMMAND ----------

tripVertices = arr.withColumnRenamed("OP_UNIQUE_CARRIER", "id").select("id").distinct()
tripEdges = arr.select(col("OP_UNIQUE_CARRIER").alias("operator"),col("ARR_DELAY_NEW").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("ORIGIN_CITY_NAME"), ',')[0].alias("city_org"), col("ORIGIN_STATE_ABR").alias("state_org"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("dst") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("operator")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))




# COMMAND ----------

from pyspark.sql.types import StringType, DoubleType
arr_num=[f.name for f in arr.schema.fields if isinstance(f.dataType, DoubleType)]
df_num = arr.select(arr_num).toPandas()
plot_corr(df_num)

# COMMAND ----------

df_dep = pd.DataFrame()
for i in dest_ls:
  j = 'K'+i
  city = dest_dict.get(i)
  try:
    city = city.replace(" City","").upper()
  except AttributeError:
    Pass
  city = city.replace("/"," ")
  city = city.replace("ST.","")
  
  print(city)
  if i in l:
    name = name_dict.get(j)
    df= bronze_weather.filter(col("NAME").contains(name))
  elif i in diff:
    df= bronze_weather.filter(col("NAME").contains(city))
  else:
    df= bronze_weather.filter(col("CALL_SIGN").contains(j))
  
  weather_avg = weather_trans(df).toPandas()
  weather_avg = weather_avg.add_prefix('arr_')
  tmp = dep[dep.DEST == i]
  tmp = pd.merge(dep[dep.DEST == i], weather_avg, on="arr_time", how="left")
  df_dep = df_dep.append(tmp)

#display(df_dep)
#dbutils.fs.rm(GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet", recurse=True)
#df_dep.write.parquet(GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet")

# COMMAND ----------

dbutils.fs.rm(GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet", recurse=True)
df_dep.to_parquet("/dbfs"+ GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet")
df_dep = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "airlines_dep_"+ airport_code + ".parquet")
display(df_dep)

# COMMAND ----------

tripVertices = df_dep.withColumnRenamed("OP_CARRIER", "id").select("id").distinct()
tripEdges = df_dep.select(col("OP_CARRIER").alias("operator"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("operator")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))

# COMMAND ----------



# COMMAND ----------

df_arr = pd.DataFrame()
arr = arr.toPandas()
for i in orig_ls:
  j = 'K'+i
  city = orig_dict.get(i)
  try:
    city = city.replace(" City","").upper()
  except AttributeError:
    Pass
  city = city.replace("/"," ")
  city = city.replace("ST.","")
  if i in l:
    name = name_dict.get(j)
    df= bronze_weather.filter(col("NAME").contains(name))
  elif i in diff:
    df= bronze_weather.filter(col("NAME").contains(city))
  else:
    df= bronze_weather.filter(col("CALL_SIGN").contains(j))
  
  weather_avg = weather_trans(df).toPandas()
  weather_avg = weather_avg.add_prefix('dep_')
  tmp = arr[arr.ORIGIN == i]
  tmp = pd.merge(arr[arr.ORIGIN == i], weather_avg, on="dep_time", how="left")
  df_arr = df_arr.append(tmp)

#display(df_arr)

# COMMAND ----------

#display(df_dep)
display(df_arr)
print(GROUP_DATA_PATH)

# COMMAND ----------

dbutils.fs.ls(GROUP_DATA_PATH)
#print(GROUP_DATA_PATH)

# COMMAND ----------


dbutils.fs.rm(GROUP_DATA_PATH + "airlines_arr_"+ airport_code + ".parquet", recurse=True)
df_arr.to_parquet("/dbfs"+GROUP_DATA_PATH + "airlines_arr_"+ airport_code + ".parquet")
df_arr = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "airlines_arr_"+ airport_code + ".parquet")
#display(df_arr)

# COMMAND ----------


#display(df_arr)
# Initiate udf function
#udf_timezone = f.udf(get_timezone, StringType())
#weather = spark.read.option("header", "true").parquet(GROUP_DATA_PATH+"weather_trans"+ airport_code + ".parquet")
#weather.createOrReplaceTempView('weather')
#arr = airline.where(airline.DEST == airport_code)

# COMMAND ----------

display(df_dep)

# COMMAND ----------

final_cols = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'CRS_DEP_TIME_HOUR', 'DEP_TIME_HOUR', 'DEP_DELAY_NEW', 'DEP_TIME_BLK', 'CRS_ARR_TIME', 'CRS_ARR_TIME_HOUR', 'ARR_TIME_HOUR', 'ARR_DELAY_NEW', 'ARR_TIME_BLK', 'DISTANCE', 'DISTANCE_GROUP', 'DEP_DEL15', 'ARR_DEL15', 'ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID', 'CRS_DEP_TIMESTAMP', 'CRS_ARR_TIMESTAMP', 'PR_ARR_DEL15', 'dep_time', 'arr_time', 'FL_PATH', 'ORIGIN_CITY','DEST_CITY', 'dep_avg_temp_f', 'dep_tot_precip_mm', 'dep_avg_wnd_mps','dep_avg_vis_m', 'dep_avg_slp_hpa', 'dep_avg_dewpt_f', 'arr_avg_temp_f','arr_tot_precip_mm', 'arr_avg_wnd_mps', 'arr_avg_vis_m','arr_avg_slp_hpa', 'arr_avg_dewpt_f']

df_arr = df_arr.select(final_cols)

# COMMAND ----------

display(weather)

# COMMAND ----------

# Create Vertices (airports) and Edges (flights)
from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.functions import lit, col

# The whole datasest
bronze_airtraffic = spark.sql("select * from dscc202_db.bronze_air_traffic")


# COMMAND ----------

tripVertices = airline.withColumnRenamed("ORIGIN", "id").select("id").distinct()
tripEdges = bronze_airtraffic.select(col("ORIGIN_AIRPORT_SEQ_ID").alias("tripId"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

# COMMAND ----------

tripGraph = GraphFrame(tripVertices,tripEdges)

# COMMAND ----------

#to be passed by the main widgets
#airport_code = 'SFO'
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("src", "state_dst", "dst")\
        .avg("delay")\
        .sort(desc("avg(delay)")))

# COMMAND ----------

# Find all of the nodes in the graph where a to c through b
motifs = tripGraph.find("(a)-[ab]->(b); (b)-[bc]->(c)").filter("(b.id = 'SFO') and (c.id = 'JFK') and (ab.delay > 500 or bc.delay > 500) and bc.tripid > ab.tripid and bc.tripid > ab.tripid + 10000")

# COMMAND ----------

ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)
display(ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(50))

# COMMAND ----------

#delay by operator
tripVertices = bronze_airtraffic.withColumnRenamed("OP_CARRIER", "id").select("id").distinct()
tripEdges = bronze_airtraffic.select(col("OP_CARRIER").alias("operator"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("operator")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))

# COMMAND ----------

display(df_arr)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
 
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.'''
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    
#df_dep.count()
#df_arr.count()

df_dep.columns == df_arr.columns

# COMMAND ----------



# COMMAND ----------

#Departure EDA

from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
import functools 
from pyspark.sql.functions import lit
#spark = SparkSession.builder.appName('pandasToSparkDF').getOrCreate()
dep_spk = df_dep.reset_index(drop=False)
dep_spk = sqlContext.createDataFrame(dep_spk)
dep_spk = dep_spk.select(final_cols)
display(dep_spk)

# COMMAND ----------

#delay by operator
tripVertices = df_arr.withColumnRenamed("FL_PATH", "id").select("id").distinct()
tripEdges = df_arr.select(col("FL_PATH").alias("path"),col("DEP_DELAY_NEW").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("path")\
        .avg("delay")\
        .sort(desc("avg(delay)")))

# COMMAND ----------



# COMMAND ----------

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
#_______________________________________________________________
# Creation of a dataframe with statitical infos on each airline:
global_stats = df_dep['DEP_DELAY_NEW'].groupby(df_dep['OP_UNIQUE_CARRIER']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('count')

font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 15}
mpl.rc('font', **font)
import matplotlib.patches as mpatches
#__________________________________________________________________
# I extract a subset of columns and redefine the airlines labeling 
df2 = df_dep.loc[:, ['OP_UNIQUE_CARRIER', 'DEP_DELAY_NEW']]
#________________________________________________________________________

fig = plt.figure(1, figsize=(16,15))
gs=GridSpec(2,2)             
ax1=fig.add_subplot(gs[0,0]) 
ax2=fig.add_subplot(gs[0,1]) 
ax3=fig.add_subplot(gs[1,:]) 
#------------------------------
# Pie chart nº1: nb of flights
#------------------------------
labels = [s for s in  global_stats.index]
sizes  = global_stats['count'].values
explode = [0.3 if sizes[i] < 20000 else 0.0 for i in range(len(abbr_companies))]
patches, texts, autotexts = ax1.pie(sizes, explode = explode,
                                labels=labels,  autopct='%1.0f%%',
                                shadow=False, startangle=0)
for i in range(len(abbr_companies)): 
    texts[i].set_fontsize(14)
ax1.axis('equal')
ax1.set_title('% of flights per company', bbox={'facecolor':'midnightblue', 'pad':5},
              color = 'w',fontsize=18)
#_______________________________________________
# I set the legend: abreviation -> airline name
comp_handler = []
for i in range(len(abbr_companies)):
    comp_handler.append(mpatches.Patch(color=colors[i],
            label = global_stats.index[i] + ': ' + abbr_companies[global_stats.index[i]]))
ax1.legend(handles=comp_handler, bbox_to_anchor=(0.2, 0.9), 
           fontsize = 13, bbox_transform=plt.gcf().transFigure)
#----------------------------------------
# Pie chart nº2: mean delay at departure
#----------------------------------------
sizes  = global_stats['mean'].values
sizes  = [max(s,0) for s in sizes]
explode = [0.0 if sizes[i] < 20000 else 0.01 for i in range(len(abbr_companies))]
patches, texts, autotexts = ax2.pie(sizes, explode = explode, labels = labels,
                                shadow=False, startangle=0,
                                autopct = lambda p :  '{:.0f}'.format(p * sum(sizes) / 100))
for i in range(len(abbr_companies)): 
    texts[i].set_fontsize(14)
ax2.axis('equal')
ax2.set_title('Mean delay at origin', bbox={'facecolor':'midnightblue', 'pad':5},
              color='w', fontsize=18)
#------------------------------------------------------
# striplot with all the values reported for the delays
#___________________________________________________________________
# I redefine the colors for correspondance with the pie charts
colors = ['firebrick', 'gold', 'lightcoral', 'aquamarine', 'c', 'yellowgreen', 'grey',
          'seagreen', 'tomato', 'violet', 'wheat', 'chartreuse', 'lightskyblue', 'royalblue']
#___________________________________________________________________
ax3 = sns.stripplot(y="AIRLINE", x="DEPARTURE_DELAY", size = 4, palette = colors,
                    data=df2, linewidth = 0.5,  jitter=True)
plt.setp(ax3.get_xticklabels(), fontsize=14)
plt.setp(ax3.get_yticklabels(), fontsize=14)
ax3.set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*[int(y) for y in divmod(x,60)])
                         for x in ax3.get_xticks()])
plt.xlabel('Departure delay', fontsize=18, bbox={'facecolor':'midnightblue', 'pad':5},
           color='w', labelpad=20)
ax3.yaxis.label.set_visible(False)
#________________________
plt.tight_layout(w_pad=3) 
