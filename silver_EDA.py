# Databricks notebook source
# MAGIC %pip install bokeh

# COMMAND ----------

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
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType, DoubleType
from pyspark.sql.functions import col, split, lit

from pyspark.sql.functions import *
from graphframes import *
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import to_date
from pyspark.sql.functions import date_format
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

#dbutils.widgets.removeAll()

dbutils.widgets.dropdown("00.Airport_Code", "JFK", ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH"])
airport_code = str(dbutils.widgets.get('00.Airport_Code'))
print(airport_code)

#For presentation time limit, define training start date: 2018-01-01, end date 2019-06-30 before covid.
training_end_date = '2019-06-30'
training_start_date = '2018-01-01'
inference_date = '2019-07-01'


# COMMAND ----------

airports = spark.sql("select * from dscc202_group09_db.airport_loc_csv")
display(airports)

# COMMAND ----------

tmp = airports.select(["ident", "name"]).distinct().toPandas()
airport_dict = tmp.set_index('ident')['name'].to_dict()

# COMMAND ----------

bronze_airtraffic = spark.sql("select * from dscc202_db.bronze_air_traffic where CANCELLED = 0")
#bronze_airtraffic = bronze_airtraffic.na.drop()
bronze_airtraffic = bronze_airtraffic.filter(bronze_airtraffic.DEST_STATE_ABR.isNotNull())
df_base = bronze_airtraffic.filter((bronze_airtraffic.DEST == airport_code) | (bronze_airtraffic.ORIGIN == airport_code)).filter(bronze_airtraffic.FL_DATE < inference_date).filter(bronze_airtraffic.FL_DATE >= training_start_date)
display(df_base)

# COMMAND ----------

air_col = ['UNI_ID', 'MONTH','DAY_OF_MONTH','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','FLIGHT_PATH','FL_DATE',
 'ORIGIN','DEST',
 'CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DEL15','DEP_DELAY_GROUP','DEP_TIME_BLK',
 'CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_DEL15','ARR_DELAY_GROUP','ARR_TIME_BLK',
 'TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN',
 'ACTUAL_ELAPSED_TIME', 'AIR_TIME','DISTANCE','DISTANCE_GROUP',
 'CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY',
 'arr_avg_temp_f','arr_tot_precip_mm','arr_avg_wnd_mps','arr_avg_vis_m','arr_avg_slp_hpa','arr_avg_dewpt_f',
 'dep_avg_temp_f','dep_tot_precip_mm','dep_avg_wnd_mps','dep_avg_vis_m','dep_avg_slp_hpa','dep_avg_dewpt_f']

# COMMAND ----------

airlines = spark.sql("select * from dscc202_group09_db.airlines_silver")
airlines = airlines.filter(col("FL_DATE")<=training_end_date).withColumn('UNI_ID', 
                    F.concat(F.col('FLIGHT_PATH'),F.lit('_'), F.col('FL_DATE'))).select(air_col).fillna({'ARR_DEL15':0, 'CARRIER_DELAY':0, 'WEATHER_DELAY':0,'NAS_DELAY':0,'SECURITY_DELAY':0, 'LATE_AIRCRAFT_DELAY':0})
display(airlines)

# COMMAND ----------

#DEPARTURE DELAY
tripVertices = bronze_airtraffic.withColumnRenamed("ORIGIN", "id").select("id").distinct()
tripEdges = bronze_airtraffic.select(col("ORIGIN_AIRPORT_SEQ_ID").alias("tripId"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))
tripGraph = GraphFrame(tripVertices,tripEdges)
#to be passed by the main widgets
#airport_code = 'SFO'
display(tripGraph.edges.filter((col("src") == lit(airport_code)) &(col("delay") > 0))\
        .groupBy("state_dst")\
        .avg("delay")\
        .sort(desc("avg(delay)")))

# COMMAND ----------

ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)
display(ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(50))

# COMMAND ----------

#ARRIVAL DELAY
tripVertices = bronze_airtraffic.withColumnRenamed("ORIGIN", "id").select("id").distinct()
tripEdges = bronze_airtraffic.select(col("ORIGIN_AIRPORT_SEQ_ID").alias("tripId"),col("ARR_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))
tripGraph = GraphFrame(tripVertices,tripEdges)
#to be passed by the main widgets
#airport_code = 'SFO'
display(tripGraph.edges.filter((col("src") == lit(airport_code)) &(col("delay") > 0))\
        .groupBy("state_dst")\
        .avg("delay")\
        .sort(desc("avg(delay)")))

# COMMAND ----------

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
#from mpl_toolkits.basemap import Basemap
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.optimize import curve_fit
from IPython.core.interactiveshell import InteractiveShell

name_dict = {
  "UA":"United Air Lines Inc.",
  "AA":"American Airlines Inc.",
  "US":"US Airways Inc.",
  "F9":"Frontier Airlines Inc.",
  "B6":"JetBlue Airways",
  "OO":"Skywest Airlines Inc.",
  "AS":"Alaska Airlines Inc.",
  "NK":"Spirit Air Lines",
  "WN":"Southwest Airlines Co.",
  "DL":"Delta Air Lines Inc.",
  "EV":"Atlantic Southeast Airlines",
  "HA":"Hawaiian Airlines Inc.",
  "MQ":"American Eagle Airlines Inc.",
  "VX":"Virgin America",
}

# COMMAND ----------

#origin departure delay by operator
tripVertices = bronze_airtraffic.withColumnRenamed("OP_CARRIER", "id").select("id").distinct()
tripEdges = bronze_airtraffic.select(col("ORIGIN_AIRPORT_SEQ_ID").alias("tripId"),col("OP_CARRIER").alias("operator"), col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("operator")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))

# COMMAND ----------

# dest departure delay by operator
tripVertices = bronze_airtraffic.withColumnRenamed("OP_CARRIER", "id").select("id").distinct()
tripEdges = bronze_airtraffic.select(col("ORIGIN_AIRPORT_SEQ_ID").alias("tripId"),col("OP_CARRIER").alias("operator"), col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"), split(col("DEST_CITY_NAME"), ',')[0].alias("city_dst"), col("DEST_STATE_ABR").alias("state_dst"))
tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("dst") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("operator")\
        .agg(count("delay"))\
        .sort(desc("count(delay)"))) 

# COMMAND ----------

df_base = df_base.toPandas()
delay_type = lambda x:((0,1)[x > 5],2)[x > 45]
df_base['DEL_LEVEL'] = df_base['DEP_DELAY_NEW'].apply(delay_type)


# COMMAND ----------

# large delays
fig = plt.figure(1, figsize=(10,7))
ax = sns.countplot(y="OP_CARRIER", hue='DEL_LEVEL', data=df_base)
labels = [name_dict[item.get_text()] for item in ax.get_yticklabels()]
ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);
plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);
ax.yaxis.label.set_visible(False)

plt.xlabel('Flight count', fontsize=16, weight = 'bold', labelpad=10)

L = plt.legend()
L.get_texts()[0].set_text('on time (t < 5 min)')
L.get_texts()[1].set_text('small delay (5 < t < 45 min)')
L.get_texts()[2].set_text('large delay (t > 45 min)')
plt.show()

# COMMAND ----------

df_base = df_base.toPandas()
airport_mean_delays = pd.DataFrame(pd.Series(df_base['ORIGIN'].unique()))
airport_mean_delays.set_index(0, drop = True, inplace = True)
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
  
for carrier in name_dict.keys():
    df1 = df_base[df_base['OP_CARRIER'] == carrier]
    if len(df1)==0:
      pass
    else:
      test = df1['DEP_DELAY_NEW'].groupby(df_base['ORIGIN']).apply(get_stats).unstack()
      airport_mean_delays[carrier] = test.loc[:, 'mean'] 

# COMMAND ----------

sns.set(context="paper")
fig = plt.figure(1, figsize=(8,8))

ax = fig.add_subplot(1,2,1)
subset = airport_mean_delays.iloc[:50,:].rename(columns =name_dict)
subset = subset.rename(index = airport_dict)
mask = subset.isnull()
sns.heatmap(subset, linewidths=0.01, cmap="Accent", mask=mask, vmin = 0, vmax = 35)
plt.setp(ax.get_xticklabels(), fontsize=10, rotation = 85) ;
ax.yaxis.label.set_visible(False)

ax = fig.add_subplot(1,2,2)    
subset = airport_mean_delays.iloc[50:100,:].rename(columns = name_dict)
subset = subset.rename(index = airport_dict)
fig.text(0.5, 1.02, "Delays: impact of the origin airport", ha='center', fontsize = 18)
mask = subset.isnull()
sns.heatmap(subset, linewidths=0.01, cmap="Accent", mask=mask, vmin = 0, vmax = 35)
plt.setp(ax.get_xticklabels(), fontsize=10, rotation = 85) ;
ax.yaxis.label.set_visible(False)

plt.tight_layout()


# COMMAND ----------

## Following is upon the the particular airport within the time window selected

# COMMAND ----------

#departure delay by day of week & FL_path
tripVertices = airlines.withColumnRenamed("DAY_OF_WEEK", "id").select("id").distinct()
tripEdges = airlines.select(col("DAY_OF_WEEK").alias("D_Week"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"),col("DEP_DEL15").alias("del15"), col("FLIGHT_PATH").alias("path"))

tripGraph2 = GraphFrame(tripVertices,tripEdges)
display(tripGraph2.edges.filter((col("src") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("D_Week","path")\
        .agg(count("del15"))\
        .sort(desc("count(del15)")))

# COMMAND ----------

#arrival delay by day of week & FL_path
tripVertices = airlines.withColumnRenamed("UNI_ID", "id").select("id").distinct()
tripEdges = airlines.select(col("DAY_OF_WEEK").alias("D_Week"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"),col("ARR_DEL15").alias("del15"), col("FLIGHT_PATH").alias("path"))

tripGraph2 = GraphFrame(tripVertices,tripEdges)
display(tripGraph2.edges.filter((col("dst") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("D_Week","path")\
        .agg(count("del15"))\
        .sort(desc("count(del15)")))

# COMMAND ----------

# Dep delay by destinations
tripVertices = airlines.withColumnRenamed("FL_PATH", "id").select("id").distinct()
tripEdges = airlines.select(col("FL_PATH").alias("path"),col("DEP_DELAY").alias("delay"), col("ORIGIN").alias("src"), col("DEST").alias("dst"))

tripGraph = GraphFrame(tripVertices,tripEdges)
display(tripGraph.edges.filter((col("dst") == lit(airport_code)) & (col("delay") > 0))\
        .groupBy("D_Week","src")\
        .agg(count("delay"))\
        .sort(desc("count(delay)")))


# COMMAND ----------

# OVERALL CORRELATION MATRIX

from pyspark.sql.types import StringType, DoubleType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, warnings, scipy 

import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.'''
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
dep_num=[f.name for f in airlines.schema.fields if isinstance(f.dataType, DoubleType)]
df_num = airlines.select(dep_num).toPandas()
plot_corr(df_num)

# COMMAND ----------

# Dew point is the temperature at which the air becomes saturated (100 percent relative humidity).
# Security Delay - caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.
# Weather Delay - caused by extreme or hazardous weather conditions that are forecasted or manifest themselves on point of departure, enroute, or on point of arrival.
# NAS Delay - within the control of the National Airspace System (NAS) may include: non-extreme weather conditions, airport operations, heavy traffic volume, air traffic control, etc. 
# Carrier Delay - within the control of the air carrier. Examples of occurrences that may determine carrier delay are: aircraft cleaning, aircraft damage, awaiting the arrival of connecting passengers or crew, baggage, bird strike, cargo loading, catering, computer, outage-carrier equipment, crew legality (pilot or attendant rest), damage by hazardous goods, engineering inspection, fueling, handling disabled passengers, late crew, lavatory servicing, maintenance, oversales, potable water servicing, removal of unruly passenger, slow boarding or seating, stowing carry-on baggage, weight and balance delays.
# Late-arriving aircraft: A previous flight with same aircraft arrived late, causing the present flight to depart late.
display(airlines)

# COMMAND ----------

class Figure_style():
  
    def __init__(self, size_x = 11, size_y = 5, nrows = 1, ncols = 1):
        sns.set_style("white")
        sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
        self.fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(size_x,size_y,))
        # convert self.axs to 2D array
        if nrows == 1 and ncols == 1:
            self.axs = np.reshape(axs, (1, -1))
        elif nrows == 1:
            self.axs = np.reshape(axs, (1, -1))
        elif ncols == 1:
            self.axs = np.reshape(axs, (-1, 1))

    def pos_update(self, ix, iy):
        self.ix, self.iy = ix, iy

    def style(self):
        self.axs[self.ix, self.iy].spines['right'].set_visible(False)
        self.axs[self.ix, self.iy].spines['top'].set_visible(False)
        self.axs[self.ix, self.iy].yaxis.grid(color='lightgray', linestyle=':')
        self.axs[self.ix, self.iy].xaxis.grid(color='lightgray', linestyle=':')
        self.axs[self.ix, self.iy].tick_params(axis='both', which='major',
                                               labelsize=10, size = 5)

    def draw_legend(self, location='upper right'):
        legend = self.axs[self.ix, self.iy].legend(loc = location, shadow=True,
                                        facecolor = 'g', frameon = True)
        legend.get_frame().set_facecolor('whitesmoke')

    def cust_plot(self, x, y, color='b', linestyle='-', linewidth=1, marker=None, label=''):
        if marker:
            markerfacecolor, marker, markersize = marker[:]
            self.axs[self.ix, self.iy].plot(x, y, color = color, linestyle = linestyle,
                                linewidth = linewidth, marker = marker, label = label,
                                markerfacecolor = markerfacecolor, markersize = markersize)
        else:
            self.axs[self.ix, self.iy].plot(x, y, color = color, linestyle = linestyle,
                                        linewidth = linewidth, label=label)
        self.fig.autofmt_xdate()

    def cust_plot_date(self, x, y, color='lightblue', linestyle='-',
                       linewidth=1, markeredge=False, label=''):
        markeredgewidth = 1 if markeredge else 0
        self.axs[self.ix, self.iy].plot_date(x, y, color='lightblue', markeredgecolor='grey',
                                  markeredgewidth = markeredgewidth, label=label)

    def cust_scatter(self, x, y, color = 'lightblue', markeredge = False, label=''):
        markeredgewidth = 1 if markeredge else 0
        self.axs[self.ix, self.iy].scatter(x, y, color=color,  edgecolor='grey',
                                  linewidths = markeredgewidth, label=label)    

    def set_xlabel(self, label, fontsize = 14):
        self.axs[self.ix, self.iy].set_xlabel(label, fontsize = fontsize)

    def set_ylabel(self, label, fontsize = 14):
        self.axs[self.ix, self.iy].set_ylabel(label, fontsize = fontsize)

    def set_xlim(self, lim_inf, lim_sup):
        self.axs[self.ix, self.iy].set_xlim([lim_inf, lim_sup])

    def set_ylim(self, lim_inf, lim_sup):
        self.axs[self.ix, self.iy].set_ylim([lim_inf, lim_sup]) 

# COMMAND ----------



# COMMAND ----------

# should not run - this is a back-end data set
from pyspark.sql.functions import *
from pyspark.sql.types import *
#airport_code = "SFO"
#df_airlines = spark.read.option("header", "true").parquet( GROUP_DATA_PATH + "airlines_"+airport_code + ".parquet")
#df_airlines.createOrReplaceTempView('airlines')
#air_schema = df_airlines.schema
#print(GROUP_DATA_PATH + "airlines_"+airport_code + ".parquet")
#df_airlines = df_airlines.withColumn("dep_time", date_trunc('hour', "CRS_DEP_TIMESTAMP")).withColumn("arr_time", date_trunc('hour', "CRS_ARR_TIMESTAMP")).withColumn("FL_PATH", f.concat(f.col("ORIGIN"),f.lit("-"),f.col("DEST"))).withColumn('ORIGIN_CITY', split(col('ORIGIN_CITY_NAME'),",")[0]).withColumn('DEST_CITY', split(col('DEST_CITY_NAME'),",")[0])


# COMMAND ----------

#group_air_path = GROUP_DATA_PATH + "/airlines_"+airport_code+".delta"
#df_air = test.writeStream.format("delta").trigger(processingTime="1 second").option('checkpointLocation', #air_base_path + "/_checkpoint").start(group_air_path)

