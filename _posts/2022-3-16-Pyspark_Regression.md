---
title: "Linear Regression with PySpark"
date: 2022-03-16
tags: [PySpark, Big Data, Linear Regression, Data Science, Machine Learning]
header:
excerpt: "Feature engineering and linear modeling using PySpark"
mathjax: "true"
---

# Intro
I'm running a Linear Regression on credit card data using PySpark. The objective is to create a regression that predicts credit card debt using several descriptive features. In a previous notebook I explored the dataset and reduced the features down to the few I wanted to test. The dataset itself is only 5000 rows but in the case that the dataset were too large to run on a single machine the data would need to be processed across several machines. Therefore I handle the feature transformations and regressions with PySpark as if this were the case.

## Set Up


```python
# importing packages
import findspark
findspark.init()
import pyspark
import pyspark.sql.functions as F
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StringType, DoubleType

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Use this to change pandas row settings. Good for when I need to see entire dataset
# pd.set_option("display.max_rows", None)
pd.set_option('display.max_rows', 100)
```


```python
# Selected features from earlier data exploration
columns = ['inccat', 'carcatvalue', 'default', 'empcat', 'agecat', 'cardtenurecat', 'card2tenurecat', 'retire', 'carown', 'jobsat', 'creddebt', 'income', 'othdebt', 'carvalue', 'debtinc', 'cardspent', 'card2spent', 'employ', 'tollten', 'wireten', 'cardtenure']
```

Below I inital Spark, import the data into the spark session and create a temporary view so that I can also query the data with SQL.


```python
sc = pyspark.SparkContext()
# sc.stop()
```


```python
spark = pyspark.sql.SparkSession(sc)

data = spark.read.csv('credit_card_regression_set.csv', header = True, inferSchema = True)

data.createOrReplaceTempView('data')
```

These are the variables decided during my initial exploration of the 130 features.


```python
y = 'creddebt'

nums = ['income',
 'othdebt',
 'carvalue',
 'debtinc',
 'cardspent',
 'card2spent',
 'employ',
 'tollten',
 'wireten',
 'cardtenure']

cats = ['inccat',
 'carcatvalue',
 'default',
 'empcat',
 'agecat',
 'cardtenurecat',
 'card2tenurecat',
 'retire',
 'carown',
 'jobsat']

binary = ['default', 'retire']
```

# Feature Engineering

Before I run the model I need to perform transformations that prepare the data for modeling. In this case It is mainly handling null values.


```python
spark.sql("""SELECT inccat, carcatvalue,default, empcat, agecat, cardtenurecat, card2tenurecat, retire,carown, jobsat FROM data""" ).show()
```

    +------+-----------+-------+------+------+-------------+--------------+------+------+------+
    |inccat|carcatvalue|default|empcat|agecat|cardtenurecat|card2tenurecat|retire|carown|jobsat|
    +------+-----------+-------+------+------+-------------+--------------+------+------+------+
    |   2.0|        1.0|    1.0|   1.0|   2.0|          2.0|           2.0|   0.0|   1.0|   1.0|
    |   1.0|        1.0|    1.0|   1.0|   2.0|          2.0|           2.0|   0.0|   1.0|   1.0|
    |   2.0|        1.0|    0.0|   5.0|   6.0|          5.0|           5.0|   0.0|   1.0|   4.0|
    |   1.0|        1.0|    1.0|   1.0|   2.0|          2.0|           2.0|   0.0|   1.0|   2.0|
    |   1.0|        1.0|    0.0|   1.0|   3.0|          3.0|           3.0|   0.0|   0.0|   1.0|
    |   4.0|       -1.0|    0.0|   5.0|   5.0|          5.0|           3.0|   0.0|  -1.0|   2.0|
    |   4.0|        2.0|    0.0|   3.0|   5.0|          2.0|           2.0|   0.0|   1.0|   2.0|
    |   4.0|        3.0|    0.0|   4.0|   4.0|          5.0|           5.0|   0.0|   1.0|   5.0|
    |   1.0|        1.0|    0.0|   4.0|   6.0|          5.0|           5.0|   1.0|   1.0|   2.0|
    |   4.0|        3.0|    0.0|   5.0|   4.0|          2.0|           2.0|   0.0|   1.0|   4.0|
    |   2.0|        2.0|    0.0|   3.0|   5.0|          5.0|           5.0|   0.0|   1.0|   3.0|
    |   1.0|        1.0|    0.0|   2.0|   3.0|          3.0|           2.0|   0.0|   1.0|   4.0|
    |   3.0|        2.0|    0.0|   4.0|   4.0|          1.0|           1.0|   0.0|   1.0|   5.0|
    |   3.0|       -1.0|    0.0|   2.0|   5.0|          5.0|           5.0|   0.0|  -1.0|   3.0|
    |   1.0|        1.0|    0.0|   5.0|   6.0|          5.0|           5.0|   1.0|   1.0|   4.0|
    |   1.0|        1.0|    0.0|   5.0|   6.0|          5.0|           5.0|   1.0|   1.0|   5.0|
    |   5.0|        3.0|    0.0|   5.0|   5.0|          5.0|           4.0|   0.0|   1.0|   5.0|
    |   5.0|        3.0|    0.0|   5.0|   5.0|          5.0|           5.0|   0.0|   1.0|   4.0|
    |   1.0|       -1.0|    0.0|   2.0|   3.0|          3.0|           3.0|   0.0|  -1.0|   1.0|
    |   1.0|        1.0|    0.0|   3.0|   6.0|          3.0|           3.0|   1.0|   1.0|   3.0|
    +------+-----------+-------+------+------+-------------+--------------+------+------+------+
    only showing top 20 rows



The categorical variables have null values listed as '-1.0'. These null values need to be handled in order for the regression to run correctly. In this case I decided to handle them to reassigning them to the most frequent value. The code below iterates through the categorical columns, queries the column for the most frequent value and replaces all null cases with that value.


```python
for column in cats:
#     column = 'carcatvalue'
    data = data.withColumn(column, data[column].cast(StringType()))
    query = """SELECT {0}, COUNT({0}) as count FROM data GROUP BY {0} ORDER BY count DESC LIMIT 1 """.format(column)
    frequent = str(spark.sql(query).first()[0])
    data = data.withColumn(column, F.regexp_replace(column, '-1.0', frequent))

data.select(cats).show()
```

    +------+-----------+-------+------+------+-------------+--------------+------+------+------+
    |inccat|carcatvalue|default|empcat|agecat|cardtenurecat|card2tenurecat|retire|carown|jobsat|
    +------+-----------+-------+------+------+-------------+--------------+------+------+------+
    |   2.0|        1.0|    1.0|   1.0|   2.0|          2.0|           2.0|   0.0|   1.0|   1.0|
    |   1.0|        1.0|    1.0|   1.0|   2.0|          2.0|           2.0|   0.0|   1.0|   1.0|
    |   2.0|        1.0|    0.0|   5.0|   6.0|          5.0|           5.0|   0.0|   1.0|   4.0|
    |   1.0|        1.0|    1.0|   1.0|   2.0|          2.0|           2.0|   0.0|   1.0|   2.0|
    |   1.0|        1.0|    0.0|   1.0|   3.0|          3.0|           3.0|   0.0|   0.0|   1.0|
    |   4.0|        1.0|    0.0|   5.0|   5.0|          5.0|           3.0|   0.0|   1.0|   2.0|
    |   4.0|        2.0|    0.0|   3.0|   5.0|          2.0|           2.0|   0.0|   1.0|   2.0|
    |   4.0|        3.0|    0.0|   4.0|   4.0|          5.0|           5.0|   0.0|   1.0|   5.0|
    |   1.0|        1.0|    0.0|   4.0|   6.0|          5.0|           5.0|   1.0|   1.0|   2.0|
    |   4.0|        3.0|    0.0|   5.0|   4.0|          2.0|           2.0|   0.0|   1.0|   4.0|
    |   2.0|        2.0|    0.0|   3.0|   5.0|          5.0|           5.0|   0.0|   1.0|   3.0|
    |   1.0|        1.0|    0.0|   2.0|   3.0|          3.0|           2.0|   0.0|   1.0|   4.0|
    |   3.0|        2.0|    0.0|   4.0|   4.0|          1.0|           1.0|   0.0|   1.0|   5.0|
    |   3.0|        1.0|    0.0|   2.0|   5.0|          5.0|           5.0|   0.0|   1.0|   3.0|
    |   1.0|        1.0|    0.0|   5.0|   6.0|          5.0|           5.0|   1.0|   1.0|   4.0|
    |   1.0|        1.0|    0.0|   5.0|   6.0|          5.0|           5.0|   1.0|   1.0|   5.0|
    |   5.0|        3.0|    0.0|   5.0|   5.0|          5.0|           4.0|   0.0|   1.0|   5.0|
    |   5.0|        3.0|    0.0|   5.0|   5.0|          5.0|           5.0|   0.0|   1.0|   4.0|
    |   1.0|        1.0|    0.0|   2.0|   3.0|          3.0|           3.0|   0.0|   1.0|   1.0|
    |   1.0|        1.0|    0.0|   3.0|   6.0|          3.0|           3.0|   1.0|   1.0|   3.0|
    +------+-----------+-------+------+------+-------------+--------------+------+------+------+
    only showing top 20 rows



The numerical columns have a similar issue. Null values are listed as '-1.0'. I'll use similar code except I will use the mean rather than the mode in the case of numeric features.


```python
for column in nums:
    data = data.withColumn(column, data[column].cast(StringType()))
    query = """SELECT MEAN({0}) FROM data""".format(column)
    mean = str(spark.sql(query).first()[0])
    data = data.withColumn(column, data[column].cast(StringType()))
    data = data.withColumn(column, F.regexp_replace(column, '-1.0', mean))
    data = data.withColumn(column, data[column].cast(DoubleType()))

data.select(nums).show()
```

    +------+-------+------------------+-------+---------+----------+------+-------+-------+----------+
    |income|othdebt|          carvalue|debtinc|cardspent|card2spent|employ|tollten|wireten|cardtenure|
    +------+-------+------------------+-------+---------+----------+------+-------+-------+----------+
    |  31.0|   2.24|              14.3|   11.1|    81.66|      67.8|   0.0| 161.05|    0.0|       2.0|
    |  15.0|   1.57|               6.8|   18.6|     42.6|     34.94|   0.0|    0.0|1683.55|       4.0|
    |  35.0|   2.54|              18.8|    9.9|   184.22|    175.75|  16.0|    0.0|    0.0|      35.0|
    |  20.0|   1.12|               8.7|    5.7|   340.99|     18.42|   0.0|    0.0|    0.0|       5.0|
    |  23.0|   0.18|              10.6|    1.7|    255.1|    252.73|   1.0|  387.7|  410.8|       8.0|
    | 107.0|   4.93|23.232580000000013|    5.6|   228.27|       0.0|  22.0|  726.6|    0.0|      18.0|
    |  77.0|   0.96|              25.6|    1.9|   822.32|    130.14|  10.0|    0.0|    0.0|       3.0|
    |  97.0|   8.02|              55.5|   14.4|    592.7|     712.1|  11.0| 1110.4|    0.0|      25.0|
    |  16.0|   0.31|               8.6|    2.6|   326.59|    141.24|  15.0|    0.0|    0.0|      26.0|
    |  84.0|   1.67|              41.0|    4.1|   199.64|    111.17|  19.0|    0.0|    0.0|       2.0|
    |  47.0|   2.68|              28.0|    8.6|   488.97|    322.07|   8.0|    0.0|    0.0|      16.0|
    |  19.0|   0.04|               9.3|    0.9|   338.26|     55.17|   4.0|    0.0|    0.0|       6.0|
    |  73.0|   0.79|              38.5|    2.8|   534.36|    198.39|  12.0|    0.0|    0.0|       1.0|
    |  63.0|   4.69|23.232580000000013|   10.5|    593.5|    384.94|   3.0| 1776.7|    0.0|      25.0|
    |  17.0|   0.39|               9.3|    9.8|   233.17|     39.93|  27.0|    0.0|    0.0|      33.0|
    |  23.0|    1.2|              11.6|    9.3|   297.47|     98.65|  31.0| 1919.0|3159.25|      40.0|
    | 171.0|  14.44|              75.8|    9.5|   305.94|     96.23|  24.0|1444.65|1840.45|      19.0|
    | 424.0|  32.26|              88.6|   10.7|   495.75|    798.31|  29.0|3165.45|    0.0|      36.0|
    |  23.0|   0.91|23.232580000000013|    4.8|   442.09|    144.69|   4.0|1099.65|    0.0|      10.0|
    |  22.0|   2.65|              10.8|   15.2|     8.11|       0.0|  10.0|    0.0|    0.0|      10.0|
    +------+-------+------------------+-------+---------+----------+------+-------+-------+----------+
    only showing top 20 rows




```python
data_train, data_test = data.randomSplit([.80, .20], seed = 4)
```

# Creating the Pipeline

With the data clean enough for modeling I need to create a pipeline that runs the regression. Techincally this process includes additional feature engineering as the categorical variables need to be indexed and one hot encoded. Finally, all of the features need to be assembled into a single vector that PySpark is capable of reading.


```python
# Index and encoding objects
index_list = [cat +'_idx' for cat in cats]
string_index = StringIndexer(inputCols = cats, outputCols = index_list)

encoded_list = [cat + '_encoded' for cat in cats]
one_h_encode = OneHotEncoder(inputCols = index_list, outputCols = encoded_list)
```


```python
# This code block allows me to decide features. I comment the extend methods that I wish to exclude when running the model
inputs = []
inputs.extend(nums)
inputs.extend(encoded_list)
inputs
```




    ['income',
     'othdebt',
     'carvalue',
     'debtinc',
     'cardspent',
     'card2spent',
     'employ',
     'tollten',
     'wireten',
     'cardtenure',
     'inccat_encoded',
     'carcatvalue_encoded',
     'default_encoded',
     'empcat_encoded',
     'agecat_encoded',
     'cardtenurecat_encoded',
     'card2tenurecat_encoded',
     'retire_encoded',
     'carown_encoded',
     'jobsat_encoded']




```python
# This code creates the assembler and regression objects.

assembler = VectorAssembler(inputCols = inputs, outputCol = 'features')

regression = LinearRegression(labelCol='creddebt')

# pipeline object contains previously created objects
pipepline = Pipeline(stages = [string_index, one_h_encode, assembler, regression])

# model object fits the pipeline to the training data. predictions
model = pipepline.fit(data_train)

predictions = model.transform(data_test)
```

## Evaluation

The table below shows the model predictions compared to the actual credit debt. 'Creddebt' is listed in the thousands. Therefore in the case of the first observation, the model predicted that this observation with an actual debt of 140 dollars would actually be about 600 dollars in debt.


```python
predictions.select(['creddebt','prediction']).show()
```

    +--------+--------------------+
    |creddebt|          prediction|
    +--------+--------------------+
    |    0.14|  0.6093043679675625|
    |    0.24| 0.43905437552333204|
    |    0.52|0.012254495737605176|
    |    0.39|   1.507075165380026|
    |    1.13|   3.206200072134242|
    |    0.02| -0.7176964758160933|
    |    0.45|  0.8612760486636706|
    |    2.46|  2.5064292029472672|
    |    6.61|   4.050840955195323|
    |   44.25|   13.44364556197883|
    |    0.15| -0.5992770881744303|
    |    8.15|    5.87001498551057|
    |    0.39|-0.26548513694924303|
    |    0.78|  1.7494149226058622|
    |    0.34| -0.4892234737470511|
    |   29.47|   9.280357979257039|
    |    0.11| 0.22478571931758262|
    |    4.73|   6.389449128368286|
    |    0.78|  1.3024720830240581|
    |    1.77|   1.021364413293012|
    +--------+--------------------+
    only showing top 20 rows




```python
# Evaluation checks the total error of the model based on an evaluation metric in this case the metric is  root mean squared error
evaluation = RegressionEvaluator(metricName = 'rmse', labelCol='creddebt').evaluate(predictions)

evaluation
```




    2.2154065173974407



# Documentation
I created a dataframe and a function to add the results of each prediction to my model to document the models improvement as I change parameters and features.


```python
metrics = pd.DataFrame({'model': [], 'rmse':[], 'mse' : []})
# metrics.to_csv('metrics.csv')
# metrics = pd.read_csv('metrics.csv')
# metrics = metrics.iloc[:,1:]
```


```python
# These functions evaluate the rmse and mse of the column and add them to my metrics dataframe.
def add_row(df, row):
    df.loc[len(df)] = row

def add_model_metrics(df, model_name, predictions):
    rmse = RegressionEvaluator(metricName = 'rmse', labelCol='creddebt').evaluate(predictions)
    mse = RegressionEvaluator(metricName = 'mse', labelCol='creddebt').evaluate(predictions)

    row = [model_name, rmse, mse]

    add_row(df, row)
```


```python
# add_model_metrics(metrics, 'nums_and_cats', predictions)
```

These are the first two models I ran one with only the numerical variables and a second with both the numerical and categorical variables. The first model has a rmse of 2.275 which suggests that the model is off by about 2270 dollars on average. With the categorical variables, the model is 60 dollars more effective on average when predicted on the test data.


```python
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num_only</td>
      <td>2.275009</td>
      <td>5.175667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nums_and_cats</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
  </tbody>
</table>
</div>



# Additional Features

## Cross Validation
Adding cross validation (5 fold) to potentially increase model performance. Code below is similar to previous pipeline code with some additions for cross validation.


```python
index_list = [cat +'_idx' for cat in cats]
string_index = StringIndexer(inputCols = cats, outputCols = index_list)

encoded_list = [cat + '_encoded' for cat in cats]

one_h_encode = OneHotEncoder(inputCols = index_list, outputCols = encoded_list)

assembler = VectorAssembler(inputCols = inputs, outputCol = 'features')

regression = LinearRegression(labelCol='creddebt')

pipeline = Pipeline(stages = [string_index, one_h_encode, assembler, regression])

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol='creddebt')

paramGrid = ParamGridBuilder().build()

cvModel = CrossValidator(estimator = pipeline, evaluator = evaluator, numFolds = 4, estimatorParamMaps = paramGrid)

cvModel = CrossValidator.fit(cvModel, data_train)

predictions = cvModel.transform(data_test)
```


```python
add_model_metrics(metrics, 'all_vars_cv_1', predictions)
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num_only</td>
      <td>2.275009</td>
      <td>5.175667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nums_and_cats</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>2</th>
      <td>all_vars_cv_1</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
  </tbody>
</table>
</div>



It appears that cross validation had no effect on the model performance.

## Log Transformation

There are a few feature adjustments that I could potentially improve the model. First, some of the continuous variables aren't normally distributed. Applying a log transformation to some of the continuous variables notably changed the distribution to be more normal.


```python
to_be_logged = ['income',
 'othdebt',
 'carvalue',
 'cardspent',
 'card2spent',
 'employ',
 'tollten',
 'wireten']

reg_nums = ['debtinc',
 'cardtenure']

log_nums = ['log_' + column for column in to_be_logged]
```


```python
# This code goes through the to_be_logged list and applies the log transformation.
from pyspark.sql.functions import log
for column in to_be_logged:
    data = data.withColumn('log_' + column, log(data[column] + 1))
```


```python
# Have to reinitialize training data to include new log values.
data_train, data_test = data.randomSplit([.80, .20], seed = 4)
```


```python
# Ran this three times once with only continuous and a second time with categorical variables third time with both log and continuous
inputs = []
inputs.extend(log_nums)
inputs.extend(reg_nums)
index_list = [cat +'_idx' for cat in cats]
inputs.extend(index_list)
inputs.extend(to_be_logged)


string_index = StringIndexer(inputCols = cats, outputCols = index_list)

encoded_list = [cat + '_encoded' for cat in cats]

one_h_encode = OneHotEncoder(inputCols = index_list, outputCols = encoded_list)

assembler = VectorAssembler(inputCols = inputs, outputCol = 'features')

regression = LinearRegression(labelCol='creddebt')

pipeline = Pipeline(stages = [string_index, one_h_encode, assembler, regression])

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol='creddebt')

paramGrid = ParamGridBuilder().build()

cvModel = CrossValidator(estimator = pipeline, evaluator = evaluator, numFolds = 4, estimatorParamMaps = paramGrid)

cvModel = CrossValidator.fit(cvModel, data_train)

predictions = cvModel.transform(data_test)
```


```python
# log only
predictions.select(['creddebt','prediction']).show()
```

    +--------+--------------------+
    |creddebt|          prediction|
    +--------+--------------------+
    |    0.14|   1.647075423659162|
    |    0.24|  0.6398959925978822|
    |    0.52|  1.0997906096953702|
    |    0.39|  0.3504364866826766|
    |    1.13|  2.9820424472769513|
    |    0.02| -1.9022647714346022|
    |    0.45|  0.7227840152447431|
    |    2.46|  3.6401236181581247|
    |    6.61|   4.143968229124148|
    |   44.25|   9.636200858653792|
    |    0.15|   -1.18055814271729|
    |    8.15|   7.317615336208826|
    |    0.39| -0.4845952073333475|
    |    0.78|  1.3886183932839593|
    |    0.34|  -0.342778877833398|
    |   29.47|   8.245049772042242|
    |    0.11|-0.11961579436530378|
    |    4.73|   4.785692639501264|
    |    0.78|-0.01416733325522...|
    |    1.77|   1.573471743150824|
    +--------+--------------------+
    only showing top 20 rows




```python
# logs_and_cats
predictions.select(['creddebt','prediction']).show()
```

    +--------+-------------------+
    |creddebt|         prediction|
    +--------+-------------------+
    |    0.14| 1.2884720002611854|
    |    0.24|-0.1333820961933423|
    |    0.52| 0.6504041999093033|
    |    0.39| 1.1970963328240192|
    |    1.13| 2.7496023407588517|
    |    0.02|-1.4263460324400405|
    |    0.45|  -0.19447120947091|
    |    2.46|  3.573529458318564|
    |    6.61|   5.61117129623733|
    |   44.25| 10.727084238308777|
    |    0.15|-2.0566687769911294|
    |    8.15|  7.613715198840104|
    |    0.39|-1.2405597564490094|
    |    0.78| 2.6866694377778852|
    |    0.34| -1.012504694987344|
    |   29.47|  9.559866729767748|
    |    0.11|-0.5206746091395491|
    |    4.73| 5.0217777453217884|
    |    0.78| 0.7543957441303526|
    |    1.77|   0.81080977472668|
    +--------+-------------------+
    only showing top 20 rows




```python
# nums_logs_and_cats
predictions.select(['creddebt','prediction']).show()
```

    +--------+-------------------+
    |creddebt|         prediction|
    +--------+-------------------+
    |    0.14|0.49792826424154235|
    |    0.24|0.41489351806923125|
    |    0.52| 1.1100405323315137|
    |    0.39| 1.0485681882122222|
    |    1.13| 1.8889272517262299|
    |    0.02|-0.6776597182744579|
    |    0.45|0.48076946156178746|
    |    2.46| 3.4513634556554464|
    |    6.61|  4.029615100979548|
    |   44.25| 14.408801210838694|
    |    0.15|-0.8155181450877169|
    |    8.15|  7.228276573264578|
    |    0.39|-0.1515153538643217|
    |    0.78| 0.9778442365562707|
    |    0.34|0.41171528082751685|
    |   29.47| 10.902899075433812|
    |    0.11|-0.5355067880359905|
    |    4.73|  5.778526907345183|
    |    0.78|  1.628078117979622|
    |    1.77| 1.4989848161401254|
    +--------+-------------------+
    only showing top 20 rows




```python
add_model_metrics(metrics, 'nums_logs_and_cats', predictions)
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num_only</td>
      <td>2.275009</td>
      <td>5.175667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nums_and_cats</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>2</th>
      <td>all_vars_cv_1</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>num_only_log</td>
      <td>2.335687</td>
      <td>5.455432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>logs_and_cats</td>
      <td>2.252368</td>
      <td>5.073160</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nums_logs_and_cats</td>
      <td>2.085990</td>
      <td>4.351353</td>
    </tr>
  </tbody>
</table>
</div>



Running all of the continous, log adjusted variables, and categorical variables together seems to have gained the best overall performance.

## Regularization

The last thing I will try is a grid search with some hyperparameter adjustments. This will include the regularization weight along with the elastic net which balances between the l1 and l2 regularization types. I will add the best iteration to my metrics dataframe.


```python
index_list = [cat +'_idx' for cat in cats]
string_index = StringIndexer(inputCols = cats, outputCols = index_list)

encoded_list = [cat + '_encoded' for cat in cats]

one_h_encode = OneHotEncoder(inputCols = index_list, outputCols = encoded_list)

assembler = VectorAssembler(inputCols = inputs, outputCol = 'features')

regression = LinearRegression(labelCol='creddebt')

pipeline = Pipeline(stages = [string_index, one_h_encode, assembler, regression])

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol='creddebt')

paramGrid = ParamGridBuilder().addGrid(regression.regParam, [0.1, 0.01, .001]).addGrid(regression.fitIntercept, [True])
paramGrid = paramGrid.addGrid(regression.elasticNetParam, [0.0, 0.5, 1.0]).build()

cvModel = CrossValidator(estimator = pipeline, evaluator = evaluator, numFolds = 4, estimatorParamMaps = paramGrid)

cvModel = CrossValidator.fit(cvModel, data_train)

predictions = cvModel.transform(data_test)
```


```python
add_model_metrics(metrics, 'original_logs_cats_hyper_tuned', predictions)
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num_only</td>
      <td>2.275009</td>
      <td>5.175667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nums_and_cats</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>2</th>
      <td>all_vars_cv_1</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>num_only_log</td>
      <td>2.335687</td>
      <td>5.455432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>logs_and_cats</td>
      <td>2.252368</td>
      <td>5.073160</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nums_logs_and_cats</td>
      <td>2.085990</td>
      <td>4.351353</td>
    </tr>
    <tr>
      <th>6</th>
      <td>original_logs_cats_hyper_tuned</td>
      <td>2.101103</td>
      <td>4.414633</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Best Regularization Parameter: ', cvModel.bestModel.stages[-1]._java_obj.parent().getRegParam())
print('Best ElasticNet Parameter: ',cvModel.bestModel.stages[-1]._java_obj.parent().getElasticNetParam())

```

    Best Regularization Parameter:  0.1
    Best ElasticNet Parameter:  0.0


The model chosen was a ridge regression with a strength of .1. Although it performs worse than the previous model it is likely that the ridge regression is more generalizable. I'll stick with this as the final model when interpreting variables.

# Best Model Interpretation

In order to interpret the model I will run a final regression in pandas using the determined parameters. This will allow me to visualize and interpret the coefficients at a more individual level.


```python
pd_data_train = data_train.toPandas()
pd_data_test = data_test.toPandas()
```


```python
inputs = []
inputs.extend(log_nums)
inputs.extend(reg_nums)
inputs.extend(cats)
inputs.extend(to_be_logged)
inputs.append('creddebt')
inputs = set(inputs)

pd_data_train = pd_data_train[inputs]
pd_data_test = pd_data_test[inputs]
```


```python
pd_data_train = pd.get_dummies(pd_data_train)
X_train = pd_data_train.drop('creddebt', axis = 1)
y_train = pd_data_train['creddebt']
pd_data_test = pd.get_dummies(pd_data_test)
X_test = pd_data_test.drop('creddebt', axis = 1)
y_test = pd_data_test['creddebt']
```


```python
pd_data_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>debtinc</th>
      <th>card2spent</th>
      <th>income</th>
      <th>log_card2spent</th>
      <th>log_income</th>
      <th>cardtenure</th>
      <th>wireten</th>
      <th>carvalue</th>
      <th>tollten</th>
      <th>log_tollten</th>
      <th>...</th>
      <th>jobsat_1.0</th>
      <th>jobsat_2.0</th>
      <th>jobsat_3.0</th>
      <th>jobsat_4.0</th>
      <th>jobsat_5.0</th>
      <th>empcat_1.0</th>
      <th>empcat_2.0</th>
      <th>empcat_3.0</th>
      <th>empcat_4.0</th>
      <th>empcat_5.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>114.70</td>
      <td>85.0</td>
      <td>4.751001</td>
      <td>4.454347</td>
      <td>15.0</td>
      <td>621.90</td>
      <td>43.10000</td>
      <td>554.75</td>
      <td>6.320319</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.1</td>
      <td>66.60</td>
      <td>39.0</td>
      <td>4.213608</td>
      <td>3.688879</td>
      <td>31.0</td>
      <td>0.00</td>
      <td>10.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.2</td>
      <td>394.15</td>
      <td>138.0</td>
      <td>5.979265</td>
      <td>4.934474</td>
      <td>1.0</td>
      <td>30.05</td>
      <td>66.20000</td>
      <td>20.50</td>
      <td>3.068053</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25.1</td>
      <td>197.63</td>
      <td>63.0</td>
      <td>5.291444</td>
      <td>4.158883</td>
      <td>17.0</td>
      <td>0.00</td>
      <td>39.30000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.2</td>
      <td>214.91</td>
      <td>37.0</td>
      <td>5.374862</td>
      <td>3.637586</td>
      <td>19.0</td>
      <td>0.00</td>
      <td>15.80000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3982</th>
      <td>5.9</td>
      <td>839.40</td>
      <td>84.0</td>
      <td>6.733878</td>
      <td>4.442651</td>
      <td>15.0</td>
      <td>887.50</td>
      <td>30.60000</td>
      <td>557.55</td>
      <td>6.325344</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3983</th>
      <td>0.8</td>
      <td>156.35</td>
      <td>47.0</td>
      <td>5.058473</td>
      <td>3.871201</td>
      <td>19.0</td>
      <td>0.00</td>
      <td>22.40000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3984</th>
      <td>1.3</td>
      <td>84.25</td>
      <td>31.0</td>
      <td>4.445588</td>
      <td>3.465736</td>
      <td>9.0</td>
      <td>0.00</td>
      <td>17.10000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3985</th>
      <td>19.5</td>
      <td>56.98</td>
      <td>26.0</td>
      <td>4.060098</td>
      <td>3.295837</td>
      <td>6.0</td>
      <td>0.00</td>
      <td>11.60000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3986</th>
      <td>13.4</td>
      <td>211.04</td>
      <td>36.0</td>
      <td>5.356775</td>
      <td>3.610918</td>
      <td>3.0</td>
      <td>380.15</td>
      <td>23.23258</td>
      <td>108.85</td>
      <td>4.699116</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3987 rows × 58 columns</p>
</div>




```python
from sklearn.linear_model import ElasticNet
lr = ElasticNet(.1, 0)
lr.fit(X_train, y_train)
```

    C:\Users\Allen\anaconda3\lib\site-packages\sklearn\utils\validation.py:71: FutureWarning: Pass l1_ratio=0 as keyword args. From version 0.25 passing these as positional arguments will result in an error
      FutureWarning)
    C:\Users\Allen\anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:531: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8504.175821644329, tolerance: 4.811267459974919
      positive)





    ElasticNet(alpha=0.1, l1_ratio=0)




```python
predictions = lr.predict(X_test)
```

Below I create a dataframe with the coefficient data. I create a column 'sign' that is meant to record whether or not the variable is positive or negative. I also create a 'coefficient_abs' column which transforms the variable into its absolute value. These columns will be useful for visualizing and comparing all of the features based on their impact and usefulness to the model.


```python
coefficient_data = pd.DataFrame({'feature' : X_train.columns, 'coefficient' : lr.coef_})

coefficient_data = coefficient_data.sort_values('coefficient', ascending = False)

coefficient_data['sign'] = coefficient_data['coefficient'].apply(lambda n: 'positive' if n > 0 else 'negative')

coefficient_data['coefficient_abs'] = coefficient_data['coefficient'].apply(lambda n: np.abs(n))

coefficient_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
      <th>sign</th>
      <th>coefficient_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>default_1.0</td>
      <td>0.362657</td>
      <td>positive</td>
      <td>0.362657</td>
    </tr>
    <tr>
      <th>14</th>
      <td>log_carvalue</td>
      <td>0.204723</td>
      <td>positive</td>
      <td>0.204723</td>
    </tr>
    <tr>
      <th>0</th>
      <td>debtinc</td>
      <td>0.202457</td>
      <td>positive</td>
      <td>0.202457</td>
    </tr>
    <tr>
      <th>15</th>
      <td>othdebt</td>
      <td>0.126545</td>
      <td>positive</td>
      <td>0.126545</td>
    </tr>
    <tr>
      <th>20</th>
      <td>inccat_3.0</td>
      <td>0.123089</td>
      <td>positive</td>
      <td>0.123089</td>
    </tr>
    <tr>
      <th>19</th>
      <td>inccat_2.0</td>
      <td>0.094148</td>
      <td>positive</td>
      <td>0.094148</td>
    </tr>
    <tr>
      <th>10</th>
      <td>log_employ</td>
      <td>0.083809</td>
      <td>positive</td>
      <td>0.083809</td>
    </tr>
    <tr>
      <th>51</th>
      <td>jobsat_5.0</td>
      <td>0.076830</td>
      <td>positive</td>
      <td>0.076830</td>
    </tr>
    <tr>
      <th>54</th>
      <td>empcat_3.0</td>
      <td>0.076712</td>
      <td>positive</td>
      <td>0.076712</td>
    </tr>
    <tr>
      <th>30</th>
      <td>agecat_6.0</td>
      <td>0.067036</td>
      <td>positive</td>
      <td>0.067036</td>
    </tr>
    <tr>
      <th>25</th>
      <td>carcatvalue_3.0</td>
      <td>0.062675</td>
      <td>positive</td>
      <td>0.062675</td>
    </tr>
    <tr>
      <th>21</th>
      <td>inccat_4.0</td>
      <td>0.052980</td>
      <td>positive</td>
      <td>0.052980</td>
    </tr>
    <tr>
      <th>45</th>
      <td>card2tenurecat_4.0</td>
      <td>0.049983</td>
      <td>positive</td>
      <td>0.049983</td>
    </tr>
    <tr>
      <th>55</th>
      <td>empcat_4.0</td>
      <td>0.049716</td>
      <td>positive</td>
      <td>0.049716</td>
    </tr>
    <tr>
      <th>2</th>
      <td>income</td>
      <td>0.049407</td>
      <td>positive</td>
      <td>0.049407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>log_wireten</td>
      <td>0.027120</td>
      <td>positive</td>
      <td>0.027120</td>
    </tr>
    <tr>
      <th>50</th>
      <td>jobsat_4.0</td>
      <td>0.026380</td>
      <td>positive</td>
      <td>0.026380</td>
    </tr>
    <tr>
      <th>24</th>
      <td>carcatvalue_2.0</td>
      <td>0.025407</td>
      <td>positive</td>
      <td>0.025407</td>
    </tr>
    <tr>
      <th>37</th>
      <td>cardtenurecat_1.0</td>
      <td>0.018748</td>
      <td>positive</td>
      <td>0.018748</td>
    </tr>
    <tr>
      <th>28</th>
      <td>agecat_4.0</td>
      <td>0.018213</td>
      <td>positive</td>
      <td>0.018213</td>
    </tr>
    <tr>
      <th>35</th>
      <td>retire_0.0</td>
      <td>0.016799</td>
      <td>positive</td>
      <td>0.016799</td>
    </tr>
    <tr>
      <th>42</th>
      <td>card2tenurecat_1.0</td>
      <td>0.016073</td>
      <td>positive</td>
      <td>0.016073</td>
    </tr>
    <tr>
      <th>41</th>
      <td>cardtenurecat_5.0</td>
      <td>0.015285</td>
      <td>positive</td>
      <td>0.015285</td>
    </tr>
    <tr>
      <th>32</th>
      <td>carown_1.0</td>
      <td>0.011392</td>
      <td>positive</td>
      <td>0.011392</td>
    </tr>
    <tr>
      <th>38</th>
      <td>cardtenurecat_2.0</td>
      <td>0.009653</td>
      <td>positive</td>
      <td>0.009653</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cardspent</td>
      <td>0.000938</td>
      <td>positive</td>
      <td>0.000938</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cardtenure</td>
      <td>0.000441</td>
      <td>positive</td>
      <td>0.000441</td>
    </tr>
    <tr>
      <th>1</th>
      <td>card2spent</td>
      <td>0.000240</td>
      <td>positive</td>
      <td>0.000240</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tollten</td>
      <td>0.000107</td>
      <td>positive</td>
      <td>0.000107</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wireten</td>
      <td>-0.000143</td>
      <td>negative</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>12</th>
      <td>employ</td>
      <td>-0.004097</td>
      <td>negative</td>
      <td>0.004097</td>
    </tr>
    <tr>
      <th>39</th>
      <td>cardtenurecat_3.0</td>
      <td>-0.010948</td>
      <td>negative</td>
      <td>0.010948</td>
    </tr>
    <tr>
      <th>31</th>
      <td>carown_0.0</td>
      <td>-0.011392</td>
      <td>negative</td>
      <td>0.011392</td>
    </tr>
    <tr>
      <th>47</th>
      <td>jobsat_1.0</td>
      <td>-0.013124</td>
      <td>negative</td>
      <td>0.013124</td>
    </tr>
    <tr>
      <th>36</th>
      <td>retire_1.0</td>
      <td>-0.016799</td>
      <td>negative</td>
      <td>0.016799</td>
    </tr>
    <tr>
      <th>43</th>
      <td>card2tenurecat_2.0</td>
      <td>-0.016905</td>
      <td>negative</td>
      <td>0.016905</td>
    </tr>
    <tr>
      <th>44</th>
      <td>card2tenurecat_3.0</td>
      <td>-0.020578</td>
      <td>negative</td>
      <td>0.020578</td>
    </tr>
    <tr>
      <th>26</th>
      <td>agecat_2.0</td>
      <td>-0.020901</td>
      <td>negative</td>
      <td>0.020901</td>
    </tr>
    <tr>
      <th>27</th>
      <td>agecat_3.0</td>
      <td>-0.022236</td>
      <td>negative</td>
      <td>0.022236</td>
    </tr>
    <tr>
      <th>3</th>
      <td>log_card2spent</td>
      <td>-0.026547</td>
      <td>negative</td>
      <td>0.026547</td>
    </tr>
    <tr>
      <th>9</th>
      <td>log_tollten</td>
      <td>-0.027404</td>
      <td>negative</td>
      <td>0.027404</td>
    </tr>
    <tr>
      <th>56</th>
      <td>empcat_5.0</td>
      <td>-0.027905</td>
      <td>negative</td>
      <td>0.027905</td>
    </tr>
    <tr>
      <th>46</th>
      <td>card2tenurecat_5.0</td>
      <td>-0.028574</td>
      <td>negative</td>
      <td>0.028574</td>
    </tr>
    <tr>
      <th>7</th>
      <td>carvalue</td>
      <td>-0.031053</td>
      <td>negative</td>
      <td>0.031053</td>
    </tr>
    <tr>
      <th>40</th>
      <td>cardtenurecat_4.0</td>
      <td>-0.032739</td>
      <td>negative</td>
      <td>0.032739</td>
    </tr>
    <tr>
      <th>48</th>
      <td>jobsat_2.0</td>
      <td>-0.040003</td>
      <td>negative</td>
      <td>0.040003</td>
    </tr>
    <tr>
      <th>52</th>
      <td>empcat_1.0</td>
      <td>-0.041940</td>
      <td>negative</td>
      <td>0.041940</td>
    </tr>
    <tr>
      <th>29</th>
      <td>agecat_5.0</td>
      <td>-0.042112</td>
      <td>negative</td>
      <td>0.042112</td>
    </tr>
    <tr>
      <th>49</th>
      <td>jobsat_3.0</td>
      <td>-0.050082</td>
      <td>negative</td>
      <td>0.050082</td>
    </tr>
    <tr>
      <th>53</th>
      <td>empcat_2.0</td>
      <td>-0.056583</td>
      <td>negative</td>
      <td>0.056583</td>
    </tr>
    <tr>
      <th>18</th>
      <td>inccat_1.0</td>
      <td>-0.069737</td>
      <td>negative</td>
      <td>0.069737</td>
    </tr>
    <tr>
      <th>23</th>
      <td>carcatvalue_1.0</td>
      <td>-0.088082</td>
      <td>negative</td>
      <td>0.088082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>log_income</td>
      <td>-0.145739</td>
      <td>negative</td>
      <td>0.145739</td>
    </tr>
    <tr>
      <th>22</th>
      <td>inccat_5.0</td>
      <td>-0.200480</td>
      <td>negative</td>
      <td>0.200480</td>
    </tr>
    <tr>
      <th>17</th>
      <td>log_cardspent</td>
      <td>-0.205127</td>
      <td>negative</td>
      <td>0.205127</td>
    </tr>
    <tr>
      <th>33</th>
      <td>default_0.0</td>
      <td>-0.362657</td>
      <td>negative</td>
      <td>0.362657</td>
    </tr>
    <tr>
      <th>11</th>
      <td>log_othdebt</td>
      <td>-1.202853</td>
      <td>negative</td>
      <td>1.202853</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefficient_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
      <th>coefficient_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>57.000000</td>
      <td>57.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.016617</td>
      <td>0.082216</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.188638</td>
      <td>0.170250</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.202853</td>
      <td>0.000107</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.028574</td>
      <td>0.016799</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000107</td>
      <td>0.031053</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.049407</td>
      <td>0.076712</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.362657</td>
      <td>1.202853</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the coefficient data I decide that using the 75% marker on the magnitude makes the most sense and filter at coefficients that don't have a beta value larger than .07. I then visualize these 'impactful coefficients' below.


```python
impactful_coefs = coefficient_data[coefficient_data['coefficient_abs'] > .07]
non_impactful_coefs = coefficient_data[coefficient_data['coefficient_abs'] < .07]
```


```python
impactful_coefs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
      <th>sign</th>
      <th>coefficient_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>default_1.0</td>
      <td>0.362657</td>
      <td>positive</td>
      <td>0.362657</td>
    </tr>
    <tr>
      <th>14</th>
      <td>log_carvalue</td>
      <td>0.204723</td>
      <td>positive</td>
      <td>0.204723</td>
    </tr>
    <tr>
      <th>0</th>
      <td>debtinc</td>
      <td>0.202457</td>
      <td>positive</td>
      <td>0.202457</td>
    </tr>
    <tr>
      <th>15</th>
      <td>othdebt</td>
      <td>0.126545</td>
      <td>positive</td>
      <td>0.126545</td>
    </tr>
    <tr>
      <th>20</th>
      <td>inccat_3.0</td>
      <td>0.123089</td>
      <td>positive</td>
      <td>0.123089</td>
    </tr>
    <tr>
      <th>19</th>
      <td>inccat_2.0</td>
      <td>0.094148</td>
      <td>positive</td>
      <td>0.094148</td>
    </tr>
    <tr>
      <th>10</th>
      <td>log_employ</td>
      <td>0.083809</td>
      <td>positive</td>
      <td>0.083809</td>
    </tr>
    <tr>
      <th>51</th>
      <td>jobsat_5.0</td>
      <td>0.076830</td>
      <td>positive</td>
      <td>0.076830</td>
    </tr>
    <tr>
      <th>54</th>
      <td>empcat_3.0</td>
      <td>0.076712</td>
      <td>positive</td>
      <td>0.076712</td>
    </tr>
    <tr>
      <th>23</th>
      <td>carcatvalue_1.0</td>
      <td>-0.088082</td>
      <td>negative</td>
      <td>0.088082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>log_income</td>
      <td>-0.145739</td>
      <td>negative</td>
      <td>0.145739</td>
    </tr>
    <tr>
      <th>22</th>
      <td>inccat_5.0</td>
      <td>-0.200480</td>
      <td>negative</td>
      <td>0.200480</td>
    </tr>
    <tr>
      <th>17</th>
      <td>log_cardspent</td>
      <td>-0.205127</td>
      <td>negative</td>
      <td>0.205127</td>
    </tr>
    <tr>
      <th>33</th>
      <td>default_0.0</td>
      <td>-0.362657</td>
      <td>negative</td>
      <td>0.362657</td>
    </tr>
    <tr>
      <th>11</th>
      <td>log_othdebt</td>
      <td>-1.202853</td>
      <td>negative</td>
      <td>1.202853</td>
    </tr>
  </tbody>
</table>
</div>




```python
impactful_coefs = impactful_coefs.sort_values('coefficient_abs', ascending = False)
```


```python
fig, ax = plt.subplots(figsize = (10,5))
plot = sns.barplot(data = impactful_coefs.head(15), x = 'feature', y = 'coefficient_abs', hue = 'sign', hue_order = ['positive', 'negative'])
plt.title('Feature Weights')
ticks = plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
```


![png](output_68_0.png)



```python
non_impactful_coefs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
      <th>sign</th>
      <th>coefficient_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>agecat_6.0</td>
      <td>0.067036</td>
      <td>positive</td>
      <td>0.067036</td>
    </tr>
    <tr>
      <th>25</th>
      <td>carcatvalue_3.0</td>
      <td>0.062675</td>
      <td>positive</td>
      <td>0.062675</td>
    </tr>
    <tr>
      <th>21</th>
      <td>inccat_4.0</td>
      <td>0.052980</td>
      <td>positive</td>
      <td>0.052980</td>
    </tr>
    <tr>
      <th>45</th>
      <td>card2tenurecat_4.0</td>
      <td>0.049983</td>
      <td>positive</td>
      <td>0.049983</td>
    </tr>
    <tr>
      <th>55</th>
      <td>empcat_4.0</td>
      <td>0.049716</td>
      <td>positive</td>
      <td>0.049716</td>
    </tr>
    <tr>
      <th>2</th>
      <td>income</td>
      <td>0.049407</td>
      <td>positive</td>
      <td>0.049407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>log_wireten</td>
      <td>0.027120</td>
      <td>positive</td>
      <td>0.027120</td>
    </tr>
    <tr>
      <th>50</th>
      <td>jobsat_4.0</td>
      <td>0.026380</td>
      <td>positive</td>
      <td>0.026380</td>
    </tr>
    <tr>
      <th>24</th>
      <td>carcatvalue_2.0</td>
      <td>0.025407</td>
      <td>positive</td>
      <td>0.025407</td>
    </tr>
    <tr>
      <th>37</th>
      <td>cardtenurecat_1.0</td>
      <td>0.018748</td>
      <td>positive</td>
      <td>0.018748</td>
    </tr>
    <tr>
      <th>28</th>
      <td>agecat_4.0</td>
      <td>0.018213</td>
      <td>positive</td>
      <td>0.018213</td>
    </tr>
    <tr>
      <th>35</th>
      <td>retire_0.0</td>
      <td>0.016799</td>
      <td>positive</td>
      <td>0.016799</td>
    </tr>
    <tr>
      <th>42</th>
      <td>card2tenurecat_1.0</td>
      <td>0.016073</td>
      <td>positive</td>
      <td>0.016073</td>
    </tr>
    <tr>
      <th>41</th>
      <td>cardtenurecat_5.0</td>
      <td>0.015285</td>
      <td>positive</td>
      <td>0.015285</td>
    </tr>
    <tr>
      <th>32</th>
      <td>carown_1.0</td>
      <td>0.011392</td>
      <td>positive</td>
      <td>0.011392</td>
    </tr>
    <tr>
      <th>38</th>
      <td>cardtenurecat_2.0</td>
      <td>0.009653</td>
      <td>positive</td>
      <td>0.009653</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cardspent</td>
      <td>0.000938</td>
      <td>positive</td>
      <td>0.000938</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cardtenure</td>
      <td>0.000441</td>
      <td>positive</td>
      <td>0.000441</td>
    </tr>
    <tr>
      <th>1</th>
      <td>card2spent</td>
      <td>0.000240</td>
      <td>positive</td>
      <td>0.000240</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tollten</td>
      <td>0.000107</td>
      <td>positive</td>
      <td>0.000107</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wireten</td>
      <td>-0.000143</td>
      <td>negative</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>12</th>
      <td>employ</td>
      <td>-0.004097</td>
      <td>negative</td>
      <td>0.004097</td>
    </tr>
    <tr>
      <th>39</th>
      <td>cardtenurecat_3.0</td>
      <td>-0.010948</td>
      <td>negative</td>
      <td>0.010948</td>
    </tr>
    <tr>
      <th>31</th>
      <td>carown_0.0</td>
      <td>-0.011392</td>
      <td>negative</td>
      <td>0.011392</td>
    </tr>
    <tr>
      <th>47</th>
      <td>jobsat_1.0</td>
      <td>-0.013124</td>
      <td>negative</td>
      <td>0.013124</td>
    </tr>
    <tr>
      <th>36</th>
      <td>retire_1.0</td>
      <td>-0.016799</td>
      <td>negative</td>
      <td>0.016799</td>
    </tr>
    <tr>
      <th>43</th>
      <td>card2tenurecat_2.0</td>
      <td>-0.016905</td>
      <td>negative</td>
      <td>0.016905</td>
    </tr>
    <tr>
      <th>44</th>
      <td>card2tenurecat_3.0</td>
      <td>-0.020578</td>
      <td>negative</td>
      <td>0.020578</td>
    </tr>
    <tr>
      <th>26</th>
      <td>agecat_2.0</td>
      <td>-0.020901</td>
      <td>negative</td>
      <td>0.020901</td>
    </tr>
    <tr>
      <th>27</th>
      <td>agecat_3.0</td>
      <td>-0.022236</td>
      <td>negative</td>
      <td>0.022236</td>
    </tr>
    <tr>
      <th>3</th>
      <td>log_card2spent</td>
      <td>-0.026547</td>
      <td>negative</td>
      <td>0.026547</td>
    </tr>
    <tr>
      <th>9</th>
      <td>log_tollten</td>
      <td>-0.027404</td>
      <td>negative</td>
      <td>0.027404</td>
    </tr>
    <tr>
      <th>56</th>
      <td>empcat_5.0</td>
      <td>-0.027905</td>
      <td>negative</td>
      <td>0.027905</td>
    </tr>
    <tr>
      <th>46</th>
      <td>card2tenurecat_5.0</td>
      <td>-0.028574</td>
      <td>negative</td>
      <td>0.028574</td>
    </tr>
    <tr>
      <th>7</th>
      <td>carvalue</td>
      <td>-0.031053</td>
      <td>negative</td>
      <td>0.031053</td>
    </tr>
    <tr>
      <th>40</th>
      <td>cardtenurecat_4.0</td>
      <td>-0.032739</td>
      <td>negative</td>
      <td>0.032739</td>
    </tr>
    <tr>
      <th>48</th>
      <td>jobsat_2.0</td>
      <td>-0.040003</td>
      <td>negative</td>
      <td>0.040003</td>
    </tr>
    <tr>
      <th>52</th>
      <td>empcat_1.0</td>
      <td>-0.041940</td>
      <td>negative</td>
      <td>0.041940</td>
    </tr>
    <tr>
      <th>29</th>
      <td>agecat_5.0</td>
      <td>-0.042112</td>
      <td>negative</td>
      <td>0.042112</td>
    </tr>
    <tr>
      <th>49</th>
      <td>jobsat_3.0</td>
      <td>-0.050082</td>
      <td>negative</td>
      <td>0.050082</td>
    </tr>
    <tr>
      <th>53</th>
      <td>empcat_2.0</td>
      <td>-0.056583</td>
      <td>negative</td>
      <td>0.056583</td>
    </tr>
    <tr>
      <th>18</th>
      <td>inccat_1.0</td>
      <td>-0.069737</td>
      <td>negative</td>
      <td>0.069737</td>
    </tr>
  </tbody>
</table>
</div>




```python
metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rmse</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num_only</td>
      <td>2.275009</td>
      <td>5.175667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nums_and_cats</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>2</th>
      <td>all_vars_cv_1</td>
      <td>2.215407</td>
      <td>4.908026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>num_only_log</td>
      <td>2.335687</td>
      <td>5.455432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>logs_and_cats</td>
      <td>2.252368</td>
      <td>5.073160</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nums_logs_and_cats</td>
      <td>2.085990</td>
      <td>4.351353</td>
    </tr>
    <tr>
      <th>6</th>
      <td>original_logs_cats_hyper_tuned</td>
      <td>2.101103</td>
      <td>4.414633</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc.stop()
# metrics.to_csv('metrics.csv')
```

# Analysis and Conclusion

The model results mostly seem intuitive. Other debt, whether or not you've ever defaulted, and amount spent on card last month are sensible indicators for credit card debt. On the lower end, it seems to be interesting that variables related to employment and job security indicate more credit debt.

There's likely alot more hyper parameter tuning that can be done to develop a model with less error. Along with that features excluded in my analysis could be included and features determined to be useless could be excluded in a new model. A gradient boosting model could also be run with regression trees to see if they are generally more efffective.
