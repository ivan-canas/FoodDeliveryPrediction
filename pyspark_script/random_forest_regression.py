from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from dotenv import load_dotenv
import os
import boto3
from pyspark.sql.functions import col, abs

load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')

os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_KEY
os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'

spark = SparkSession.builder.appName('RandomForestRegrsor').getOrCreate()

s3 = boto3.client('s3')
bucket_name = 'food-delivery-prediction'
s3_file_name = 'Food_Delivery_Times.csv'
local_file_name = 'data/local_delivery_file.csv'

def get_dataframe_from_csv(bucket_name, object_name, local_name):
    s3.download_file(bucket_name, object_name, local_name)
    delivery_df = spark.read.option('header', 'true').option('inferSchema', 'true').\
        csv(local_name)
    #delivery_df.show()
    return delivery_df

def split_data(delivery_df):
    training_df, test_df = delivery_df.randomSplit([0.75, 0.25])
    return training_df, test_df

def prepare_random_forest_regresor(training_df):
    # Change Strings to numeric
    traffic_index = StringIndexer(inputCol='Traffic_Level', outputCol='traffico_indexed')
    vehicle_index = StringIndexer(inputCol='Vehicle_Type', outputCol='vehicle_indexed')
    weather_index = StringIndexer(inputCol='Weather', outputCol='weather_indexed')

    training_df_transform = traffic_index.fit(training_df).transform(training_df)
    training_df_transform = vehicle_index.fit(training_df_transform).transform(training_df_transform)
    training_df_transform = weather_index.fit(training_df_transform).transform(training_df_transform)

    # Set a vector value with feature data
    assembler = VectorAssembler(
        inputCols=['Distance_km','traffico_indexed', 'vehicle_indexed', 'weather_indexed'],
        outputCol='forest_features'
    )

    final_df = assembler.transform(training_df_transform)
    #final_df.show()
    return final_df

def train_random_forest_regresor(training_df):
    training_df.show()
    rf = RandomForestRegressor(
        featuresCol='forest_features',
        labelCol='Delivery_Time_min',
        predictionCol='prediction_time',
        numTrees=50,
        maxDepth=5,
        seed=1
    )
    rf_model = rf.fit(training_df)
    return rf_model

def test_predictions_acc(test_df, rf_model):
    predictions_df = rf_model.transform(test_df)
    predictions_df = predictions_df.withColumn('prediction_real_diff',
     abs(col('prediction_time') - col('Delivery_Time_min')))
    predictions_df.show()

delivery_df = get_dataframe_from_csv(bucket_name, s3_file_name, local_file_name)
delivery_df = prepare_random_forest_regresor(delivery_df)
training_df, test_df = split_data(delivery_df)
rf_model = train_random_forest_regresor(training_df)
test_predictions_acc(test_df, rf_model)