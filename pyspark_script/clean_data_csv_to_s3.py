import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from dotenv import load_dotenv
import boto3

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_KEY
os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'

base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "../data/local_delivery_file.csv")

spark = SparkSession.builder.appName("DeliveryTimeProcessing").getOrCreate()
s3 = boto3.client('s3')
bucket_name = 'food-delivery-prediction'
s3_file_name = 'Food_Delivery_Times.csv'
local_file_name = 'data/local_delivery_file.csv'


def create_dataframe_from_csv(bucket_name, object_name, local_name):
    s3.download_file(bucket_name, object_name, local_name)
    delivery_df = spark.read.option("header", "true").option("inferSchema", "true").\
        csv(local_name)
    #delivery_df.show()
    return delivery_df

def clean_dataframe_values(raw_df, local_name):
    # if na in distance, delivery, vehicle -> drop
    cleaned_df = raw_df.na.drop(subset=[
        'Distance_km','Delivery_Time_min','Vehicle_Type'])
    # if na in others fill with most common
    # Weather
    common_weather = cleaned_df.groupBy('Weather').count().\
        orderBy(col('count').desc()).first()['Weather']
    cleaned_df = cleaned_df.na.fill({'Weather': common_weather})
    # Traffic_Level
    common_traffic = cleaned_df.groupBy('Traffic_Level').count().\
        orderBy(col('count').desc()).first()['Traffic_Level']
    cleaned_df = cleaned_df.na.fill({'Traffic_Level': common_traffic})
    # Time_of_Day
    common_time = cleaned_df.groupBy('Time_of_Day').count().\
        orderBy(col('count').desc()).first()['Time_of_Day']
    cleaned_df = cleaned_df.na.fill({'Time_of_Day': common_time})
    # Preparation_Time_min
    common_prep = cleaned_df.groupBy('Preparation_Time_min').count().\
        orderBy(col('count').desc()).first()['Preparation_Time_min']
    cleaned_df = cleaned_df.na.fill({'Preparation_Time_min': common_prep})
    # Courier_Experience_yrs
    common_exp = cleaned_df.groupBy('Courier_Experience_yrs').count().\
        orderBy(col('count').desc()).first()['Courier_Experience_yrs']
    cleaned_df = cleaned_df.na.fill({'Courier_Experience_yrs': common_exp})

    # Check outliners with IQR in distance and delivery time
    # Distance
    dist_q1, dist_q3 = cleaned_df.approxQuantile('Distance_km', [0.25, 0.75], 0.1)
    dist_iqr = dist_q3 - dist_q1
    low_limit = dist_q1 - 1.5 * dist_iqr
    high_limit = dist_q3 + 1.5 * dist_iqr
    cleaned_df = cleaned_df.filter(
        (col('Distance_km') >= low_limit) & (col('Distance_km') <= high_limit))
    # Delivery time
    delivery_q1, delivery_q3 = cleaned_df.approxQuantile(
        'Delivery_Time_min', [0.25, 0.75], 0.1)
    delivery_iqr = delivery_q3 - delivery_q1
    low_limit = delivery_q1 - 1.5 * delivery_iqr
    high_limit = delivery_q3 + 1.5 * delivery_iqr
    #outliners = cleaned_df.filter(
    #    (col('Delivery_Time_min') < low_limit) | 
    #     (col('Delivery_Time_min') > high_limit))
    #outliners.show()
    cleaned_df = cleaned_df.filter(
        (col('Delivery_Time_min') >= low_limit) & 
         (col('Delivery_Time_min') <= high_limit))
    #cleaned_df.show()
    

def upload_csv_to_s3():
    with open(csv_path, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, s3_file_name)


delivery_df = create_dataframe_from_csv(bucket_name, s3_file_name, local_file_name)
clean_dataframe_values(delivery_df, local_file_name)
upload_csv_to_s3()








