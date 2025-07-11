# FoodDeliveryPrediction
Food Delivery Time Prediction ETL - with PySpark & Airflow
## Description
This project predicts food delivery times using PySpark and Random Forest Regression.
The workflow includes reading data from an AWS S3 bucket, cleaning and transforming it, saving processed data back to S3, and training a predictive model.
All tasks are organized and orchestrated with Apache Airflow.

<img width="1633" height="379" alt="Image" src="https://github.com/user-attachments/assets/afc93c5a-c82d-4b52-a2bd-ef2dd18b0450" />

<img width="1556" height="292" alt="Image" src="https://github.com/user-attachments/assets/d30b9de0-b0fb-46e4-89ca-ad110dd55128" />

## Features
Reading and writing data to AWS S3 using PySpark and boto3.

Data cleaning and transformation (handling categorical variables and nulls).

Training a Random Forest regression model.

Orchestrating the full pipeline with Airflow DAGs.

## Requirements
Python 3.8+

Apache Spark

Apache Airflow

AWS CLI configured

Libraries: pyspark, boto3, apache-airflow, python-dotenv

## Usage
Set up AWS credentials in a .env file.

Install dependencies with pip install -r requirements.txt.

Run Airflow (airflow scheduler and airflow webserver).

Trigger the DAG from the Airflow UI to process data and train the model.

## Future Improvements
Cross-validation and hyperparameter tuning.

Notifications and alerts in Airflow.

Deploying the trained model for real-time predictions.
