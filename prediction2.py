import json
import os
import re
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import paramiko
from datetime import datetime, timedelta, timezone
import pika
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import load_model
import tensorflow as tf
from config.database import db
from bson import ObjectId
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from model.projectFile import ProjectFile


global forecast_duration, selected_inputs, parameters, result_directory, prediction_file_path, model_id, prediction_id, project_id, user_id

#RABBITMQ PUBLISHER CLASS****************************************************************************************************START
class Publisher:
	def __init__ (self, config):
		self.config = config

	def publish(self, routing_key, message):
		connection = self.create_connection()
		channel = connection.channel()
		channel.basic_publish(
			exchange=routing_key, 
            routing_key="",
			body=message
		)
		print(f" [x] Sent message {message} for {routing_key}")
	def create_connection(self):
		param = pika.ConnectionParameters(
			host=self.config["host"], 
            port=self.config["port"]
		)
		return pika.BlockingConnection(param)
rabbitmq_config = {"host": "localhost", "port": 5672}
publisher = Publisher(rabbitmq_config)
#RABBITMQ PUBLISHER CLASS******************************************************************************************************END

logging.basicConfig(filename='your_log_file.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_log_path(user_id, project_id):
    user_log_folder = f"/home/user/symbiont_aiml/{user_id}/{project_id}/log"
    os.makedirs(user_log_folder, exist_ok=True)
    log_file_path = os.path.join(user_log_folder, "predicting_session.log")
    return log_file_path

def save_predict_img(user_id, project_id):
    predict_results_dir = f"/home/user/symbiont_aiml/{user_id}/{project_id}/prediction_result/"
    os.makedirs(predict_results_dir, exist_ok=True)
    predict_results_path = os.path.join(predict_results_dir, "prediction_plot.png")
    plt.savefig(predict_results_path)
    return predict_results_path

def setGlobal(forecast_duration, selected_inputs, parameters, result_directory, prediction_file_path, model_id, prediction_id, project_id, user_id):
    log_file_path = get_user_log_path(user_id, project_id)
    logger = logging.getLogger(user_id)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    selected_inputs = selected_inputs
    forecast_duration = forecast_duration
    logger.info("Successfully access selected inputs")
    parameters["Sequence Length"] = int(parameters["Sequence Length"])
    logger.info("Successfully converted Sequence Length", parameters["Sequence Length"], type(parameters["Sequence Length"]))
    #parameters["Units"] = int(parameters["Units"])
    #logger.info("Successfully converted Units")
    #parameters["Batch Size"] = int(parameters["Batch Size"])
    #logger.info("Successfully converted Batch Size")
    #parameters["Epochs"] = int(parameters["Epochs"])
    #logger.info("Successfully converted Epochs")
    #parameters["Learning Rate"] = float(parameters["Learning Rate"])
    #logger.info("Successfully converted Learning Rate")
    #parameters["Training Ratio"] = float(parameters["Training Ratio"])
    #logger.info("Successfully converted Training Ratio")
    parameters = parameters
    logger.info("Successfully access parameters")

    prediction_file_path = prediction_file_path
    model_file_path = f"/home/user/symbiont_aiml/{user_id}/{project_id}/model/"
    model_id = model_id


    return logger

def get_bangkok_time():
    utc_now = datetime.utcnow()
    bangkok_time = utc_now.replace(tzinfo=timezone.utc) + timedelta(hours=7)
    return bangkok_time

def load_file(prediction_file_path, user_id):
    try:
        local_path = '/home/user/symbiont_aiml'
        logger.info(f"Trying to open file: {prediction_file_path}")
        with open(f"/home/user/symbiont_aiml{prediction_file_path}",'r') as f:
            # f.prefetch()
            logger.info("File opened successfully.")
            df = pd.read_csv(f)
            logger.info("CSV file loaded successfully.")
            return df
    except paramiko.AuthenticationException as auth_error:
        logger.error(f"Authentication error: {str(auth_error)}")
        log_message = ("Error Authentication error during prediction")
        publisher.publish(user_id, log_message)
        return None
    except Exception as e:
        logger.error(f"Error in load_file: {str(e)}")
        log_message = ("Error in load_file during prediction")
        publisher.publish(user_id, log_message)
        return None
    
def load_trained_model(user_id, project_id):
    try:
        local_path = '/home/user/symbiont_aiml'
        model_file_path = f"/home/user/symbiont_aiml/{user_id}/{project_id}/model/"
        #model1 = load_model(f"/home/user/symbiont_aiml/{user_id}/{project_id}/model/")
        if os.path.exists(model_file_path):
            model1 = load_model(model_file_path)
            logger.info("Model loaded successfully.")
            logger.info(model1.summary())
            return model1
    except FileNotFoundError:
        logger.error(f"File not found: {model_file_path}")
        log_message = ("Error File not found during prediction")
        publisher.publish(user_id, log_message)
        return None
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        log_message = ("Error in load_model during prediction")
        publisher.publish(user_id, log_message)
        return None

def load_classification_model(user_id, project_id):
    try:
        local_path = '/home/user/symbiont_aiml'
        model_file_path = f"/home/user/symbiont_aiml/{user_id}/{project_id}/model_class/"
        if os.path.exists(model_file_path):
            model1 = load_model(model_file_path)
            logger.info("Model loaded successfully.")
            logger.info(model1.summary())
            return model1
    except FileNotFoundError:
        logger.error(f"File not found: {model_file_path}")
        log_message = ("Error File not found during prediction")
        publisher.publish(user_id, log_message)
        return None
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        log_message = ("Error in load_model during prediction")
        publisher.publish(user_id, log_message)
        return None

def attribute(df, selected_inputs, model_id):
    try:
        # logger.info(selected_inputs)
        logger.info(f"Attribute select from DF:\n {df}")
        temp = df[selected_inputs]
        logger.info(temp)
        binary_columns = []
        numerical_columns = []
        for col in temp:
            unique_values = temp[col].unique()
            if len(unique_values)<=2 and set(unique_values)<={0, 1}:
                binary_columns.append(col)
            else:
                numerical_columns.append(col)
        binary_data = df[binary_columns].values
        numerical_data = df[numerical_columns].values
        logger.info(f'BINARY DATA: {binary_columns}')
        logger.info(f'NUMERICAL DATA: {numerical_columns}')
        # # scaler = MinMaxScaler()
        # # numerical_data_scaled = scaler.fit_transform(df[numerical_columns]) if numerical_columns else np.array([])
        temp2 = np.concatenate((numerical_data, binary_data), axis = 1)
        selected_inputs = numerical_columns + binary_columns
        temp = df[selected_inputs]
        # logger.info(temp2)
        # return temp, temp2, selected_inputs, numerical_columns, binary_columns
        return temp, temp2, numerical_columns, binary_columns, selected_inputs
    except KeyError as e:
        logger.error(f"Error accessing selected inputs: {str(e)}")
        log_message = ("Error accessing selected inputs during prediction")
        publisher.publish(user_id, log_message)
        return None

def start_prediction(forecast_duration, selected_inputs, parameters, result_directory, predicting_file_path, model_id, prediction_id, project_id, user_id):
    try:
        db["predictions"].update_one({"_id": ObjectId(prediction_id)},{"$set": {"status": "In Progress","start_timestamp": get_bangkok_time() }})
        logger = setGlobal(forecast_duration, selected_inputs, parameters, result_directory, predicting_file_path, model_id, prediction_id, project_id, user_id)
        if not all([forecast_duration, parameters, selected_inputs, predicting_file_path]):
            log_message = ("Error Missing required input data during prediction")
            publisher.publish(user_id, log_message)
            raise ValueError("Missing required input data.")
        logger.info("Prediction process started")
        df = load_file(predicting_file_path, user_id)
        if df is None:
            log_message = ("Error Failed to load file during prediction")
            publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to load file.")
        #df.index = df.pop('index')
        df.index = df.pop('Timestamp')
        #df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
        logger.info(f"Loaded DataFrame:\n{df}")
        # temp, temp2, selected_inputs, numerical_columns, binary_columns = attribute(df, selected_inputs, user_id)
        temp, temp2, numerical_columns, binary_columns, selected_inputs = attribute(df, selected_inputs, user_id)
        if temp is None:
            log_message = ("Error Failed to process selected inputs during prediction")
            publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to process selected inputs.")
        
        
        model = load_trained_model(user_id, project_id)
        logger.info("Model1 loaded successfully to model.")
        logger.info(model.summary())
        logger.info(temp)
        input_sequence = []
        forecast_duration = forecast_duration * 24
        sequences_length = parameters["Sequence Length"]
        print(parameters["Sequence Length"])
        # input_sequence = temp[-sequences_length:].to_numpy()
        input_sequence = temp2[-sequences_length:]
        input_sequence = input_sequence.reshape((1, sequences_length, temp.shape[1]))
        logger.info(f"Initialize input sequence:\n{input_sequence}")
        forecasted_values = []
        predictions = []
        last_timestamp = temp.index[-1]
        date_range = pd.date_range(start=last_timestamp, periods=forecast_duration + 1, freq='1H')
        formatted_dates = [date.strftime('%Y-%m-%dT%H:%M:%S.%fZ') for date in date_range]
        forecasted_df = pd.DataFrame(index=formatted_dates, columns=temp.columns)
        forecasted_df = forecasted_df.iloc[1:]
        forecasted_df.index.name = 'Timestamp'
        print(f"Forecasted duration: {forecast_duration}")
        logger.info(f"Loaded forecasted_df:\n{forecasted_df}")
        logger.info(f"Last timestamp:\n{last_timestamp}")

        #forecasted_df.index = pd.to_datetime(forecasted_df.index, format='%Y-%m-%dT%H:%M:%S.%fZ')
        #logger.info(f"Forecasted Values with Timestamp:\n{forecasted_df}")

        for step in range(forecast_duration):
            predictions = model.predict(input_sequence)
            predictions = np.maximum(predictions, 0)
            predictions[predictions < 0.0001] = 0
            logger.info(predictions[0, :])
            next_step_prediciton = predictions[0, :]
            forecasted_df.iloc[step] = next_step_prediciton
            input_sequence = np.concatenate((input_sequence[:, 1:, :], next_step_prediciton.reshape(1, 1, temp.shape[1])), axis=1)

        forecasted_df = forecasted_df.copy()
        # # min_values = df[numerical_columns].min()
        # # max_values = df[numerical_columns].max()
        # # for col in numerical_columns:
        # #     min_val = min_values[col]
        # #     max_val = max_values[col]
        # #     reverted_df[col] = reverted_df[col] * (max_val - min_val) + min_val
        forecasted_df[binary_columns] = forecasted_df[binary_columns].applymap(lambda x: 1 if x >= 0.5 else 0)
        
        try:
            save_predict_img(user_id, project_id)
            plt.figure(figsize=(10, 6))
            for column in forecasted_df.columns:
                plt.plot(forecasted_df.index, forecasted_df[column], label=column)
            plt.title('Forecasted Values')
            plt.xlabel('Timestamp')
            plt.ylabel('Sensor Values')
            plt.legend()
            plt.grid(True)
            predict_results_dir = f"/home/user/symbiont_aiml/{user_id}/{project_id}/prediction_result/"
            os.makedirs(predict_results_dir, exist_ok=True)
            predict_results_path = os.path.join(predict_results_dir, "prediction_plot.png")
            plt.savefig(predict_results_path)
            predict_results_path = os.path.join(predict_results_dir, "forecasted_values.csv")
            forecasted_df.to_csv(predict_results_path)
        
        except Exception as e:
            logger.error(f"Error in forecasting process: {str(e)}")
            log_message = ("Error in forecasting process during prediction")
            publisher.publish(user_id, log_message)

        try:
            forecasting_result_file_check = db["project files"].find_one({"project_id": project_id, "filename": "prediction_plot.png"})
            if not forecasting_result_file_check:
                projectFile = ProjectFile(**{
                "filename": "prediction_plot.png",
                "file_type": ".png",
                "asset_type": "forecasted result",
                "file_size": None,
                "file_path": f"/{user_id}/{project_id}/prediction_result/prediction_plot.png",
                "created_timestamp": get_bangkok_time(),
                "project_id": project_id
            })
                result =  db["project files"].insert_one(dict(projectFile))
            else:
                db["project files"].update_one({"project_id": project_id},{"$set": {"created_timestamp": get_bangkok_time()}})
        except Exception as e:
            logger.error(f"Error in saving prediction result image: {str(e)}")
            log_message = ("Error in saving prediction result image during prediction")
            publisher.publish(user_id, log_message)

        try:
            forecasting_result_file_check = db["project files"].find_one({"project_id": project_id, "filename": "forecasted_values.csv"})
            if not forecasting_result_file_check:
                projectFile = ProjectFile(**{
                "filename": "forecasted_values.csv",
                "file_type": "text/csv",
                "asset_type": "forecasted result",
                "file_size": None,
                "file_path": f"/{user_id}/{project_id}/prediction_result/forecasted_values.csv",
                "created_timestamp": get_bangkok_time(),
                "project_id": project_id
            })
                result = db["project files"].insert_one(dict(projectFile))
            else:
                db["project files"].update_one({"project_id": project_id},{"$set": {"created_timestamp": get_bangkok_time()}})
        except Exception as e:
            logger.error(f"Error in saving prediction result: {str(e)}")
            log_message = ("Error in saving prediction csv result during prediction")
            publisher.publish(user_id, log_message)

        log_message = f"Prediction completed for prediction ID: {prediction_id}"
        publisher.publish(prediction_id, log_message)
        publisher.publish(user_id, log_message)
        db["predictions"].update_one({"_id": ObjectId(prediction_id)},{"$set": {"status": "Completed","end_timestamp": get_bangkok_time() }})
        #db["predictions"].update_one({"_id": ObjectId(model_id)},{"$set": {"progress_percentage": "100"}})

        start_classification(forecast_duration, selected_inputs, parameters, result_directory, predicting_file_path, model_id, prediction_id, project_id, user_id)

        return forecasted_values
    
    except Exception as e:
        logger.error(f"Error in prediciton process: {str(e)}")
        log_message = "Error in prediciton process"
        publisher.publish(user_id, log_message)
        publisher.publish(prediction_id, log_message)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def start_classification(forecast_duration, selected_inputs, parameters, result_directory, predicting_file_path, model_id, prediction_id, project_id, user_id):
    try:
        log_message = f"Classification started for prediction ID: {prediction_id}"
        publisher.publish(prediction_id, log_message)
        publisher.publish(user_id, log_message)
        new_data = pd.read_csv(f"/home/user/symbiont_aiml/{user_id}/{project_id}/prediction_result/forecasted_values.csv")
        new_data.index = new_data.pop('Timestamp')
        #new_data.index = pd.to_datetime(new_data.index, format='%Y-%m-%d %H:%M:%S.%f%z')

        model = load_classification_model(user_id, project_id)

        classify = model.predict(new_data)
        classify_labels = np.argmax(classify, axis=1)
        logger.info(f"Completed classification on new data")

        df_result = pd.DataFrame(new_data)
        df_result['class'] = classify_labels

        label_mapping = {0: 'critical', 1: 'normal', 2: 'warning'}
        df_result['class'] = df_result['class'].map(label_mapping)

        classify_results_dir = f"/home/user/symbiont_aiml/{user_id}/{project_id}/prediction_result/"
        os.makedirs(classify_results_dir, exist_ok=True)
        classify_results_path = os.path.join(classify_results_dir, "forecasted_values.csv")
        df_result.to_csv(classify_results_path)

        db["predictions"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Completed","end_timestamp": get_bangkok_time() }})
        db["predictions"].update_one({"_id": ObjectId(model_id)},{"$set": {"progress_percentage": "100"}})
        log_message = f"Classification completed for prediction ID: {prediction_id}"
        publisher.publish(prediction_id, log_message)
        publisher.publish(user_id, log_message)

        # Symbiont webhoook
        access_token = get_access()
        if access_token != None:
            query_influx(access_token)
        try:
            prediction = db["predictions"].find_one({"_id" : ObjectId(prediction_id)})
            regex_pattern = re.compile(re.escape(prediction["result_directory"]))
            file = db["project files"].find_one({"file_path" : {"$regex": regex_pattern}, "file_type": "text/csv"})
            webhook_url = f"https://{prediction['symbiont_interface_path']}/{prediction['symbiont_interface_tag']}"
            try:
                local_file_path = f"/home/user/symbiont_aiml{file['file_path']}"
                with open(local_file_path, "r") as local_file:
                    df = pd.read_csv(local_file)
                    label_mapping = {'critical': 0, 'normal': 2, 'warning': 1}
                    df['class'] = df['class'].map(label_mapping)
                    numeric_columns = df.select_dtypes(include=['float64']).columns
                    df[numeric_columns] = df[numeric_columns].round(1)
                    data = {"data": []}
                    if "Timestamp" in df.columns:
                        df = df.rename(columns={'Timestamp': 'ts'})
                    data["data"] = df.to_dict(orient='records')
                    print(data)
                    res = requests.post(webhook_url, verify=False, data=json.dumps(data,cls=NpEncoder), headers={'Content-Type':'application/json'})
                    print('POST TO SYMBIONT')
                    return res
            except Exception as e:
                raise e
        except Exception as e:
            raise e

    except Exception as e:
        logger.error(f"Error in classification process: {str(e)}")
        log_message = "Error in classification process"
        publisher.publish(user_id, log_message)


def get_access():
	get_access_token = "https://10.167.5.100:8090/symbiont-webapi/v1/auth/accesstoken"
	payload = {
    "Username": "admin",
    "Password": "ploegweg"
	}
	response = requests.post(get_access_token, json = payload, verify=False)
	if response.status_code != 200:
			return None
	try:
		json_payload = response.json()
		expires_in = json_payload["ExpiresIn"]
		username = json_payload["Username"]
		access_token = json_payload["AccessToken"]
		return access_token
	except Exception as e:
		print(f"get_access Exception occurred: {e}")
		return None

def query_influx(token):
	tag_id_list = {
		"discharge_pressure": "80f301a8-de5b-40de-b97c-357bc42a22f5",
		"pump_room_temp": "853a7a33-41c5-47ab-8a4c-d8034a30fb58",
		"batt_1_voltage": "ec25f105-3916-4870-b024-73d924df66d2",
		"batt_2_voltage": "e3afee28-badd-425c-8a00-ef0f8afd166d",
		"batt_1_failure": "440ba6e1-4724-4d5c-8a4e-f79b96a9d7f9",
		"batt_2_failure": "45f75ff1-b73d-4e48-aac4-e30e3c87fb4f",
		"batt_1_low": "93f9bb2e-da60-426b-8ee1-f23f36491052",
		"batt_2_low": "3dee6988-70d4-49ba-893b-1d64676d753c",
		"batt_1_current": "e522ca5b-eaa6-46df-a0c7-9acf5a3b7700",
		"batt_2_current": "81e3b50b-ad72-4358-a8a1-33f8f2758449",
		"charger_1_failure": "7638b888-95e9-47f3-8586-8659f7dac879",
		"charger_2_failure": "5df9f3ea-3011-42b7-a0c8-ca89ed71452d",
		"engine_running": "f3349a3f-22d1-4f14-94d7-c8b37dec45e5",
		"class": "2e4d7c26-ca84-4d7e-b0a5-1887f5daff14"
	}
	get_tag_url = 'https://10.167.5.100:8090/symbiont-webapi/v1/influx/query'

	for i in tag_id_list:
		query_str = f"delete from Historical where TagId='{tag_id_list[i]}'"
		payload = {
			"Id": "string",
			"QueryMode": 1,
			"QueryText": query_str
		}

		headers = {'Authorization': f'Bearer {token}'}
		response = requests.post(get_tag_url, json = payload, headers=headers, verify=False)

		try:
			json_payload = response.json()
			print("payload > ", json_payload)
		
		except Exception as e:	
			print(f"query_influx Exception occurred: {e}")


        


