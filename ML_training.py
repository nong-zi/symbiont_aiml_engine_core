import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import paramiko
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import tensorflow as tf
from config.database import db
from bson import ObjectId
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, classification_report
import logging
import pika

from model.projectFile import ProjectFile

global selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path, logger

logging.basicConfig(filename='your_log_file.log', level=logging.INFO)
logger = logging.getLogger(__name__)
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

def get_user_log_path(user_id, project_id):
    user_log_folder = f"/home/user/symbiont_aiml/{user_id}/{project_id}/log"
    os.makedirs(user_log_folder, exist_ok=True)
    log_file_path = os.path.join(user_log_folder, "training_session.log")
    return log_file_path

def save_train_img(user_id, project_id):
    train_results_dir = f"/home/user/symbiont_aiml/{user_id}/{project_id}/training_result/"
    os.makedirs(train_results_dir, exist_ok=True)
    train_results_path = os.path.join(train_results_dir, "loss_plot.png")
    plt.savefig(train_results_path)
    rmse_plot_path = os.path.join(train_results_dir, "rmse_plot.png")
    plt.savefig(rmse_plot_path)
    return train_results_path, rmse_plot_path

def setGlobal(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path):
    try:
        log_file_path = get_user_log_path(user_id, project_id)
        logger = logging.getLogger(user_id)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        selected_inputs = selected_inputs
        logger.info(f"Successfully access selected inputs {selected_inputs}")
        parameters["Sequence Length"] = int(parameters["Sequence Length"])
        logger.info("Successfully converted Sequence Length", parameters["Sequence Length"], type(parameters["Sequence Length"]))
        parameters["First Units"] = int(parameters["First Units"])
        parameters["Second Units"] = int(parameters["Second Units"])
        logger.info("Successfully converted Units")
        parameters["Batch Size"] = int(parameters["Batch Size"])
        logger.info("Successfully converted Batch Size")
        parameters["Epochs"] = int(parameters["Epochs"])
        logger.info("Successfully converted Epochs")
        parameters["Learning Rate"] = float(parameters["Learning Rate"])
        logger.info("Successfully converted Learning Rate")
        parameters["Optimizer"] = str(parameters["Optimizer"])
        logger.info("Successfully converted Optimizer")
        parameters["First Activation"] = str(parameters["First Activation"])
        parameters["Second Activation"] = str(parameters["Second Activation"])
        logger.info("Successfully converted Activation Function")
        parameters["Training Ratio"] = float(parameters["Training Ratio"])
        logger.info("Successfully converted Training Ratio")
        parameters = parameters
        logger.info("Successfully access parameters")

        file_path = file_path
        training_file_path = training_file_path
        model_id = model_id

        class_parameters["First Units"] = int(class_parameters["First Units"])
        class_parameters["Second Units"] = int(class_parameters["Second Units"])
        logger.info("Successfully converted Class_Units")
        class_parameters["Batch Size"] = int(class_parameters["Batch Size"])
        logger.info("Successfully converted Class_Batch Size")
        class_parameters["Epochs"] = int(class_parameters["Epochs"])
        logger.info("Successfully converted Class_Epochs")
        class_parameters["Learning Rate"] = float(class_parameters["Learning Rate"])
        logger.info("Successfully converted Class_Learning Rate")
        class_parameters["Optimizer"] = str(class_parameters["Optimizer"])
        logger.info("Successfully converted Class_Optimizer")
        class_parameters["First Activation"] = str(class_parameters["First Activation"])
        class_parameters["Second Activation"] = str(class_parameters["Second Activation"])
        logger.info("Successfully converted Class_Activation Function")
        class_parameters["Training Ratio"] = float(class_parameters["Training Ratio"])
        logger.info("Successfully converted Class_Training Ratio")
        class_training_file_path = class_training_file_path
        return logger
    except Exception as e:
        publisher.publish('Error setting global')

def get_bangkok_time():
    utc_now = datetime.utcnow()
    bangkok_time = utc_now.replace(tzinfo=timezone.utc) + timedelta(hours=7)
    return bangkok_time

def load_file(training_file_path, model_id):
    try:
        local_path = '/home/user/symbiont_aiml'
        logger.info(f"Trying to open file: {local_path}{training_file_path}")
        with open(f"{local_path}{training_file_path}",'r') as f:
            # f.prefetch()
            logger.info("File opened successfully.")
            df = pd.read_csv(f)
            logger.info("CSV file loaded successfully.")
            #NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
            if df.isna().any().any():
                na_rows = df.isna().any(axis=1)
                df = df[~na_rows]
                log_message = f"Cleaned Dataset by removing rows with null."
                publisher.publish(model_id, log_message)
            #NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
            return df
    except paramiko.AuthenticationException as auth_error:
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        logger.error(f"Authentication error: {str(auth_error)}")
        log_message = ("Error Authentication error during training")
        #publisher.publish(user_id, log_message)
        return None
    except Exception as e:
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        logger.error(f"Error in load_file: {str(e)}")
        log_message = ("Error in load_file during training")
        #publisher.publish(user_id, log_message)
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
        logger.info(f'BINARY DATA: {binary_data}')
        logger.info(f'NUMERICAL DATA: {numerical_data}')
        # # scaler = MinMaxScaler()
        # # numerical_data_scaled = scaler.fit_transform(df[numerical_columns]) if numerical_columns else np.array([])
        temp = np.concatenate((numerical_data, binary_data), axis = 1)
        selected_inputs = numerical_columns + binary_columns
        # logger.info(temp)
        return temp, selected_inputs
        # return temp
    except KeyError as e:
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        logger.error(f"Error accessing selected inputs: {str(e)}")
        log_message = ("Error accessing selected inputs during training")
        #publisher.publish(user_id, log_message)
        return None

def df_to_X_y(df, window_size, model_id):
    try:
        #df_as_np = df.to_numpy()
        df_as_np = df
        X = []
        y = []
        logger.info(f"Length of df_as_np: {len(df)}")
        if len(df) <= int(window_size):
            print(f"Not enough data points for window size {window_size}")
            logger.error(f"Not enough data points for window size {window_size}")
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            return None, None
        for i in range(len(df_as_np) - window_size):
            window_rows = df_as_np[i:i + window_size]
            X.append(window_rows)
            label = df_as_np[i + window_size]
            y.append(label)
        return np.array(X), np.array(y)
    except Exception as e:
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        logger.error(f"Error converting DataFrame to X, y: {str(e)}")
        log_message = ("Error converting DataFrame to X, y during training")
        #publisher.publish(user_id, log_message)
        return None, None

def evaluate_model(predictions, actuals, model_id, selected_inputs):
    logger.info(f"this is evaulate: {selected_inputs}")
    results = []
    try:
        for i in range(actuals.shape[1]):
            mse = mean_squared_error(actuals[:, i], predictions[:, i])
            variance_target = np.var(actuals[:, i])
            if variance_target != 0:
                mse_percentage = round((mse / variance_target) * 100, 3)
            else:
                mse_percentage = 0
            rmse = np.sqrt(mse)
            mean_target = np.mean(actuals[:, i])
            mae = mean_absolute_error(actuals[:, i], predictions[:, i])
            if mean_target != 0:
                rmse_percentage = round((rmse / mean_target) * 100, 3)
                mae_percentage = round((mae / mean_target) * 100, 3)
            else:
                rmse_percentage = 0
                mae_percentage = 0
            
            results.append(
                {   'feature' : selected_inputs[i],
                    'MSE' : f'{mse_percentage}%',
                    'RMSE' : f'{rmse_percentage}%',
                    'MAE' : f'{mae_percentage}%',
                })
        return results
    except Exception as e:
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        logger.error(f"Error evaluating model: {str(e)}")
        log_message = ("Error evaluating model during training")
        #publisher.publish(user_id, log_message)
        return None
    
def start_training(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path):
    try:
        log_message = f"LSTM Training process started for project ID: {project_id}"
        publisher.publish(model_id, log_message)
        publisher.publish(user_id, log_message)
        logger = setGlobal(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path)
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "In Progress","start_timestamp": get_bangkok_time() }})
        if not all([selected_inputs, parameters, training_file_path]):
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            log_message = ("Error Missing required input data")
            #publisher.publish(user_id, log_message)
            raise ValueError("Missing required input data.")
        logger.info("Training process started")
        df = load_file(training_file_path, model_id)
        if df is None:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            log_message = ("Error Failed to load file during training")
            #publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to load file.")
        #df.index = df.pop('index')
        df.index = df.pop('Timestamp')
        #df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
        #df.index = pd.to_datetime(df.index, format=['%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S.%z'])
        logger.info(f"Loaded DataFrame:\n{df}")
        temp, selected_inputs = attribute(df, selected_inputs, model_id)
        # temp = attribute(df, selected_inputs, model_id)
        if temp is None:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            log_message = ("Error Failed to process selected inputs during training")
            #publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to process selected inputs.")
        X, y = df_to_X_y(temp, parameters["Sequence Length"], model_id)
        if X is None or y is None:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            log_message = ("Error Failed to convert DataFrame to X, y during training")
            #publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to convert DataFrame to X, y.")
        logger.info("Successfuly convert DataFrame to X, y.")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-parameters["Training Ratio"], random_state=42)
        logger.info('completed data split')
        logger.info(f"X train shape: {X_train.shape}, y train shape: {y_train.shape}, X validate shape: {X_val.shape}, y validate shape: {y_val.shape}")
        model = Sequential()
        model.add(InputLayer((parameters["Sequence Length"], X_train.shape[2])))
        logger.info('added layer')
        model.add(Bidirectional(LSTM(parameters["First Units"])))
        logger.info('added algorithm')
        model.add(Dense(parameters["Second Units"], activation=parameters["First Activation"]))
        logger.info('added activation function')
        model.add(Dense(X_train.shape[2], activation=parameters["Second Activation"]))
        logger.info("Model architecture summary:")
        logger.info(model.summary())

        
        
        
        cp = ModelCheckpoint(f'/home/user/symbiont_aiml/{user_id}/{project_id}/model/', save_best_only=True)
        if (parameters['Optimizer'] == 'Adam'):
            model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=parameters["Learning Rate"] ), metrics=[RootMeanSquaredError()])
        # elif (parameters['Optimizer'] == 'Adam'):
        #     model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=parameters["Learning Rate"] ), metrics=[RootMeanSquaredError()])

        class TrainingProgress(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
            def on_epoch_end(self, epoch, logs=None):
                verbose_level = self.params['verbose']if self.params else 1
                if epoch >= 0 and verbose_level > 0:
                    progress = (round((epoch + 1) / (self.params['epochs']+class_parameters['Epochs']) * 100, 2))
                    log_message = f"Progress: {progress}%\t\tEpoch {epoch + 1}\t\tLoss: {logs['loss']:.6f}\t\tRMSE: {logs['root_mean_squared_error']:.6f}"
                    logger.info(log_message)
                    publisher.publish(model_id, log_message)
                    db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"progress_percentage": progress}})
                
        
        
        training_progress = TrainingProgress()


        history1 = model.fit(X_train, y_train, epochs=parameters["Epochs"], batch_size=parameters["Batch Size"],
                            validation_data=(X_val, y_val), callbacks=[cp, training_progress], verbose=2)

        save_train_img(user_id, project_id)
        plt.plot(history1.history['loss'], label='Training loss')
        plt.plot(history1.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        train_results_dir = f"/home/user/symbiont_aiml/{user_id}/{project_id}/training_result/"
        os.makedirs(train_results_dir, exist_ok=True)
        train_results_path = os.path.join(train_results_dir, "loss_plot.png")
        plt.savefig(train_results_path)

        plt.close()
        
        plt.plot(history1.history['root_mean_squared_error'], label='Training RMSE')
        plt.plot(history1.history['val_root_mean_squared_error'], label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        rmse_plot_path = os.path.join(train_results_dir, 'rmse_plot.png')
        plt.savefig(rmse_plot_path)
        
        logger.info(f"Saved training image result")
        try:
            traing_result_file_check = db["project files"].find_one({"project_id": project_id, "filename": "loss_plot.png"})
            if not traing_result_file_check:
                projectFile = ProjectFile(**{
                "filename": "loss_plot.png",
                "file_type": ".png",
                "asset_type": "training result",
                "file_size": None,
                "file_path": f"/{user_id}/{project_id}/training_result/loss_plot.png",
                "created_timestamp": get_bangkok_time(),
                "project_id": project_id
            })
                result = db["project files"].insert_one(dict(projectFile))
            else:
                db["project files"].update_one({"project_id": project_id},{"$set": {"created_timestamp": get_bangkok_time()}})
        except Exception as e:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            logger.error(f"Error in saving training result image: {str(e)}")
            log_message = ("Error in saving training result image during training")
            #publisher.publish(user_id, log_message)
        try:
            traing_result_file_check = db["project files"].find_one({"project_id": project_id, "filename": "rmse_plot.png"})
            if not traing_result_file_check:
                projectFile = ProjectFile(**{
                "filename": "rmse_plot.png",
                "file_type": ".png",
                "asset_type": "training result",
                "file_size": None,
                "file_path": f"/{user_id}/{project_id}/training_result/rmse_plot.png",
                "created_timestamp": get_bangkok_time(),
                "project_id": project_id
            })
                result = db["project files"].insert_one(dict(projectFile))
            else:
                db["project files"].update_one({"project_id": project_id},{"$set": {"created_timestamp": get_bangkok_time()}})
        except Exception as e:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            logger.error(f"Error in saving training result image: {str(e)}")
            log_message = ("Error in saving training result image during training")
            #publisher.publish(user_id, log_message)

        train_predictions = model.predict(X_train)
        logger.info(f"TRAINED PREDICTIONS HERE: {train_predictions}")
        train_results = evaluate_model(train_predictions, y_train, model_id, selected_inputs)
        print("this is the test result -->", train_results)
        
        if train_results is None:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            log_message = ("Error Failed to evaluate model during training")
            #publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to evaluate model.")
        # formatted_results = {k: v for d in train_results for k, v in d.items()}
        # logger.info(f"Training results:\n{formatted_results}")

        
        #db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Completed","end_timestamp": get_bangkok_time() }})
        #db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"progress_percentage": "100"}})
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"evaluation": train_results}})

        log_message = f"LSTM Training completed for project ID: {project_id}"
        publisher.publish(model_id, log_message)
        log_message = "<------------------------------------------------------------------------->"
        publisher.publish(model_id, log_message)
        

        start_classification_training(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path)
        
    except Exception as e:
        #db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        #logger.error(f"Error in training process: {str(e)}")
        log_message = "Error in training process"
        #publisher.publish(user_id, log_message)
        #publisher.publish(model_id, log_message)

def start_classification_training(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path):
    try:
        logger = setGlobal(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path)
        logger.info("Classification training started")
        log_message = f"Classification training started for project ID: {project_id}"
        publisher.publish(model_id, log_message)
        publisher.publish(user_id, log_message)
        df = load_file(class_training_file_path, user_id)
        if df is None:
            db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
            log_message = ("Error Failed to load file during training")
            #publisher.publish(user_id, log_message)
            raise RuntimeError("Failed to load file.")
        df.index = df.pop('Timestamp')
        #df.index = pd.to_datetime(df.index, format = ['%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S.%z'])
        logger.info("Done preprocessing data for classification")
        X = df[selected_inputs]
        y = df['class']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_one_hot = pd.get_dummies(y_encoded)

        X_train,X_test,y_train, y_test = train_test_split(X, y_one_hot, test_size=class_parameters["Training Ratio"], random_state=42)
        logger.info(X_train.shape)
        logger.info(y_train.shape)

        class TrainingProgress(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
            def on_epoch_end(self, epoch=class_parameters['Epochs'], logs=None):
                verbose_level = self.params['verbose']if self.params else 1
                if epoch >= 0 and verbose_level > 0:
                    progress = (round((epoch + 1 + parameters['Epochs']) / (parameters['Epochs']+class_parameters['Epochs']) * 100, 2))
                    log_message = (f"Progress: {progress}%\t\tEpoch: {epoch + 1}\t\tLoss: {logs['loss']:.6f} \t\tAccuracy: {logs['accuracy']:.6f}")
                    logger.info(log_message)
                    publisher.publish(model_id, log_message)
                    db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"progress_percentage": progress}})
    
        training_progress = TrainingProgress()

        model = Sequential()
        model.add(Dense(class_parameters["First Units"], input_dim=X_train.shape[1], activation=class_parameters["First Activation"]))
        model.add(Dense(class_parameters["Second Units"], activation=class_parameters["Second Activation"]))
        model.add(Dense(3, activation='softmax')) # 3 output neurons for 'normal', 'warning', 'critical'
        cp = ModelCheckpoint(f'/home/user/symbiont_aiml/{user_id}/{project_id}/model_class/', save_best_only=True)
        if class_parameters['Optimizer'] == "Adam":
            model.compile(optimizer=Adam(learning_rate=class_parameters["Learning Rate"] ), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=class_parameters["Epochs"], batch_size=class_parameters["Batch Size"], validation_data=(X_test, y_test), callbacks=(cp, training_progress))
        logger.info("Model architecture summary:")
        logger.info(model.summary())

        loss, accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        #NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        predictions = model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(np.argmax(y_test.values, axis=1), predicted_labels)
        label_mapping = {0: 'critical', 1: 'normal', 2: 'warning'}
        class_report = classification_report(np.argmax(y_test.values, axis=1), predicted_labels, target_names = label_mapping.values())
        logger.info(f"classification report: \n{class_report}")
        #NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Completed","end_timestamp": get_bangkok_time() }})
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"progress_percentage": "100"}})
        log_message = f"Classification training ended for project ID: {project_id}"
        lines = class_report.strip().split('\n')
        data = {}
        class_labels = []
        for line in lines[2:]:
            line_data = line.split()
            if len(line_data) == 5:
                class_label = line_data[0]
                class_labels.append(class_label)
                data[class_label] = {
                    "precision": float(line_data[1]),
                    "recall": float(line_data[2]),
                    "f1-score": float(line_data[3]),
                    "support": float(line_data[4])
                }
            elif len(line_data) == 2 and line_data[0] == "accuracy":
                data["accuracy"] = float(line_data[1])
        evaluation_data = []
        for class_label in class_labels:
            class_data = {
                "class": class_label,
                "precision": data[class_label]["precision"],
                "recall": data[class_label]["recall"],
                "f1-score": data[class_label]["f1-score"],
                "support": data[class_label]["support"]
            }
            evaluation_data.append(class_data)

        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"class_evaluation": evaluation_data}})
        publisher.publish(model_id, log_message)
        publisher.publish(user_id, log_message)

    except Exception as e:
        db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Failed","end_timestamp": get_bangkok_time() }})
        logger.error(f"Error in classification training process: {str(e)}")

