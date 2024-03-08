from datetime import datetime, timedelta
import subprocess
import time
import json
import os
from config.database import db
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from ML_training import *
from prediction2 import *
# default_args = {
# 'owner': 'coder2j',
# 'retries': 5,
# 'retry_delay': timedelta(minutes=5)
#}


default_args = {
	'owner': 'symbiont_aiml'
}

def prdicting_lstm():
	pass

def run_script(forecast_duration, selected_inputs, parameters, result_directory, prediction_file_path, model_id, prediction_id, project_id, user_id):
	result = start_prediction(forecast_duration, selected_inputs, parameters, result_directory, prediction_file_path, model_id, prediction_id, project_id, user_id)

#def greet():
#	print("Greet")
#	for i in range(0, 100):
#		time.sleep(1)
#		print(" >> delayed : ", i)
#	print("! Done Greeting.")

def meet(**kwargs):
	forecast_duration = kwargs['dag_run'].conf.get('forecast_duration')
	selected_inputs = kwargs['dag_run'].conf.get('selected_inputs')
	parameters = kwargs['dag_run'].conf.get('parameters')
	result_directory = kwargs['dag_run'].conf.get('result_directory')
	prediction_file_path = kwargs['dag_run'].conf.get('prediction_file_path')
	model_id = kwargs['dag_run'].conf.get('model_id')
	prediction_id = kwargs['dag_run'].conf.get('prediction_id')
	project_id = kwargs['dag_run'].conf.get('project_id')
	user_id = kwargs['dag_run'].conf.get('user_id')
	kwargs['ti'].xcom_push(key='model_id', value=model_id)
	print(forecast_duration)
	print(selected_inputs)
	print(parameters)
	print(result_directory)
	print(prediction_file_path)
	print(model_id)
	print(project_id)
	print(user_id)
	print("Meet n")
	result = run_script(forecast_duration, selected_inputs, parameters, result_directory, prediction_file_path, model_id, prediction_id, project_id, user_id)
	return {"script_result": result}
	#for i in range(0, 100):
	#	time.sleep(1)
	#	print(" >> delayed : ", i)
	#print("! Done meeting.")



def failure_meet(context):
	model_id = context['it'].xcom_pull(task_ids='meet', key='model_id', include_prior_dates=True)
	# model_id = kwargs['dag_run'].conf.get('model_id')
	print("failure meet is called.", context['task_instance'])
	db["predictions"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Aborted","end_timestamp": get_bangkok_time()}})
	return "sucessfully abort"
	# logging
	# update mongodb


#def failure_greet(context):
#	print("failure greet is called.", context['task_instance'])

with DAG (
	default_args=default_args,
	dag_id='predicting_dag_v01',
	description='AIML engine predicting session using LSTM for Symbiont AIML project.',
	catchup=False,
	max_active_runs=2,
 	start_date=datetime(2024, 1, 3),
	schedule='@once') as dag:

	predicting_task = PythonOperator(
		task_id='predicting',
		python_callable=meet,
 		on_failure_callback=failure_meet
	)

	#task2 = PythonOperator(
	#	task_id='greet',
	#	python_callable=greet,
	#	on_failure_callback=failure_greet
	#)

	#task1.set_downstream(task2)
 	#task1.set_downstream(task2)

