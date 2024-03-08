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
# default_args = {
# 'owner': 'coder2j',
# 'retries': 5,
# 'retry_delay': timedelta(minutes=5)
#}


default_args = {
	'owner': 'symbiont_aiml'
}

def training_lstm():
	pass

def run_script(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path):
	result = start_training(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path)
	# result2 = start_classification_training(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id)

#def greet():
#	print("Greet")
#	for i in range(0, 100):
#		time.sleep(1)
#		print(" >> delayed : ", i)
#	print("! Done Greeting.")

def meet(**kwargs):
	selected_inputs = kwargs['dag_run'].conf.get('selected_inputs')
	parameters = kwargs['dag_run'].conf.get('parameters')
	file_path = kwargs['dag_run'].conf.get('file_path')
	training_file_path = kwargs['dag_run'].conf.get('training_file_path')
	model_id = kwargs['dag_run'].conf.get('model_id')
	user_id = kwargs['dag_run'].conf.get('user_id')
	project_id = kwargs['dag_run'].conf.get('project_id')
	class_algorithm_name = kwargs['dag_run'].conf.get('class_algorithm_name')
	class_parameters = kwargs['dag_run'].conf.get('class_parameters')
	class_training_file_path = kwargs['dag_run'].conf.get('class_training_file_path')
	kwargs['ti'].xcom_push(key='model_id', value=model_id)
	print(selected_inputs)
	print(parameters)
	print(file_path)
	print(training_file_path)
	print(model_id)
	print("Meet n")
	result = run_script(selected_inputs, parameters, file_path, training_file_path, model_id, user_id, project_id, class_parameters, class_training_file_path)
	return {"script_result": result}
	#for i in range(0, 100):
	#	time.sleep(1)
	#	print(" >> delayed : ", i)
	#print("! Done meeting.")



def failure_meet(context):
	model_id = context['it'].xcom_pull(task_ids='meet', key='model_id', include_prior_dates=True)
	# model_id = kwargs['dag_run'].conf.get('model_id')
	print("failure meet is called.", context['task_instance'])
	db["models"].update_one({"_id": ObjectId(model_id)},{"$set": {"status": "Aborted","end_timestamp": get_bangkok_time()}})
	return "sucessfully abort"
	# logging
	# update mongodb


#def failure_greet(context):
#	print("failure greet is called.", context['task_instance'])

with DAG (
	default_args=default_args,
	dag_id='training_dag_v01',
	description='AIML engine training session using LSTM for Symbiont AIML project.',
	catchup=False,
	max_active_runs=1,
 	start_date=datetime(2024, 1, 3),
	on_failure_callback=failure_meet,
	schedule='@once') as dag:

	task1 = PythonOperator(
		task_id='training',
		python_callable=meet,
 		
	)
