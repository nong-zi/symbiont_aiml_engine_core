import requests
import json
from datetime import datetime
import time


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

	
if __name__ == "__main__":
	access_token = get_access()
	if access_token != None:
		query_influx(access_token)