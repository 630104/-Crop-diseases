import requests
import json

def send_request(image_path, env_data):
    api_endpoint = 'https://api.example.com/predict'
    files = {'image': ('image.jpg', open(image_path, 'rb'))}
    data = {'env_data': env_data}
    response = requests.post(api_endpoint, files=files, json=data)
    if response.status_code == 200:
        print('Request successful!')
    else:
        print(f'Request failed: {response.status_code}')
    return response.json()