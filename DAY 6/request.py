import requests

url = 'http://localhost:2500/predict_api'
r = requests.post(url,json={'User ID':15668575,'Age':26,'Estimated Salary':43000})

print(r.json())


