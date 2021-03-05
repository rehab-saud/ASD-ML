import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'q1':1, 'q2':1, 'q3':0, 'q4':1, 'q5':1, 'q6':0, 'q7':1, 'q8':1, 'q9':0, 'q10':1})

print(r.json())