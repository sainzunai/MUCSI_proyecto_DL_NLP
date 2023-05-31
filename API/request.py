import requests
res = requests.post('http://localhost:5000/api/depression/', json={"prompt":"lalala"})
if res.ok:
    print(res.json())