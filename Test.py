import json

with open(r"D:\Course\GenAI-Code\FastAPI\patients.json","r") as f:
    data = json.load(f)
    print(data.keys())

    data["2"].delattr()
    print(data)