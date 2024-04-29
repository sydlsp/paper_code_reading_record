import json

file_path="./mydata/train/train.json"

json_file=open(file_path,'r')

json_content=json_file.read()

#将json转化为字典
json_content=json.loads(json_content)


print(json_content['images'][1])

"""
[{'id': 1, 'name': 'car'}
{'id': 2, 'name': 'truck'}
{'id': 3, 'name': 'bus'}
{'id': 4, 'name': 'van'}
{'id': 5, 'name': 'freight_car'}]
"""