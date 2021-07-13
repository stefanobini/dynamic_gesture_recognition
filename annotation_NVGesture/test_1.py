import json


def load_json(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


data = load_json('nvgesture.json')
d = data['database'].items()
train = [0]*25
val = [0]*25
test = [0]*25
for key, value in d:
    if value['subset'] == 'training':
        c_train = int(key.split('/')[0][-2:])
        train[c_train-1] += 1
    elif value['subset'] == 'validation':
        c_val = int(key.split('/')[0][-2:])
        val[c_val-1] += 1
    else:
        c_test = int(key.split('/')[0][-2:])
        test[c_test-1] += 1
print(train)
print(val)
print(test)