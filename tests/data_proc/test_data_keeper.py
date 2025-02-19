import os
import shutil

from model_tuner.data_proc.data_keeper import DataKeeper, DataFormat


dirpath_dk = r'D:\WORK\Salvador\repo\model_tuner\proto\test_data\test_data_keeper'
if os.path.exists(dirpath_dk):
    shutil.rmtree(dirpath_dk)
os.makedirs(dirpath_dk, exist_ok=True)

dk = DataKeeper(dirpath_dk)

data_info = {}

data_info['data1'] = {
    'params': {'par11': 11, 'par12': '12'},
    'data': {'data_content': 'DATA1'}
}
data_info['data2'] = {
    'params': {'par21': 21, 'par22': '22'},
    'data': {'data_content': 'DATA2'}
}

data_names = ['data1', 'data2']

for data_name, info in data_info.items():
    dk.store_data(info['data'], data_name, info['params'])

data_name = 'data1'
info = data_info[data_name]
data_new = {'data_content': 'DATA1_NEW'}

# Re-write data (when prohibited)
try:
    dk.store_data(data_new, data_name, info['params'])  # it should fail
except Exception as e:
    print(f'Exception: {e}')

# Re-write data and read it back
dk.store_data(data_new, data_name, info['params'], allow_rewrite=True)
data_new_stored = dk.get_data(data_name, info['params'])
print(data_new_stored)

# Add new data with the same name but different params
data_name = 'data2'
params_new = {'par21': 21, 'par22': '22_new'},
data_new = {'data_content': 'DATA2_NEW'}
dk.store_data(data_new, data_name, params_new)
data_new_stored = dk.get_data(data_name, params_new)
print(data_new_stored)

# Add new data without params, save as json
data_name = 'data3'
data_new = {'data_content': 'DATA3'}
dk.store_data(data_new, data_name, data_format=DataFormat.JSON)
data_new_stored = dk.get_data(data_name)
print(data_new_stored)
