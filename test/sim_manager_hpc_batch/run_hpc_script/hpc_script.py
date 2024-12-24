import sys
import time


print('Test script started')
print(f'Arguments: {sys.argv[1:]}', flush=True)

time.sleep(5)

fpath_result = sys.argv[1]
with open(fpath_result, 'w') as fid:
    fid.write('This is the result of the test script')

print('Test script finished')