import os
import sys
import time


print('Test script started')
print(f'Arguments: {sys.argv[1:]}', flush=True)

time.sleep(5)

dirpath_base = sys.argv[1]
fpath_result = os.path.join(dirpath_base, 'result', 'result.out')
with open(fpath_result, 'w') as fid:
    fid.write('This is the result of the test script')

print('Test script finished')