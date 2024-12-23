import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssh_client import SSHParams, SSHClient


def path_join(*parts, sep='/'):
    return sep.join(str(part).strip(sep) for part in parts)

# SSH parameters
ssh_par = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)

# Remote folder
dirpath_remote = 'ddn/niknovikov19/test/model_tuner/test_ssh_upload'

# Local folder
dirpath_local = str(Path(__file__).resolve().parent)

with SSHClient(ssh_par) as ssh:    
    # Create a file locally
    fname_test = 'test_file'
    fpath_test_local = path_join(dirpath_local, fname_test, sep='\\')
    with open(fpath_test_local, 'w') as fid:
        fid.write('This is a test file to upload via SSH')
    
    # Test file path in the remote filesystem
    fpath_test_remote = path_join(dirpath_remote, fname_test, sep='/')
    
    # Create remote folder
    ssh.fs.makedirs(dirpath_remote, recreate=True)
    
    # If the test file already exists in the remote folder - delete it
    if ssh.fs.exists(fpath_test_remote):
        print('Remote test file exists - deleting')
        ssh.fs.remove(fpath_test_remote)
    else:
        print('Remote test file does not exist yet')
    
    # Confirm that the file does not exist
    if ssh.fs.exists(fpath_test_remote):
        print('Remote test file still exists - BAD!')
        
    # Upload test file
    ssh.fs.upload_file(fpath_test_local, fpath_test_remote)
    print('Test file uploaded')
    
    # Read the remote test file content
    with ssh.fs.open(fpath_test_remote, 'r') as fid:
        content = fid.read()
        print('Content of the remote test file:')
        print(content)
