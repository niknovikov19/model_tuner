from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssh_client import SSHClient


# SSH parameters
hostname = 'lethe.downstate.edu'
username = 'niknovikov19'
port = 1415
private_key_path = r'C:\Users\aleks\.ssh\id_rsa_lethe'

# HPC folder
remote_dir = 'ddn/niknovikov19'

# SSH Client
ssh_client = SSHClient(hostname, username, port, private_key_path)

# Test ssh-based file system
with ssh_client.get_filesys_handler() as fs:
    print(f'Contents of {remote_dir} via SSHFS')
    print(fs.listdir(remote_dir))

# Test ssh-based command running
with ssh_client.get_conn_handler() as conn:
    result = conn.run('ls -l', hide=True)
    print(f'\nContents of {remote_dir} via "ls - l"')
    print(result.stdout.strip())
    
