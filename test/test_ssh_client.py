from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssh_client import SSHParams, SSHClient


# SSH parameters
ssh_par = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)

# HPC folder
remote_dir = 'ddn/niknovikov19'

with SSHClient(ssh_par) as ssh:
    # Test ssh-based file system
    print(f'Contents of {remote_dir} via SSHFS')
    print(ssh.fs.listdir(remote_dir))

    # Test ssh-based command running
    result = ssh.conn.run('ls -l', hide=True)
    print(f'\nContents of {remote_dir} via "ls - l"')
    print(result.stdout.strip())
    
