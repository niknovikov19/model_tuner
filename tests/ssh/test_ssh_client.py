from pathlib import Path
import sys

# Folder that contains model_tuner package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from model_tuner.ssh import SSHParams, SSHClient


# SSH parameters
ssh_par_lethe = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)
ssh_par_grid = SSHParams(
    host='grid',
    user='niknovikov19',
    fpath_private_key=r'C:\Users\aleks\.ssh\id_ed25519_grid'
)

# HPC folder
remote_dir = 'ddn/niknovikov19'

with SSHClient(
        ssh_par_fs=ssh_par_lethe,
        ssh_par_conn=[ssh_par_lethe, ssh_par_grid]
        ) as ssh:
    # Test ssh-based file system
    print(f'Contents of {remote_dir} via SSHFS')
    print(ssh.fs.listdir(remote_dir))

    # Test ssh-based command running
    result = ssh.conn.run('ls -l', hide=True)
    print(f'\nContents of {remote_dir} via "ls - l"')
    print(result.stdout.strip())
    
