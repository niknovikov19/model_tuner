from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssh_conn_custom import SSHConnCustom
from ssh_params import SSHParams


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

with SSHConnCustom([ssh_par_lethe, ssh_par_grid]) as conn:
    result = conn.run('uname -a', hide=True)
    print(result.stdout.strip())
    conn.close()
