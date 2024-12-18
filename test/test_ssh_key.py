from fs.sshfs import SSHFS
import paramiko

# Path to your private key
private_key_path = r'C:\Users\aleks\.ssh\id_rsa_lethe'

# Server details
hostname = 'lethe.downstate.edu'
username = 'niknovikov19'
port = 1415

# Load the private key
try:
    private_key = paramiko.RSAKey(filename=private_key_path)
    print("Private key loaded successfully.")
    
    with SSHFileSystem(hostname, username=username,
                       client_keys=[private_key_path], port=port) as fs:
        print("Contents of remote directory '/':")
        print(fs.listdir('/'))

    # Test SSHFS connection
    fs = SSHFileSystem(hostname, username=username,
                       client_keys=[private_key_path], port=port)
    print("SSHFileSystem initialized successfully.")
    
    print("Contents of remote directory '/':")
    print(fs.listdir('/'))
        
    fs.close()
    print("SSHFS connection successful!")

except Exception as e:
    print("Error during SSHFS connection:", e)
