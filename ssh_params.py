from dataclasses import dataclass
from typing import Optional


@dataclass
class SSHParams:
    host: str
    user: str
    port: Optional[int] = 22
    fpath_private_key: Optional[str] = None  # on the local machine