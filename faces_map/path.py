from pathlib import Path
from typing import Union

def ensure_dir(path: Union[str, Path]) -> None:
    """Make sure that the (filepath's) directory and its parents exist"""
    path = Path(path)
    if path.exists():
        return
    is_dir: bool = not path.suffixes
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
