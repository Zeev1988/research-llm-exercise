import os
from pathlib import Path
from typing import List



def get_python_files_from_repo(root: Path) -> List[Path]:
    files = []
    for dirpath, dir_name, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".py"):
                files.append(Path(dirpath) / name)

    return files
