from re import A
import sys
from pathlib import Path


def setup_local_libraries(anchor_file=None):
    """Add AIStudio local dependency folders to sys.path when present.

    The competition environment may install wheels into /home/aistudio/libraries
    or a sibling ../libraries directory. Call this before importing optional
    packages such as flash-attn.
    """
    if anchor_file is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(anchor_file).resolve().parent

    candidates = [
        base_dir / "libraries",
        base_dir.parent / "libraries",
        Path("/home/aistudio/libraries"),
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


setup_local_libraries()
