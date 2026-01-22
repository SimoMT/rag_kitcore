def ensure_dir(path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)
