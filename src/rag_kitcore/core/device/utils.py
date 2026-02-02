import torch

# -------------------------------------------------
# Device selection
# -------------------------------------------------

def get_device(preferred_index: int = 0) -> str:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        index = preferred_index if preferred_index < count else 0
        return f"cuda:{index}"
    return "cpu"
