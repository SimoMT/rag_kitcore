import os

def resolve_input_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith((".pdf", ".docx"))
        )

    raise ValueError(f"Invalid path: {path}")
