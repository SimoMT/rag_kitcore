import yaml
import subprocess

with open("/app/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = cfg.get("llm", {}).get("model")
if not model:
    raise ValueError("No llm.model found in config.yaml")

print(f"Pulling Ollama model: {model}")
subprocess.run(["/bin/ollama", "pull", model], check=True)
