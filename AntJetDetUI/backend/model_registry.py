import json
import os

MODELS_FILE = "models.json"

def load_models():
    if not os.path.exists(MODELS_FILE):
        return {}
    try:
        with open(MODELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load models.json: {e}")
        return {}

def save_models(models_dict):
    try:
        with open(MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(models_dict, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Could not save models.json: {e}")
