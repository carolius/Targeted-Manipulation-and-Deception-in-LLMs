import json
import os

folder = "targeted_llm_manipulation/config/env_configs/tickets"

for file in os.listdir(folder):
    if file.endswith(".json"):
        print(file)
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)
        for key, value in data["histories"].items():
            # assert len(value) == 5, f"History does not contain 5 messages: {key}. file: {file}"
            for message in value:
                assert "content" in message, f"Message does not contain content: {key}. file: {file}"
                assert "role" in message, f"Message does not contain role: {key}. file: {file}"
