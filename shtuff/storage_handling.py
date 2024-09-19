import json


def read_json(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except UnicodeDecodeError as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None


def save_json(file_path: str, data) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)



