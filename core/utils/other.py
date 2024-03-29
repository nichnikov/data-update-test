import json


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def read_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data
