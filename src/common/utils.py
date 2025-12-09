import json

from typing import List, Dict

def load_jsonl(file_path: str) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Dict]: List of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Saves a list of dictionaries into a JSONL file.
    
    Args:
        data (List[Dict]): List of dictionaries to save.
        file_path (str): Path to the output JSONL file.

    Returns:
        None
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')