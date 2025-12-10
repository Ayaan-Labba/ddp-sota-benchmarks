import json
import os

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

def save_predictions(predictions: List[Dict[str, any]], output_dir: str, filename: str) -> None:
    """
    Saves model predictions to a JSONL file in the specified output directory.

    Args:
        predictions (List[Dict[str, any]]): List of prediction dictionaries to save.
        output_dir (str): Predictions output directory.
        filename (str): Name of the output JSNOL file.
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    save_jsonl(predictions, output_path)
    print(f"Predictions saved to {output_path}")