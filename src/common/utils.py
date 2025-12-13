import json
import os

from typing import List, Dict, Set, Tuple

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

def get_ground_truth(example: Dict, entity_map: Dict, relation_map: Dict) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
    """Extracts ground truth entities and relations into sets for comparison.
    
    Args:
        example (Dict): The data example containing entities and relations.
        entity_map (Dict): Mapping from entity label tokens to standardised types.
        relation_map (Dict): Mapping from relation label tokens to standardised types.

    Returns:
        Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]: Sets of ground truth entities and relations.
    """
    gt_entities = set()
    gt_relations = set()

    for entity in example.get('entities', []):
        entity_span = entity['text']
        entity_type = entity_map.get(entity['type'], entity['type'])
        gt_entities.add((entity_span, entity_type))
    
    for relation in example.get('relations', []):
        head_span = relation['head']['text']
        tail_span = relation['tail']['text']
        relation_type = relation_map.get(relation['type'], relation['type'])
        gt_relations.add((head_span, relation_type, tail_span))

    return gt_entities, gt_relations

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