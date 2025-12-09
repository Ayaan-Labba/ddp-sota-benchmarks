import json

from typing import List, Dict, Set, Tuple

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

def calculate_metrics(preds: List[Tuple[Set, Set]], golds: List[Tuple[Set, Set]]) -> Dict:
    """Calculates precision, recall and f1-score for entities and relations.
    
    Args:
        preds (List[Tuple[Set, Set]]): List of tuples containing predicted entities and relations.
        golds (List[Tuple[Set, Set]]): List of tuples containing ground truth entities and relations.

    Returns:
        Dict: Dictionary containing precision, recall and f1-score for entities and relations.
    """
    total_ent_tp, total_ent_fp, total_ent_fn = 0, 0, 0
    total_rel_tp, total_rel_fp, total_rel_fn = 0, 0, 0

    for (pred_ents, pred_rels), (gold_ents, gold_rels) in zip(preds, golds):
        # Entity metrics
        total_ent_tp += len(pred_ents.intersection(gold_ents))
        total_ent_fp += len(pred_ents.difference(gold_ents))
        total_ent_fn += len(gold_ents.difference(pred_ents))

        # Relation metrics
        total_rel_tp += len(pred_rels.intersection(gold_rels))
        total_rel_fp += len(pred_rels.difference(gold_rels))
        total_rel_fn += len(gold_rels.difference(pred_rels))

    ent_precision = total_ent_tp / (total_ent_tp + total_ent_fp) if (total_ent_tp + total_ent_fp) > 0 else 0
    ent_recall = total_ent_tp / (total_ent_tp + total_ent_fn) if (total_ent_tp + total_ent_fn) > 0 else 0
    ent_f1 = 2 * (ent_precision * ent_recall) / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0

    rel_precision = total_rel_tp / (total_rel_tp + total_rel_fp) if (total_rel_tp + total_rel_fp) > 0 else 0
    rel_recall = total_rel_tp / (total_rel_tp + total_rel_fn) if (total_rel_tp + total_rel_fn) > 0 else 0
    rel_f1 = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0

    return {
        "entity_precision": ent_precision,
        "entity_recall": ent_recall,
        "entity_f1": ent_f1,
        "relation_precision": rel_precision,
        "relation_recall": rel_recall,
        "relation_f1": rel_f1,
    }

def get_metrics(file_path: str) -> Dict[str, float]:
    """
    Reads from predictions file and calculates precision, recall and f1-score.

    Args:
        file_path (str): Path to the predictions file.

    Returns:
        Dict[str, float]: Dictionary containing calculated metrics.
    """
    metrics = {}
    all_preds_sets = []
    all_labels_sets = []
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    data = json.loads(line)
                    # Convert saved lists back to sets of tuples
                    pred_entities = set(tuple(e) for e in data['pred_entities'])
                    pred_relations = set(tuple(r) for r in data['pred_relations'])
                    gt_entities = set(tuple(e) for e in data['gt_entities'])
                    gt_relations = set(tuple(r) for r in data['gt_relations'])

                    all_preds_sets.append((pred_entities, pred_relations))
                    all_labels_sets.append((gt_entities, gt_relations))
                
                except json.JSONDecodeError: # handle malformed JSON lines
                        print(f"Skipping malformed JSON line {line_count} in {file_path}")

        metrics = calculate_metrics(all_preds_sets, all_labels_sets)
    
    except Exception as e: # handle file read errors
            print(f"Error reading or processing {file_path}: {e}")

    return metrics