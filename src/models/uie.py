import json
import os
import torch
import sys

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict, Set, Tuple

sys.path.append("../")

from common.utils import load_jsonl

def build_ssi(entity_types: List[str], relation_types: List[str]) -> str:
    """
    Builds the Structured Schema Instructor (SSI) string.
    Uses <spot> for entities and <asoc> for relations.

    Args:
        entity_types (List[str]): List of entity type strings.
        relation_types (List[str]): List of relation type strings.
    
    Returns:
        str: The constructed SSI string.
    """
    ssi = "<spot> " + "<spot> ".join(entity_types)
    ssi += " <asoc> " + "<asoc> ".join(relation_types)
    ssi += " <extra_id_2> "
    return ssi

def parse_sel(sel_string: str, s_tok: str, e_tok: str, t_tok: str) -> List[Dict[str, any]]:
    """
    Parses the generated SEL string into the structured format using defined markers.
    
    Structured format:
        [
            {
                "span": "{entity span}",
                "spot": "{entity type}",
                "asoc":
                    [
                        ["{relation type}", "{object span}"],
                        ...
                    ]
            },
            ...
        ]

    Args:
        sel_string (str): The SEL string to parse.
        s_tok (str): Start token.
        e_tok (str): End token.
        t_tok (str): Token separating spot and asoc.
    
    Returns:
        structured_output (List[Dict[str, any]]): A list of dictionaries containing parsed triplets.
    """
    structured_output = []

    # The entire SEL is wrapped in s_tok/e_tok
    if sel_string.startswith(s_tok) and sel_string.endswith(e_tok):
        sel_string = sel_string[len(s_tok):-len(e_tok)].strip()
    else:
        return structured_output

    # Each block is wrapped in s_tok/e_tok
    records: List[str] = []
    start_index = sel_string.find(s_tok)

    # Handle cases where the string might not contain any records or is malformed
    if start_index == -1 and len(sel_string) > 0:
        return structured_output

    while start_index != -1:
        end_index = len(sel_string)
        curr_start = start_index
        while curr_start < end_index and curr_start != -1 and end_index != -1:
            end_index = sel_string.find(e_tok, curr_start + len(s_tok))
            curr_start = sel_string.find(s_tok, curr_start + len(s_tok))
        
        if end_index == -1:
            return structured_output
        
        # Extract the content between s_tok and e_tok
        record_content = sel_string[start_index + len(s_tok):end_index].strip()
        if record_content:
             records.append(record_content)

    # Process each record
    for record in records:
        try:
            # Extract entity type
            start = record.find(s_tok)
            subj_sep = record.find(s_tok)
            if not (start == 0 and subj_sep > start):
                 continue
            
            entity_type = record[start + len(s_tok):subj_sep].strip()

            # Extract entity span and relations (if any)
            remaining_record = record[subj_sep + len(t_tok):].strip()
            
            # Check if there are associations after the main span
            rel_start = remaining_record.find(s_tok)
            
            subj_span = ""
            relations = []

            if rel_start != -1: # relations exist
                subj_span = remaining_record[:rel_start].strip()
                while rel_start != -1:
                    rel_end = remaining_record.find(e_tok, rel_start + len(s_tok))
                    if rel_end == -1:
                        break
                    
                    # Extract the content between s_tok and e_tok
                    relation_str = remaining_record[rel_start + len(s_tok):rel_end].strip()
                    target_sep = relation_str.find(t_tok)
                    if target_sep == -1:
                        continue

                    rel_type = relation_str[:target_sep].strip()
                    obj_span = relation_str[target_sep + len(t_tok):].strip()
                    if rel_type and obj_span:
                        relations.append((rel_type, obj_span))
                    
                    rel_start = remaining_record.find(s_tok, rel_end + len(e_tok))

            else: # no relations, the rest is the entity span
                subj_span = remaining_record

            # Add to output if valid
            if entity_type and subj_span:
                record_info = {'span': subj_span, 'spot': entity_type, 'asoc': relations}
                structured_output.append(record_info)

        except Exception as e:
            print(f"Error parsing block: {record}\nError: {e}")
            continue

    return structured_output

def get_sel_preds(parsed_sel: List[Dict['str', any]]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
    """Extracts predicted entities and relations from the parsed SEL structure.
    
    Args:
        parsed_sel (List[Dict[str, any]]): Parsed SEL structure.
    
    Returns:
        Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]: Sets of predicted entities and relations.
    """
    pred_entities = set()
    pred_relations = set()

    for record in parsed_sel:
        subj_span = record['span']
        subj_type = record['spot']
        pred_entities.add((subj_span, subj_type))
        for rel_type, obj_span in record['asoc']:
            pred_relations.add((subj_span, rel_type, obj_span))

    return pred_entities, pred_relations