import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm
from datasets import dataset_dict
from typing import List, Dict

def clean_text(text_str: str) -> str:
    """
    Normalize whitespace (e.g., \n, \t) to a single space.
    Normalize Unicode with NFKC.
    Strips leading/trailing whitespace.
    """
    text_str = unicodedata.normalize('NFKC', text_str) # unicode normalization
    text_str = re.sub(r'\s+', ' ', text_str) # whitespace normalization
    return text_str.strip()

def process_bigbio(dataset_dicts: List[dataset_dict], name: str, splits: List[str]) -> None:
    """
    Process a HuggingFace bigbio dataset into the standard format and save to "data/{name}/{split}.jsonl".

    Standard Format:
    {
        "id": str,
        "text": str,
        "entities": 
            [
                {
                    "text": str,
                    "type", str,
                    "offsets": [(int, int), ...]
                }, 
                ...
            ],
        "relations": 
            [
                {
                    "type": str,
                    "head", dict(entity),
                    "tail", dict(entity)
                },
                ...
            ]
    }

    Args:
        dataset_dicts (List[dataset_dict]): List of HuggingFace dataset splits for the given dataset.
        name (str): Name of the dataset (e.g., "ChemProt").
        splits (List[str]): List of split names corresponding to dataset_dicts (e.g., ["train", "validation", "test"]).
    
    Returns:
        None
    """
    print(f"Processing {name} dataset ...\n")
    output_dir = f"../data/{name.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset in zip(splits, dataset_dicts):
        processed_data = []
        for doc in tqdm(dataset, desc=f"Processing {split}"):
            doc_id = doc['id']
            
            # Clean and combine every passage in a document to a single text string
            doc_text = clean_text(' '.join([' '.join(passage['text']) for passage in doc['passages']]))
            
            # Process entities
            doc_ent_map = {} # mapping to retrieve entities by id for relations
            for ent in doc['entities']:
                id = ent['id']
                type = ent['type']
                text = ' '.join(ent['text'])
                offsets = [(offset[0], offset[1]) for offset in ent['offsets']] # include every occurrence in the document
                data = {'text': text, 'type': type, 'offsets': offsets}
                doc_ent_map[id] = data
            
            doc_ents = list(doc_ent_map.values())
            
            # Process relations
            doc_rels = []
            for rel in doc['relations']:
                type = rel['type']
                h_id, t_id = rel['arg1_id'], rel['arg2_id']
                head, tail = doc_ent_map.get(h_id), doc_ent_map.get(t_id) # handle missing entities with .get()
                if not (head and tail):
                    continue
                    
                doc_rels.append({'type': type, 'head': head, 'tail': tail})

            processed_data.append({
            'id': doc_id, 
            'text': doc_text, 
            'entities': doc_ents, 
            'relations': doc_rels
            })

        # Save processed split to a JSNOL file
        output_path = f"{output_dir}/{split}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n') # dump each dictionary as its own line in the file
        
        print(f"Saved processed {split} split to {output_path}.\n")
    
    print(f"Finished processing {name} dataset.")

def process_bc5cdr(dataset_paths: List[str], splits: List[str]) -> None:
    """
    Process BC5CDR BioC XML files into the standard format and save to "data/bc5cdr/{split}.jsonl".

    Standard Format:
    {
        "id": str,
        "text": str,
        "entities": 
            [
                {
                    "text": str,
                    "type", str,
                    "offsets": [(int, int), ...]
                }, 
                ...
            ],
        "relations": 
            [
                {
                    "type": str,
                    "head", dict(entity),
                    "tail", dict(entity)
                },
                ...
            ]
    }

    Args:
        dataset_paths (List[str]): List of file paths to the BC5CDR XML files.
        splits (List[str]): List of split names corresponding to dataset_paths (e.g., ["train", "validation", "test"]).
    
    Returns:
        None
    """
    print(f"Processing BC5CDR dataset ...\n")
    output_dir = "../data/bc5cdr"
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset_path in zip(splits, dataset_paths):
        tree = ET.parse(dataset_path)
        root = tree.getroot()

        processed_data = []
        for doc in tqdm(root.findall('document'), desc=f"Processing {split}"):
            doc_id = doc.find('id').text
            
            # Process and combine entities and texts from all passages in the document
            doc_text = []
            doc_ent_map = {}
            curr_offset = 0
            for passage in doc.findall('passage'):
                text = passage.find('text').text
                doc_text.append(text)
                for ent in passage.findall('annotation'):
                    id = ent.find('infon[@key="MESH"]').text
                    location = ent.find('location')
                    start = int(location.get('offset')) + curr_offset
                    length = int(location.get('length'))
                    if not (start and length):
                        continue

                    if id not in doc_ent_map:
                        doc_ent_map[id] = {
                            'text': ent.find('text').text,
                            'type': ent.find('infon[@key="type"]').text,
                            'offsets': []
                        }
                    
                    doc_ent_map[id]['offsets'].append((start, start + length))
                
                curr_offset += len(text) + 1
            
            doc_text = clean_text(" ".join(doc_text))
            doc_ents = list(doc_ent_map.values())

            # Process relations
            doc_rels = []
            for rel in doc.findall('relation'):
                h_id, t_id = rel.find('infon[@key="Chemical"]').text, rel.find('infon[@key="Disease"]').text
                head, tail = doc_ent_map.get(h_id), doc_ent_map.get(t_id)
                if not (head and tail):
                    continue

                doc_rels.append({
                    'type': "CID", # same relation type for all relations in BC5CDR
                    'head': head,
                    'tail': tail
                })
            
            processed_data.append({
                'id': doc_id,
                'text': doc_text,
                'entities': doc_ents,
                'relations': doc_rels
            })

        # Save processed split to a JSONL file
        output_path = f"{output_dir}/{split}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n') # dump each dictionary as its own line in the file
            
        print(f"Saved processed {split} split to {output_path}.\n")
    
    print("Finished processing BC5CDR dataset.")

def process_biored(dataset_paths: List[str], splits: List[str]) -> None:
    """
    Process BioRED BioC JSON files into the standard format and save to "data/biored/{split}.jsonl".

    Standard Format:
    {
        "id": str,
        "text": str,
        "entities": 
            [
                {
                    "text": str,
                    "type", str,
                    "offsets": [(int, int), ...]
                }, 
                ...
            ],
        "relations": 
            [
                {
                    "type": str,
                    "head", dict(entity),
                    "tail", dict(entity)
                },
                ...
            ]
    }

    Args:
        dataset_paths (List[str]): List of file paths to the BioRED JSON files.
        splits (List[str]): List of split names corresponding to dataset_paths (e.g., ["train", "validation", "test"]).
    
    Returns:
        None
    """
    print(f"Processing BioRED dataset ...\n")
    output_dir = "../data/biored"
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset_path in zip(splits, dataset_paths):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        processed_data = []
        for doc in tqdm(dataset['documents'], desc=f"Processing {split}"):
            doc_id = doc['id']
            
            # Process and combine entities and texts from all passages in the document
            doc_text = []
            doc_ent_map = {}
            curr_offset = 0
            for passage in doc['passages']:
                text = passage['text']
                doc_text.append(text)
                for ent in passage['annotations']:
                    id = ent['infons']['identifier']
                    offsets = []
                    for loc in ent['locations']:
                        start = int(loc.get('offset')) + curr_offset
                        length = int(loc.get('length'))
                        if not start and length:
                            continue

                        offsets.append((start, start + length))
                    
                    if id not in doc_ent_map:
                        doc_ent_map[id] = {
                            'text': ent['text'],
                            'type': ent['infons']['type'],
                            'offsets': []
                        }
                    
                    doc_ent_map[id]['offsets'].extend(offsets)
                
                curr_offset += len(text) + 1
            
            doc_text = clean_text(" ".join(doc_text))
            doc_ents = list(doc_ent_map.values())

            # Process relations
            doc_rels = []
            for rel in doc['relations']:
                h_id, t_id = rel['infons']['entity1'], rel['infons']['entity2']
                head, tail = doc_ent_map.get(h_id), doc_ent_map.get(t_id)
                if not (head and tail):
                    continue

                doc_rels.append({"type": rel['infons']['type'], "head": head, "tail": tail})

            processed_data.append({
                'id': doc_id,
                'text': doc_text,
                'entities': doc_ents,
                'relations': doc_rels
            })

        # Save processed split to a JSONL file
        output_path = f"{output_dir}/{split}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n') # dump each dictionary as its own line in the file
        
        print(f"Saved processed {split} split to {output_path}.\n")
    
    print("Finished processing BioRED dataset.")

def get_types(name: str, splits: List[str]) -> Dict[str, List[str]]:
    """
    Saves and returns entity and relation types present in the given dataset.

    Args:
        name (str): Name of the dataset (e.g., "ChemProt").
        splits (List[str]): List of splits to check (e.g., ["train", "validation", "test"]).
    
    Returns:
        Dict[str, List[str]]: A dictionary with 'entity_types' and 'relation_types' as keys and lists of types as values.
    """
    print(f"Getting entity and relation types for {name} dataset ...\n")
    ent_types = set()
    rel_types = set()
    data_dir = f"../data/{name.lower()}"
    for split in splits:
        data = []
        with open(data_dir + f"/{split}.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        for example in tqdm(data, desc=f"Getting {split} entity and relation types"):
            for ent in example['entities']:
                ent_types.add(ent['type'])
            
            for rel in example['relations']:
                rel_types.add(rel['type'])
    
    types = {
        'entity_types': list(ent_types),
        'relation_types': list(rel_types)
    }

    # Save types to a JSON file
    output_path = data_dir + "/types.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(types, f, indent=4)
    
    print(f"Saved entity and relation types to {output_path}.\n")
    return types