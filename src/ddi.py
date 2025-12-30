import json
import os
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm
from typing import List, Dict


def preprocess_ddi(dataset_dir, output_dir):
    data_samples = []
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all XML files in the directory
    for root_dir, _, files in tqdm(os.walk(dataset_dir), desc=f"Processing train files"):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root_dir, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                doc_id = root.get('id')

                # Iterate through sentences
                for sent in root.findall('sentence'):
                    sent_id = sent.get('id')
                    text = sent.get('text')
                    
                    # Extract Entities (NER)
                    ent_map: Dict[str, List] = {}
                    for ent in sent.findall('entity'):
                        e_id = ent.get('id')
                        ent_map[e_id] = []
                        e_type = ent.get('type')
                        e_text = ent.get('text')
                        offsets = ent.get('charOffset').split(';')
                        for offset in offsets:
                            start, end = tuple(offset.split('-'))
                            ent_map[e_id].append({'id': e_id, 'text': e_text, 'type': e_type, 'start': start, 'end': end})

                    entities = list(ent_map.values())

                    # Extract Relation Triplets (RE)
                    relations = []
                    for pair in sent.findall('pair'):
                        e1_id = pair.get('e1')
                        e2_id = pair.get('e2')
                        ddi = pair.get('ddi')
                        for head in ent_map[e1_id]:
                            for tail in ent_map[e2_id]:
                                h_offset, h_text, h_type = [head['start'], head['end']], head['text'], head['type']
                                t_offset, t_text, t_type = [tail['start'], tail['end']], tail['text'], tail['type']
                                relation = {'head_id': e1_id, 
                                            'tail_id': e2_id, 
                                            'head': h_offset, 
                                            'tail': t_offset, 
                                            'head_text': h_text, 
                                            'tail_text': t_text, 
                                            'head_type': h_type, 
                                            'tail_type': t_type, 
                                            'ddi': ddi}
                                if ddi == 'true':
                                    rel_type = pair.get('type')
                                    relation['type'] = rel_type
                                
                                relations.append(relation)

                    # Store parsed sample
                    data_samples.append({
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'text': text,
                        'entities': entities,
                        'relations': relations
                    })
    
    output_path = os.path.join(output_dir, "train.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(data_samples)} examples to {output_path}")

def preprocess_ddi_test(dataset_dir, output_dir):
    data_samples = []
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all XML files in the directory
    for root_dir, _, files in tqdm(os.walk(dataset_dir), desc="Processing test split"):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root_dir, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                doc_id = root.get('id')

                # Iterate through sentences
                for sent in root.findall('sentence'):
                    sent_id = sent.get('id')
                    text = sent.get('text')
                    
                    # Extract Entities (NER)
                    ent_map: Dict[str, List] = {}
                    for ent in sent.findall('entity'):
                        e_id = ent.get('id')
                        ent_map[e_id] = []
                        e_type = ent.get('type')
                        e_text = ent.get('text')
                        offsets = ent.get('charOffset').split(';')
                        ent_map[e_id] = {'id': e_id, 'text': e_text, 'type': e_type, 'offsets': offsets}

                    entities = list(ent_map.values())

                    # Extract Relation Triplets (RE)
                    relations = []
                    for pair in sent.findall('pair'):
                        e1_id = pair.get('e1')
                        e2_id = pair.get('e2')
                        ddi = pair.get('ddi')
                        head, tail = ent_map[e1_id], ent_map[e2_id]
                        h_offsets, h_text, h_type = head['offsets'], head['text'], head['type']
                        t_offsets, t_text, t_type = tail['offsets'], tail['text'], tail['type']
                        relation = {'head_id': e1_id, 
                                    'tail_id': e2_id, 
                                    'head': h_offsets, 
                                    'tail': t_offsets, 
                                    'head_text': h_text, 
                                    'tail_text': t_text, 
                                    'head_type': h_type, 
                                    'tail_type': t_type, 
                                    'ddi': ddi}
                        if ddi == 'true':
                            rel_type = pair.get('type')
                            relation['type'] = rel_type
                        
                        relations.append(relation)

                    # Store parsed sample
                    data_samples.append({
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'text': text,
                        'entities': entities,
                        'relations': relations
                    })
    
    output_path = os.path.join(output_dir, "test.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(data_samples)} examples to {output_path}")

def preprocess_ddi_seq2seq(dataset_dir, output_dir, split):
    data_samples = []
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all XML files in the directory
    for root_dir, _, files in tqdm(os.walk(dataset_dir), desc=f"Processing {split} files"):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root_dir, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                doc_id = root.get('id')

                # Iterate through sentences
                for sent in root.findall('sentence'):
                    sent_id = sent.get('id')
                    text = sent.get('text')
                    
                    # Extract Entities (NER)
                    ent_map: Dict[str, Dict] = {}
                    for ent in sent.findall('entity'):
                        e_id = ent.get('id')
                        ent_map[e_id] = []
                        e_type = ent.get('type')
                        e_text = ent.get('text')
                        offsets = ent.get('charOffset').split(';')
                        start, end = tuple(offsets[0].split('-')) # only text matching matters in seq2seq
                        ent_map[e_id] = {'id': e_id, 'text': e_text, 'type': e_type, 'start': start, 'end': end}

                    entities = list(ent_map.values())

                    # Extract Relation Triplets (RE)
                    relations = []
                    for pair in sent.findall('pair'):
                        e1_id = pair.get('e1')
                        e2_id = pair.get('e2')
                        ddi = pair.get('ddi')
                        head, tail = ent_map[e1_id], ent_map[e2_id]
                        h_offset, h_text, h_type = [head['start'], head['end']], head['text'], head['type']
                        t_offset, t_text, t_type = [tail['start'], tail['end']], tail['text'], tail['type']
                        relation = {'head_id': e1_id, 
                                    'tail_id': e2_id, 
                                    'head': h_offset, 
                                    'tail': t_offset, 
                                    'head_text': h_text, 
                                    'tail_text': t_text, 
                                    'head_type': h_type, 
                                    'tail_type': t_type, 
                                    'ddi': ddi}
                        if ddi == 'true':
                            rel_type = pair.get('type')
                            relation['type'] = rel_type
                        
                        relations.append(relation)

                    # Store parsed sample
                    data_samples.append({
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'text': text,
                        'entities': entities,
                        'relations': relations
                    })
    
    output_path = os.path.join(output_dir, f"{split}.jsonl")    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(data_samples)} examples to {output_path}")

def main():
    train_dir = "/home/bt19d200/Ayaan/raw-datasets/DDICorpus/Train"
    test_dir = "/home/bt19d200/Ayaan/raw-datasets/DDICorpus/Test/Test for DDI Extraction task"
    extraction_dir = "bio-domain-datasets"
    seq2seq_dir = "bio-domain-datasets-seq2seq"
    
    print("---------- Preprocessing DDI Dataset ----------\n")
    print("---------- Processing for extraction models ----------")
    preprocess_ddi(dataset_dir=train_dir, output_dir=extraction_dir)
    preprocess_ddi_test(dataset_dir=test_dir, output_dir=extraction_dir)
    print()
    print("---------- Processing for seq2seq models ----------")
    preprocess_ddi_seq2seq(dataset_dir=train_dir, output_dir=seq2seq_dir, split='train')
    preprocess_ddi_seq2seq(dataset_dir=test_dir, output_dir=seq2seq_dir, split='test')
    print()
    print("---------- Finished preprocessing DDI Dataset ----------")


if __name__ == "__main__":
    main()