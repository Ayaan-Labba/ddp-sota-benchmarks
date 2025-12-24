import json
import os
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm
from typing import List


def process_bc5cdr(dataset_paths: List[str], splits = List[str]) -> None:
    """
    Process BC5CDR BioC XML files into the standard format and save to "data/bc5cdr/{split}.jsonl".

    Standard Format:
    {
        "id": str,
        "text": str,
        "entities":
            [
                {
                    "id": str,
                    "text": str,
                    "type": str,
                    "offset": [int, int]
                }, 
                ...
            ],
        "relations": 
            [
                {
                    "type": str,
                    "head_id": str,
                    "tail_id": str
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
    print(f"--- Processing BC5CDR dataset --- \n")
    output_dir = "../data/bc5cdr"
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset_path in zip(splits, dataset_paths):
        tree = ET.parse(dataset_path)
        root = tree.getroot()
        processed_data = []
        for doc in tqdm(root.findall('document'), desc=f"Processing {split} split"):
            doc_id = doc.find('id').text

            # Combine title and abstract texts and entities
            doc_text = ""
            doc_ents = []
            offset_shift = 0 # accounts for space added before every passage
            for passage in doc.findall('passage'):
                text = passage.find('text').text
                if doc_text:
                    doc_text += " " + text
                    offset_shift += 1
                else:
                    doc_text = text
                    offset_shift = 0

                for ent in passage.findall('annotation'):
                    location = ent.find('location')
                    start = int(location.get('offset')) + offset_shift
                    length = int(location.get('length'))
                    doc_ents.append({
                        'id': ent.find('infon[@key="MESH"]').text,
                        'text': ent.find('text').text,
                        'type': ent.find('infon[@key="type"]').text,
                        'offset': [start, start + length]
                    })

            # Get relations
            doc_rels = []
            for rel in doc.findall('relation'):
                doc_rels.append({
                    'type': "CID", # same relation type for all relations in BC5CDR
                    'head': rel.find('infon[@key="Chemical"]').text,
                    'tail': rel.find('infon[@key="Disease"]').text
                })
            
            processed_data.append({
                'id': doc_id,
                'text': doc_text,
                'entities': doc_ents,
                'relations': doc_rels
            })

        # Save processed split to JSONL file
        output_path = f"{output_dir}/{split}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n') # dump each dictionary as its own line in the file
            
        print(f"Saved processed {split} split to {output_path}.\n")
    
    print("--- Finished processing BC5CDR dataset. ---")

def main():
    train_path = "/home/bt19d200/Ayaan/raw-datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml"
    val_path = "/home/bt19d200/Ayaan/raw-datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml"
    test_path = "/home/bt19d200/Ayaan/raw-datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml"
    dataset_paths = [train_path, val_path, test_path]
    splits = ['train', 'val', 'test']
    process_bc5cdr(dataset_paths=dataset_paths, splits=splits)

if __name__ == "__main__":
    main()