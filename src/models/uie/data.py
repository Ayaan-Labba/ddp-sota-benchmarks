import sys

from typing import List, Dict, Set, Tuple, Any, Optional

sys.path.append("../")

from common.data import GenerativeIEDataset
from configs.config_loader import ConfigLoader

class UIEDataset(GenerativeIEDataset):
    """
    Dataset class for UIE models.
    Implements methods to create labels in Structured Extraction Language (SEL) format.

    Args:
        data (List[Dict[str, Any]]): List of examples in standard format.
        dataset_name (str): Name of the dataset.
        tokenizer: Tokenizer for the specific model.
        config_loader (Optional[ConfigLoader]): ConfigLoader instance.
    """

    def __init__(
            self, 
            data: List[Dict[str, Any]], 
            dataset_name: str, 
            tokenizer, 
            config_loader: Optional[ConfigLoader] = None, 
            **kwargs
        ):
        super().__init__(
            data=data,
            dataset_name=dataset_name,
            model_name='uie',  # automatically set model name
            tokenizer=tokenizer,
            config_loader=config_loader,
            **kwargs
        )
        
        # Extract UIE-specific tokens
        self.START = self.special_tokens['start']
        self.END = self.special_tokens['end']
        self.TARGET = self.special_tokens['target']

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
        ssi += " <extra_id_2>"
        return ssi
    
    def create_inputs(self, example: Dict[str, Any]) -> str:
        """Creates input text with ssi prefix."""
        ent_types = [self.map_entity_type(ent) for ent in self.dataset_config.entity_types]
        rel_types = [self.map_relation_type(rel) for rel in self.dataset_config.relation_types]
        ssi = self.build_ssi(ent_types, rel_types)
        return f"{ssi} {example['text']}".strip()
    
    def create_labels(self, example: Dict[str, Any]) -> str:
        """Creates the Structured Extraction Language (SEL) string"""
        records = []
        
        for ent in example.get('entities', []):
            ent_span = ent['text']
            ent_type = self.map_entity_type(ent['type'])
            ent_rels = []
            
            for rel in example.get('relations', []):
                if (rel['head']['text'] == ent['text'] and 
                    all(off in ent['offsets'] for off in rel['head']['offsets'])):
                    
                    rel_type = self.map_relation_type(rel['type'])
                    obj_span = rel['tail']['text']
                    ent_rels.append(
                        f"{self.START} {rel_type} {self.TARGET} {obj_span} {self.END}"
                    )
            
            if ent_rels:
                record = (f"{self.START} {ent_type} {self.TARGET} {ent_span} " + 
                         " ".join(ent_rels) + f" {self.END}")
            else:
                record = f"{self.START} {ent_type} {self.TARGET} {ent_span} {self.END}"
            
            records.append(record)
        
        return " ".join(records).strip()
    
    @staticmethod
    def parse_sel(sel_string: str, start_token: str, end_token: str, target_token: str) -> List[Dict[str, Any]]:
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
            START (str): Start token.
            END (str): End token.
            TARGET (str): Token separating spot and asoc.
        
        Returns:
            structured_output (List[Dict[str, any]]): A list of dictionaries containing parsed triplets.
        """
        structured_output = []

        # The entire SEL is wrapped in start_token/end_token
        if sel_string.startswith(start_token) and sel_string.endswith(end_token):
            sel_string = sel_string[len(start_token):-len(end_token)].strip()
        else:
            return structured_output

        # Each block is wrapped in start_token/end_token
        records: List[str] = []
        start_index = sel_string.find(start_token)

        # Handle cases where the string might not contain any records or is malformed
        if start_index == -1 and len(sel_string) > 0:
            return structured_output

        while start_index != -1:
            end_index = len(sel_string)
            curr_start = start_index
            while curr_start < end_index and curr_start != -1 and end_index != -1:
                end_index = sel_string.find(end_token, curr_start + len(start_token))
                curr_start = sel_string.find(start_token, curr_start + len(start_token))
            
            if end_index == -1:
                return structured_output
            
            # Extract the content between start_token and end_token
            record_content = sel_string[start_index + len(start_token):end_index].strip()
            if record_content:
                records.append(record_content)

        # Process each record
        for record in records:
            try:
                # Extract entity type
                start = record.find(start_token)
                subj_sep = record.find(target_token)
                if not (start == 0 and subj_sep > start):
                    continue
                
                entity_type = record[start + len(start_token):subj_sep].strip()

                # Extract entity span and relations (if any)
                remaining_record = record[subj_sep + len(target_token):].strip()
                
                # Check if there are associations after the main span
                rel_start = remaining_record.find(start_token)
                
                subj_span = ""
                relations = []

                if rel_start != -1: # relations exist
                    subj_span = remaining_record[:rel_start].strip()
                    while rel_start != -1:
                        rel_end = remaining_record.find(end_token, rel_start + len(start_token))
                        if rel_end == -1:
                            break
                        
                        # Extract the content between START and END
                        relation_str = remaining_record[rel_start + len(start_token):rel_end].strip()
                        target_sep = relation_str.find(target_token)
                        if target_sep == -1:
                            continue

                        rel_type = relation_str[:target_sep].strip()
                        obj_span = relation_str[target_sep + len(target_token):].strip()
                        if rel_type and obj_span:
                            relations.append((rel_type, obj_span))
                        
                        rel_start = remaining_record.find(start_token, rel_end + len(end_token))

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
    
    @staticmethod
    def get_sel_preds(parsed_sel: List[Dict[str, Any]]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
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
    
    def get_predictions(self, sel_string: str) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
        """
        Parse and extract predictions in one go.

        Args:
            sel_string (str): The SEL string to parse.
        
        Returns:
            Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]
        """
        parsed = self.parse_sel(sel_string, self.START, self.END, self.TARGET)
        return self.get_sel_preds(parsed)