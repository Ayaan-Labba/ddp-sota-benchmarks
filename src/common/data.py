import sys

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.append("../")

from configs.config_loader import get_config_loader, DatasetConfig, ModelConfig, LabelMapping, ConfigLoader


class BaseIEDataset(Dataset, ABC):
    """Base class for all Relation Extraction datasets
    
    Args:
        data (List[Dict[str, Any]]): List of examples in standard format.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        tokenizer: Tokenizer for the specific model.
        config_loader (Optional[ConfigLoader]): ConfigLoader instance.
    """
    
    def __init__(
            self, 
            data: List[Dict[str, Any]], 
            dataset_name: str, 
            model_name: str, 
            tokenizer, 
            config_loader: Optional[ConfigLoader] = None, 
            **kwargs
        ):
        self.data = data
        self.dataset_name = dataset_name
        self.model_name = model_name
        
        # Get or create config loader
        if config_loader is None:
            config_loader = get_config_loader()
        
        self.config_loader = config_loader
        
        # Load configurations
        self.dataset_config: DatasetConfig = config_loader.get_dataset_config(dataset_name)
        self.model_config: ModelConfig = config_loader.get_model_config(model_name)
        
        # Set up mappings
        self.entity_map: Dict[str, str] = self.dataset_config.label_mapping.entities
        self.relation_map: Dict[str, str] = self.dataset_config.label_mapping.relations
        
        # Model-specific attributes
        self.tokenizer = tokenizer
        self.special_tokens: Dict[str, str] = self.model_config.special_tokens
        self.max_input_length: int = self.model_config.max_input_length
        
    def __len__(self):
        return len(self.data)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return processed example for the specific model and dataset"""
        pass

    @abstractmethod
    def create_inputs(self, idx: int) -> Dict[str, Any]:
        """Return inputs for the specific model and dataset"""
        pass
    
    @abstractmethod
    def create_labels(self, example: Dict[str, Any]) -> Any:
        """Create model and dataset specific labels from standard format"""
        pass
    
    def map_entity_type(self, entity_type: str) -> str:
        """Map entity type to semantically rich tokens"""
        return self.entity_map.get(entity_type, entity_type)
    
    def map_relation_type(self, relation_type: str) -> str:
        """Map relation type to semantically rich tokens"""
        return self.relation_map.get(relation_type, relation_type)


class GenerativeIEDataset(BaseIEDataset):
    """Base class for generative (seq2seq) models.
    
    Args:
        data (List[Dict[str, Any]]): List of examples in standard format.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        tokenizer: Tokenizer for the specific model.
        config_loader (Optional[ConfigLoader]): ConfigLoader instance.
    """
    
    def __init__(
            self, 
            data: List[Dict[str, Any]], 
            dataset_name: str, 
            model_name: str, 
            tokenizer: AutoTokenizer, 
            config_loader = None, 
            **kwargs
        ):
        super().__init__(data, dataset_name, model_name, tokenizer, config_loader, **kwargs)
        self.max_output_length = self.model_config.max_output_length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        input_text = self.create_inputs(example)
        label_text = self.create_labels(example)
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label_encoding = self.tokenizer(
            label_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'id': example['id'],
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': label_encoding['input_ids'].squeeze(),
            'text': example['text']
        }