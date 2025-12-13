import yaml
import sys

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# sys.path.append("../../")

@dataclass
class LabelMapping:
    entities: Dict[str, str]
    relations: Dict[str, str]

@dataclass
class ModelConfig:
    """Model-specific parameters like special tokens and sequence lengths."""
    name: str
    special_tokens: Dict[str, str] = field(default_factory=dict)
    max_input_length: int = 512
    max_output_length: int = 256
    other_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    name: str
    label_mapping: LabelMapping
    entity_types: list
    relation_types: list
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.model_configs.get(model_name)


class ConfigLoader:
    """
    Loads and manages dataset and model configurations from YAML files.
    
    Args:
        datasets_path: Path to datasets config file.
        models_path: Path to models config file.
    """
    
    def __init__(self, datasets_path: str = "configs/datasets.yaml", models_path: str = "configs/models.yaml"):
        self.datasets_path = Path(datasets_path)
        self.models_path = Path(models_path)
        
        # Load configurations
        self.dataset_mappings: Dict[str, Dict[str, Dict[str, str]]] = self.load_yaml(self.datasets_path)
        self.model_settings: Dict[str, Dict[str, Any]] = self.load_yaml(self.models_path)
        
        # Build structured configs
        self.dataset_configs = self.build_dataset_configs()
        self.model_configs = self.build_model_configs()
    
    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """
        Load YAML config file.
        
        Args:
            path: Path to the YAML file.
        
        Returns:
            Parsed YAML content as a dictionary.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def build_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """
        Build DatasetConfig objects from loaded configurations

        Returns:
            A dictionary mapping dataset names to DatasetConfig objects.
        """
        configs = {}
        for dataset_name, mappings in self.dataset_mappings.items():
            # Create label mapping
            label_mapping = LabelMapping(
                entities = mappings.get('entities', {}),
                relations = mappings.get('relations', {})
            )
            
            # Create dataset config
            configs[dataset_name] = DatasetConfig(
                name=dataset_name,
                label_mapping=label_mapping,
                entity_types=list(mappings.get('entities', {}).keys()),
                relation_types=list(mappings.get('relations', {}).keys()),
            )
        
        return configs
    
    def build_model_configs(self) -> ModelConfig:
        """
        Create ModelConfig objects from loaded configurations.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            model_settings: Settings dictionary for the model
        
        Returns:
            A dictionary mapping model names to ModelConfig objects.
        """
        configs = {}
        for model_name, settings in self.model_settings.items():
            configs[model_name] = ModelConfig(
                name=model_name,
                special_tokens=settings.get('special_tokens', {}),
                max_input_length=settings.get('max_input_length', 512),
                max_output_length=settings.get('max_output_length', 512),
                other_params=settings.get('other_params', {})
            )
        
        return configs
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset.
        
        Returns:
            DatasetConfig object for the specified dataset.
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset '{dataset_name}' not found in configurations")
        
        return self.dataset_configs[dataset_name]
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model.
        
        Returns:
            ModelConfig object for the specified model.
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in configurations")
        
        return self.model_configs[model_name]
    
    def list_datasets(self) -> List:
        """List all available datasets"""
        return list(self.dataset_configs.keys())
    
    def list_models(self) -> List:
        """List all available models"""
        return list(self.model_configs.keys())


# Global config loader instance
config_loader = None

def get_config_loader(
    datasets_path: str = None,
    models_path: str = None,
    reload: bool = False
) -> ConfigLoader:
    """
    Get or create the global config loader instance.
    
    Args:
        datasets_path: Path to datasets config file
        models_path: Path to models config file
        reload: Force reload of configurations
    
    Returns:
        ConfigLoader instance
    """
    global config_loader
    
    if config_loader is None or reload:
        if datasets_path is None:
            project_root = Path(__file__).parent.parent.parent
            datasets_path = project_root / "src/configs/datasets.yaml"
        
        if models_path is None:
            project_root = Path(__file__).parent.parent.parent
            models_path = project_root / "src/configs/models.yaml"
        
        config_loader = ConfigLoader(datasets_path, models_path)
    
    return config_loader