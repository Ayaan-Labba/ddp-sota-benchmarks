import torch
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Dict, Any

# sys.path.append("../../")

from src.models.uie.data import UIEDataset
from src.common.utils import get_ground_truth

def run_inference(
        model: AutoModelForSeq2SeqLM, 
        dataloader: DataLoader, 
        tokenizer: AutoTokenizer
    ) -> List[Dict[str, Any]]:
    """
    Run inference on the test set and collect predictions.

    Args:
        model (AutoModelForSeq2SeqLM): The UIE model for inference.
        dataloader (DataLoader): DataLoader for the test dataset.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        dataset (UIEDataset): The dataset object containing config and parsing methods.
    
    Returns:
        List of prediction dictionaries with both predictions and ground truth
    """
    predictions = []
    dataset: UIEDataset = dataloader.dataset
    batch_size: int = dataloader.batch_size
    dataset_config = dataset.dataset_config
    model_config = dataset.model_config
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            example_ids = batch['id']
            input_texts = batch['text']
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=model_config.max_output_length
            )
            
            # Decode predictions
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            # Process each prediction
            for idx, (example_id, input_text, pred_text) in enumerate(zip(example_ids, input_texts, pred_texts)):
                # Get the original example for ground truth
                original_example = dataset[batch_idx * batch_size + idx]
                
                # Get predictions
                pred_entities, pred_relations = dataset.get_predictions(pred_text)
                
                # Get ground truth using the mapping from dataset config
                gt_entities, gt_relations = get_ground_truth(
                    original_example,
                    dataset_config.label_mapping.entities,
                    dataset_config.label_mapping.relations
                )
                
                # Store prediction
                prediction = {
                    'id': example_id,
                    'text': input_text,
                    'pred_sel': pred_text,
                    'pred_entities': [list(e) for e in pred_entities],
                    'pred_relations': [list(r) for r in pred_relations],
                    'gt_entities': [list(e) for e in gt_entities],
                    'gt_relations': [list(r) for r in gt_relations]
                }
                predictions.append(prediction)
    
    return predictions