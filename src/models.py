import torch
import pytorch_lightning as pl

from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from typing import List, Optional

class GenerativeModel(pl.LightningModule):
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 5e-4,
        warmup_steps: int = 0,
        max_source_length: int = 512,
        max_target_length: int = 512
    ) -> None:
        """
        Base wrapper for Seq2Seq IE models (REBEL, UIE).
        
        Args:
            model: The loaded HuggingFace model (e.g., T5ForConditionalGeneration or BartForConditionalGeneration).
            tokenizer: The corresponding tokenizer.
            learning_rate: Learning rate for the optimizer.
            warmup_steps: Steps for linear warmup.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        If labels are provided, the model returns loss automatically.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Expects batch to be a dict containing 'input_ids', 'attention_mask', and 'labels'.
        Data tokenization should happen in the DataLoader/Collate function for efficiency.
        """
        outputs = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Sets up AdamW optimizer and a linear scheduler."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches # use the number of training steps from the trainer
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def inference(self, texts: List[str], num_beams: int = 3) -> List[str]:
        """
        End-to-end inference: Tokenize raw text -> Generate -> Detokenize.
        """
        self.eval() # Ensure model is in eval mode
        
        # Tokenize input text
        inputs = self.tokenizer(
            texts, 
            max_length=self.hparams.max_source_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()} # move inputs to the same device as the model
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.hparams.max_target_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return decoded_preds