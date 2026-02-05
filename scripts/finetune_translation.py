# """
# Fine-tune Translation Model (Helsinki-NLP) on Domain-Specific Data
# Use QA datasets to create domain-specific parallel corpus
# """

# from pathlib import Path
# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from datasets import load_from_disk, Dataset, DatasetDict
# from transformers import (
#     MarianMTModel,
#     MarianTokenizer,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
#     DataCollatorForSeq2Seq,
#     EarlyStoppingCallback
# )
# from evaluate import load
# import torch
# import logging
# from typing import Dict, List, Optional
# import numpy as np

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class TranslationFineTuner:
#     """Fine-tune Helsinki-NLP translation models"""
    
#     def __init__(
#         self,
#         model_name: str = "Helsinki-NLP/opus-mt-en-vi",
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         max_length: int = 512
#     ):
#         self.model_name = model_name
#         self.device = device
#         self.max_length = max_length
        
#         logger.info(f"Loading model: {model_name}")
#         self.tokenizer = MarianTokenizer.from_pretrained(model_name)
#         self.model = MarianMTModel.from_pretrained(model_name).to(device)
        
#         self.bleu_metric = load("bleu")
        
#     def create_parallel_corpus_from_xquad(
#         self,
#         xquad_en_path: Path,
#         xquad_vi_path: Path
#     ) -> DatasetDict:
#         """
#         Create EN-VI parallel corpus from XQuAD parallel benchmark
#         XQuAD has same questions/contexts in both EN and VI
#         """
#         logger.info("Loading XQuAD parallel datasets")
#         xquad_en = load_from_disk(str(xquad_en_path))
#         xquad_vi = load_from_disk(str(xquad_vi_path))
        
#         parallel_data = []
        
#         # Use validation split (XQuAD only has validation)
#         en_examples = list(xquad_en["validation"])
#         vi_examples = list(xquad_vi["validation"])
        
#         assert len(en_examples) == len(vi_examples), "XQuAD EN and VI must have same size"
        
#         for en_ex, vi_ex in zip(en_examples, vi_examples):
#             # Question pairs
#             parallel_data.append({
#                 "en": en_ex["question"],
#                 "vi": vi_ex["question"],
#                 "type": "question"
#             })
            
#             # Context pairs
#             parallel_data.append({
#                 "en": en_ex["context"],
#                 "vi": vi_ex["context"],
#                 "type": "context"
#             })
            
#             # Answer pairs
#             if en_ex["answers"]["text"] and vi_ex["answers"]["text"]:
#                 parallel_data.append({
#                     "en": en_ex["answers"]["text"][0],
#                     "vi": vi_ex["answers"]["text"][0],
#                     "type": "answer"
#                 })
        
#         logger.info(f"Created {len(parallel_data)} parallel sentence pairs")
        
#         # Split into train/validation (80/20)
#         split_idx = int(0.8 * len(parallel_data))
#         train_data = parallel_data[:split_idx]
#         val_data = parallel_data[split_idx:]
        
#         return DatasetDict({
#             "train": Dataset.from_list(train_data),
#             "validation": Dataset.from_list(val_data)
#         })
    
#     def preprocess_function(self, examples: Dict) -> Dict:
#         """
#         Preprocess for EN→VI translation
#         """
#         inputs = examples["en"]
#         targets = examples["vi"]
        
#         model_inputs = self.tokenizer(
#             inputs,
#             max_length=self.max_length,
#             truncation=True,
#             padding="max_length"
#         )
        
#         # Setup the tokenizer for targets
#         labels = self.tokenizer(
#             targets,
#             max_length=self.max_length,
#             truncation=True,
#             padding="max_length"
#         )
        
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
    
#     def compute_metrics(self, eval_preds):
#         """
#         Compute BLEU score
#         """
#         preds, labels = eval_preds
        
#         # Decode predictions
#         if isinstance(preds, tuple):
#             preds = preds[0]
        
#         # Replace -100 in labels (padding)
#         labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
#         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
#         # Format for BLEU
#         decoded_preds = [pred.strip() for pred in decoded_preds]
#         decoded_labels = [[label.strip()] for label in decoded_labels]
        
#         result = self.bleu_metric.compute(
#             predictions=decoded_preds,
#             references=decoded_labels
#         )
        
#         return {
#             "bleu": result["bleu"] * 100
#         }
    
#     def fine_tune(
#         self,
#         train_dataset: Dataset,
#         eval_dataset: Dataset,
#         output_dir: Path,
#         num_epochs: int = 3,
#         learning_rate: float = 3e-5,
#         batch_size: int = 8,
#         warmup_steps: int = 500,
#         save_steps: int = 500,
#         eval_steps: int = 500
#     ):
#         """
#         Fine-tune translation model
#         """
#         logger.info("Preprocessing datasets")
#         train_dataset = train_dataset.map(
#             self.preprocess_function,
#             batched=True,
#             remove_columns=train_dataset.column_names
#         )
        
#         eval_dataset = eval_dataset.map(
#             self.preprocess_function,
#             batched=True,
#             remove_columns=eval_dataset.column_names
#         )
        
#         # Data collator
#         data_collator = DataCollatorForSeq2Seq(
#             tokenizer=self.tokenizer,
#             model=self.model,
#             padding=True
#         )
        
#         # Training arguments
#         training_args = Seq2SeqTrainingArguments(
#             output_dir=str(output_dir),
#             eval_strategy="steps",
#             eval_steps=eval_steps,
#             save_strategy="steps",
#             save_steps=save_steps,
#             learning_rate=learning_rate,
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             num_train_epochs=num_epochs,
#             weight_decay=0.01,
#             warmup_steps=warmup_steps,
#             logging_steps=100,
#             load_best_model_at_end=True,
#             metric_for_best_model="bleu",
#             greater_is_better=True,
#             predict_with_generate=True,
#             generation_max_length=self.max_length,
#             generation_num_beams=4,
#             fp16=torch.cuda.is_available(),
#             report_to="none",
#             save_total_limit=3
#         )
        
#         # Trainer
#         trainer = Seq2SeqTrainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             tokenizer=self.tokenizer,
#             data_collator=data_collator,
#             compute_metrics=self.compute_metrics,
#             callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
#         )
        
#         # Train
#         logger.info("Starting training")
#         trainer.train()
        
#         # Save best model
#         best_model_path = output_dir / "best_model"
#         logger.info(f"Saving best model to {best_model_path}")
#         trainer.save_model(str(best_model_path))
#         self.tokenizer.save_pretrained(str(best_model_path))
        
#         # Evaluate
#         logger.info("Final evaluation")
#         metrics = trainer.evaluate()
        
#         logger.info("\n" + "="*60)
#         logger.info("Training completed!")
#         logger.info("="*60)
#         logger.info(f"Best BLEU score: {metrics['eval_bleu']:.2f}")
#         logger.info(f"Model saved to: {best_model_path}")
#         logger.info("="*60)
        
#         return metrics
    
#     def evaluate_baseline(self, eval_dataset: Dataset) -> float:
#         """
#         Evaluate baseline (before fine-tuning)
#         """
#         logger.info("Evaluating baseline model")
        
#         predictions = []
#         references = []
        
#         for example in eval_dataset:
#             pred = self.tokenizer.decode(
#                 self.model.generate(
#                     **self.tokenizer(
#                         example["en"],
#                         return_tensors="pt",
#                         max_length=self.max_length,
#                         truncation=True
#                     ).to(self.device),
#                     max_length=self.max_length,
#                     num_beams=4
#                 )[0],
#                 skip_special_tokens=True
#             )
#             predictions.append(pred.strip())
#             references.append([example["vi"].strip()])
        
#         result = self.bleu_metric.compute(
#             predictions=predictions,
#             references=references
#         )
        
#         bleu_score = result["bleu"] * 100
#         logger.info(f"Baseline BLEU: {bleu_score:.2f}")
        
#         return bleu_score


# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Fine-tune Translation Model")
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="Helsinki-NLP/opus-mt-en-vi",
#         help="Base translation model"
#     )
#     parser.add_argument(
#         "--xquad-en",
#         type=str,
#         default="datasets/processed/xquad_en_normalized",
#         help="XQuAD English dataset path"
#     )
#     parser.add_argument(
#         "--xquad-vi",
#         type=str,
#         default="datasets/processed/xquad_vi_normalized",
#         help="XQuAD Vietnamese dataset path"
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="fine_tuning/models/translation_finetuned",
#         help="Output directory"
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=3,
#         help="Number of training epochs"
#     )
#     parser.add_argument(
#         "--learning-rate",
#         type=float,
#         default=3e-5,
#         help="Learning rate"
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=8,
#         help="Batch size"
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#         help="Device"
#     )
#     parser.add_argument(
#         "--eval-baseline",
#         action="store_true",
#         help="Evaluate baseline before training"
#     )
    
#     args = parser.parse_args()
    
#     project_root = Path(__file__).parent.parent
#     xquad_en_path = project_root / args.xquad_en
#     xquad_vi_path = project_root / args.xquad_vi
#     output_dir = project_root / args.output
    
#     # Initialize fine-tuner
#     fine_tuner = TranslationFineTuner(
#         model_name=args.model,
#         device=args.device
#     )
    
#     # Create parallel corpus
#     parallel_corpus = fine_tuner.create_parallel_corpus_from_xquad(
#         xquad_en_path,
#         xquad_vi_path
#     )
    
#     logger.info(f"Train size: {len(parallel_corpus['train'])}")
#     logger.info(f"Validation size: {len(parallel_corpus['validation'])}")
    
#     # Evaluate baseline (optional)
#     if args.eval_baseline:
#         baseline_bleu = fine_tuner.evaluate_baseline(parallel_corpus["validation"])
    
#     # Fine-tune
#     output_dir.mkdir(parents=True, exist_ok=True)
#     metrics = fine_tuner.fine_tune(
#         train_dataset=parallel_corpus["train"],
#         eval_dataset=parallel_corpus["validation"],
#         output_dir=output_dir,
#         num_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         batch_size=args.batch_size
#     )
    
#     logger.info("\nFine-tuning completed!")
#     logger.info(f"Model saved to: {output_dir / 'best_model'}")
    
#     if args.eval_baseline:
#         improvement = metrics["eval_bleu"] - baseline_bleu
#         logger.info(f"BLEU improvement: +{improvement:.2f}")


# if __name__ == "__main__":
#     main()
