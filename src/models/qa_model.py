from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class XLMRobertaQA:
    # Wrapper cho XLM-RoBERTa QA model với LoRA support
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Setup LoRA nếu cần
        if use_lora:
            logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}")
            lora_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "value", "key"],
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model.to(self.device)
    
    def prepare_train_features(
        self,
        examples: Dict,
        max_length: int = 384,
        doc_stride: int = 128
    ) -> Dict:
        # Tokenize context và question
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Map answer positions sang token positions
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            # Nếu không có answer (is_impossible)
            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
                continue
            
            # Lấy answer đầu tiên
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Tìm token start/end
            token_start_index = 0
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1
            
            token_end_index = len(offsets) - 1
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1
            
            # Kiểm tra answer có nằm trong context không
            sequence_ids = tokenized.sequence_ids(i)
            context_start = sequence_ids.index(1) if 1 in sequence_ids else 0
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) if 1 in sequence_ids else len(sequence_ids)
            
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            elif token_start_index < context_start or token_end_index > context_end:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                tokenized["start_positions"].append(token_start_index)
                tokenized["end_positions"].append(token_end_index)
        
        return tokenized
    
    def prepare_validation_features(
        self,
        examples: Dict,
        max_length: int = 384,
        doc_stride: int = 128
    ) -> Dict:
        # Tương tự train nhưng giữ example_id và offset_mapping để post-process
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        
        tokenized["example_id"] = []
        for i in range(len(tokenized["input_ids"])):
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])
            
            # Set context mask
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        
        return tokenized
    
    def predict(
        self,
        question: str,
        context: str,
        top_k: int = 1
    ) -> List[Dict[str, any]]:
        # Inference cho 1 question-context pair
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get answer spans
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        
        # Top-k answers
        results = []
        start_indexes = torch.argsort(start_logits, descending=True)[:top_k * 2]
        end_indexes = torch.argsort(end_logits, descending=True)[:top_k * 2]
        
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if end_idx < start_idx or end_idx - start_idx > 30:
                    continue
                
                answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
                answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                score = (start_logits[start_idx] + end_logits[end_idx]).item()
                
                results.append({
                    "text": answer_text,
                    "score": score,
                    "start": start_idx.item(),
                    "end": end_idx.item()
                })
                
                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def save(self, output_dir: Path):
        # Lưu model và tokenizer
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    def load(self, model_dir: Path):
        # Load model và tokenizer từ checkpoint
        model_dir = Path(model_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_dir}")


def create_qa_model(
    model_name: str = "xlm-roberta-base",
    checkpoint_path: Optional[Path] = None,
    **kwargs
) -> XLMRobertaQA:
    # Factory function để tạo QA model
    model = XLMRobertaQA(model_name=model_name, **kwargs)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load(checkpoint_path)
    
    return model
