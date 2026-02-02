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
from pathlib import Path


class XLMRobertaQA:
    
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        if use_lora:
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
        top_k: int = 1,
        max_answer_length: int = 50,
        n_best_size: int = 20
    ) -> List[Dict[str, any]]:
        # Inference cho 1 question-context pair
        self.model.eval()  # Ensure eval mode
        
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
        
        # Lấy sequence_ids để xác định context tokens
        sequence_ids = inputs.sequence_ids(0)
        context_start = sequence_ids.index(1) if 1 in sequence_ids else 0
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) if 1 in sequence_ids else len(sequence_ids)
        
        # Top-k answers - tối ưu hơn với n_best_size
        results = []
        start_indexes = torch.argsort(start_logits, descending=True)[:n_best_size]
        end_indexes = torch.argsort(end_logits, descending=True)[:n_best_size]
        
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                # Skip invalid spans
                if end_idx < start_idx or end_idx - start_idx > max_answer_length:
                    continue
                
                # Chỉ cho phép answers từ context (sequence_id == 1)
                if start_idx < context_start or end_idx > context_end:
                    continue
                
                answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
                answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Skip empty answers
                if not answer_text.strip():
                    continue
                
                # Tính score theo product (better than sum)
                score = (start_logits[start_idx] * end_logits[end_idx]).item()
                
                results.append({
                    "text": answer_text.strip(),
                    "score": score,
                    "start": start_idx.item(),
                    "end": end_idx.item()
                })
                
                if len(results) >= top_k * 3:  # Get more candidates
                    break
            if len(results) >= top_k * 3:
                break
        
        # Nếu không tìm thấy answer nào, return best guess từ context
        if not results:
            # Tìm span có score cao nhất trong context
            best_score = float('-inf')
            best_start, best_end = context_start, context_start
            
            for i in range(context_start, min(context_end, context_start + 50)):
                for j in range(i, min(i + max_answer_length, context_end)):
                    score = (start_logits[i] * end_logits[j]).item()
                    if score > best_score:
                        best_score = score
                        best_start, best_end = i, j
            
            answer_tokens = inputs["input_ids"][0][best_start:best_end + 1]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            results.append({
                "text": answer_text,
                "score": best_score,
                "start": best_start,
                "end": best_end
            })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    
    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load(self, model_dir: Path):
        model_dir = Path(model_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.to(self.device)


def create_qa_model(
    model_name: str = "xlm-roberta-base",
    checkpoint_path: Optional[Path] = None,
    **kwargs
) -> XLMRobertaQA:
    # Factory function để tạo QA model
    # Nếu có checkpoint_path, ưu tiên dùng nó làm model_name để tránh load 2 lần
    if checkpoint_path and Path(checkpoint_path).exists():
        return XLMRobertaQA(model_name=str(checkpoint_path), **kwargs)
    
    return XLMRobertaQA(model_name=model_name, **kwargs)
