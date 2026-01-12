from datasets import load_dataset, load_from_disk
import os
import pathlib
import json


def check_squad_datasets(datasets_dir):
    squad_dir = datasets_dir / "squad"
    squad = load_from_disk(squad_dir)
    print(f"SQuAD dataset loaded from {squad_dir}")
    print(f"Number of training samples: {len(squad['train'])}")
    print(f"Number of validation samples: {len(squad['validation'])}")
    print("Sample training example:")
    sample = squad['train'][0]
    print(f"id : {sample['id']}")
    print(f"title {sample['title'][:80]}")
    print(f"context: {sample['context'][:80]}")
    print(f"question: {sample['question'][:80]}")
    print(f"answers_text: {', '.join(map(str, sample['answers']['text']))}, answer_start: {', '.join(map(str, sample['answers']['answer_start']))}")

    sample = squad['train'][0]
    print("Sample training example:")
    print("max question lenth:", max(len(sample['question'].split()) for sample in squad['train']))
    print("min question lenth:", min(len(sample['question'].split()) for sample in squad['train' ]))
    question_lengths = [len(sample['question'].split()) for sample in squad['train']]
    avg_length = sum(question_lengths) / len(question_lengths)
    print(f"Average question length: {avg_length:.2f} words")

def check_viquad_datasets(datasets_dir):
    viquad_dir = datasets_dir / "viquad"
    viquad = load_from_disk(viquad_dir)
    print("\n")
    print(f"ViQuAD dataset loaded from {viquad_dir}")
    print(f"Number of training samples: {len(viquad['train'])}")
    print(f"Number of validation samples: {len(viquad['validation'])}")
    print("Sample training example:")
    sample = viquad['train'][0]
    print(f"id : {sample['id']}")
    print(f"title {sample['title'][:80]}")
    print(f"context: {sample['context'][:80]}")
    print(f"question: {sample['question'][:80]}")
    print(f"answers_text: {', '.join(map(str, sample['answers']['text']))}, answer_start: {', '.join(map(str, sample['answers']['answer_start']))}")
    
    # Lấy 1000 samples đầu tiên và tính độ dài câu hỏi
    train_samples = viquad['train'].select(range(min(1000, len(viquad['train']))))
    question_lengths = [len(sample['question'].split()) for sample in train_samples]
    avg_length = sum(question_lengths) / len(question_lengths)
    print(f"Average question length (first 1000 samples): {avg_length:.2f} words")  
    print(f"Max question length (first 1000 samples): {max(question_lengths)} words ")
    print(f"Min question length (first 1000 samples): {min(question_lengths)} words ")


def check_xquad_datasets(datasets_dir):
    xquad_vi_dir = datasets_dir / "xquad_vi"
    xquad_en_dir = datasets_dir / "xquad_en"
    
    xquad_vi = load_from_disk(xquad_vi_dir)
    xquad_en = load_from_disk(xquad_en_dir)
    print("\n")
    print(f"XQuAD Vietnamese dataset loaded from {xquad_vi_dir}")
    print(f"Number of validation samples: {len(xquad_vi['validation'])}")
    
    print(f"XQuAD English dataset loaded from {xquad_en_dir}")
    print(f"Number of validation samples: {len(xquad_en['validation'])}")
    
    print("Sample Vietnamese validation example:")
    sample_vi = xquad_vi['validation'][0]
    print(f"id : {sample_vi['id']}")
    print(f"context: {sample_vi['context'][:80]}")
    print(f"question: {sample_vi['question'][:80]}")
    print(f"answers_text: {', '.join(map(str, sample_vi['answers']['text']))}, answer_start: {', '.join(map(str, sample_vi['answers']['answer_start']))}")
    print("\nSample English validation example:")
    
    
    sample_en = xquad_en['validation'][0]
    print(f"id : {sample_en['id']}")
    print(f"context: {sample_en['context'][:80]}")
    print(f"question: {sample_en['question'][:80]}")
    print(f"answers_text: {', '.join(map(str, sample_en['answers']['text']))}, answer_start: {', '.join(map(str, sample_en['answers']['answer_start']))}")
    
if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parent.parent
    datasets_dir = project_root / "datasets" / "raw"
    
    check_squad_datasets(datasets_dir)
    check_viquad_datasets(datasets_dir)
    check_xquad_datasets(datasets_dir)