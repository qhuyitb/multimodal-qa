from datasets import load_dataset, DatasetDict, load_from_disk, concatenate_datasets
import os
from pathlib import Path
import json

project_root = Path(__file__).parent.parent

def normalize_dataset_format(dataset, language = "vi", dataset_name = "unknown") :
    """
    Chuẩn hóa format dataset về cấu trúc chung
    """
    def normalize_example(example):
        normalized = {
            'id': example['id'],
            'context': example['context'],
            'question': example['question'],
            'answers': example['answers'],
            'language': language,
            'dataset': dataset_name
        }
        if "title" in example:
            normalized['title'] = example['title']
        else:
            normalized['title'] = ""
        if "is_impossible" in example:
            normalized['is_impossible'] = example['is_impossible']
        else:
            normalized['is_impossible'] = False
        return normalized
    return dataset.map(normalize_example, remove_columns=dataset.column_names)
       
def create_unified_dataset():
    dataset_dir = project_root / "datasets" / "raw"
    processed_dir = project_root / "datasets" / "processed"
    os.makedirs(processed_dir, exist_ok = True)
    # load dataset
    squad = load_from_disk(dataset_dir / "squad")
    viquad = load_from_disk(dataset_dir / "viquad")
    xquad_vi = load_from_disk(dataset_dir / "xquad_vi")
    xquad_en = load_from_disk(dataset_dir / "xquad_en")
    
    # normalize
    squad_norm = DatasetDict({
        'train': normalize_dataset_format(squad['train'], language = 'en', dataset_name = 'squad'),
        'validation': normalize_dataset_format(squad['validation'], language = 'en', dataset_name = 'squad')
    })
    
    viquad_norm = DatasetDict({
        'train': normalize_dataset_format(viquad['train'], language = 'vi', dataset_name = 'viquad'),
        'validation': normalize_dataset_format(viquad['validation'], language = 'vi', dataset_name = 'viquad'),
        'test': normalize_dataset_format(viquad['test'], language = 'vi', dataset_name = 'viquad')
    })
    
    xquad_vi_norm = DatasetDict({
        'validation': normalize_dataset_format(xquad_vi['validation'], language = 'vi', dataset_name='xquad_vi'),
        
    })
    xquad_en_norm = DatasetDict({
        'validation': normalize_dataset_format(xquad_en['validation'], language = 'en', dataset_name = 'xquad_en')
    })
    
    # save normalize datasets
    squad_norm.save_to_disk(processed_dir / "squad_normalized")
    viquad_norm.save_to_disk(processed_dir / "viquad_normalized")
    xquad_vi_norm.save_to_disk(processed_dir / "xquad_vi_normalized")
    xquad_en_norm.save_to_disk(processed_dir / "xquad_en_normalized")
    
    print("Normalized datasets saved")
    
    print("\nCreating Vietnamese Unified Dataset")
    
    vi_train = viquad_norm['train']
    vi_validation = viquad_norm['validation']
    vi_test = concatenate_datasets([
        viquad_norm['test'],
        xquad_vi_norm['validation']
    ])
    
    unified_vi = DatasetDict({
        'train': vi_train,
        'validation': vi_validation,
        'test': vi_test
    })
    
    unified_vi.save_to_disk(processed_dir / "unified_vietnamese_qa")
    
    print(f"Vietnamese Unified:")
    print(f"Train:      {len(vi_train):,} samples")
    print(f"Validation: {len(vi_validation):,} samples")
    print(f"Test:       {len(vi_test):,} samples")
    
 
    print("\nCreating English Unified Dataset")
    
    en_train = squad_norm['train']
    en_validation = squad_norm['validation']
    en_test = xquad_en_norm['validation']
    
    unified_en = DatasetDict({
        'train': en_train,
        'validation': en_validation,
        'test': en_test
    })
    
    unified_en.save_to_disk(processed_dir / "unified_english_qa")
    
    print(f"English Unified:")
    print(f"Train:      {len(en_train):,} samples")
    print(f"Validation: {len(en_validation):,} samples")
    print(f"Test:       {len(en_test):,} samples")
    
    
    print("\nCreating Cross-lingual Test Set")
    
    xquad_parallel = DatasetDict({
        'en': xquad_en_norm['validation'],
        'vi': xquad_vi_norm['validation']
    })
    
    xquad_parallel.save_to_disk(processed_dir / "xquad_parallel_benchmark")
    
    print(f"Cross-lingual Benchmark:")
    print(f"EN: {len(xquad_parallel['en']):,} samples")
    print(f"VI: {len(xquad_parallel['vi']):,} samples")
    
    print("Dataset preprocessing completed!")

    
    return unified_vi, unified_en, xquad_parallel


if __name__ == "__main__":

    print("Dataset Normalization & Unification")

    
    create_unified_dataset()
