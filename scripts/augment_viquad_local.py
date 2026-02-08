"""
Simple and Effective ViQuAD Data Augmentation
Uses CONTEXT-AWARE synonym replacement only (no noisy back-translation)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk, DatasetDict, Dataset
from typing import List, Dict
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


VIETNAMESE_SYNONYMS = {
    # Question words 
    "gì": ["thứ gì", "điều gì", "phần nào"],  
    "ai": ["người nào", "nhân vật nào"],
    "khi nào": ["lúc nào", "thời điểm nào", "thời gian nào"],
    "ở đâu": ["tại đâu", "nơi nào", "địa điểm nào"],
    "như thế nào": ["ra sao", "thế nào", "bằng cách nào"],
    "tại sao": ["vì sao", "do đâu", "lý do gì"],
    "bao nhiêu": ["mấy"],
    
    # Common verbs - CONTEXT-AWARE only
    "giữ": ["đảm nhận", "nắm giữ", "đảm nhiệm"],  # For "giữ chức vụ"
    "làm": ["thực hiện", "đảm nhận", "tiến hành"],
    "sử dụng": ["dùng", "sử dụng"],
    "tạo": ["lập", "thành lập", "hình thành"],
    "thành lập": ["lập", "hình thành", "sáng lập"],
    "diễn ra": ["xảy ra", "tổ chức"],
    "kết thúc": ["chấm dứt", "hoàn thành", "kết thúc"],
    "bắt đầu": ["khởi đầu", "bắt đầu"],
    
    # Adjectives - SAFE
    "quan trọng": ["thiết yếu", "trọng yếu", "then chốt", "chính yếu"],
    "đặc biệt": ["riêng biệt", "nổi bật", "độc đáo"],
    "chính": ["chủ yếu", "căn bản", "cơ bản"],
    "nhiều": ["rất nhiều", "đa số", "phần lớn"],
    "lớn": ["to", "khổng lồ", "đồ sộ"],
    "nhỏ": ["nhỏ bé", "bé"],
    "nhanh": ["nhanh chóng", "mau lẹ"],
    "chậm": ["chậm chạp", "lâu"],
    
    # Nouns - DOMAIN SPECIFIC
    "chức vụ": ["cương vị", "vị trí"],
    "tên gọi": ["danh xưng", "tên"],
    "sự kiện": ["biến cố", "sự việc", "tình huống"],
    "thời gian": ["thời điểm", "khoảng thời gian"],
    "tác phẩm": ["công trình", "sáng tác"],
    "triều đại": ["triều đình", "nhà"],
    "hoàng đế": ["vua", "quân chủ"],
    "chiến tranh": ["cuộc chiến", "trận chiến"],
    
    # Conjunctions - SAFE
    "và": ["cùng", "với", "cùng với"],
    "hoặc": ["hay"],
    "nhưng": ["song", "tuy nhiên"],
    "vì": ["do", "bởi vì"],
}

# Blacklist 
BLACKLIST_WORDS = {
    "được",  # "được" → "đạt được" is WRONG in "được sử dụng"
    "có",    # "có" → "sở hữu" is WEIRD
    "đến",   # Context dependent
    "là",    
    "của",  
    "trong", # Preposition
    "trên", "dưới", "sau", "trước",  # Positional
}


class ContextAwareQADataAugmenter:
    """Context-aware QA augmentation - only safe replacements"""
    
    def __init__(self, synonym_dict, seed=42):
        self.synonym_dict = synonym_dict
        random.seed(seed)
    
    def is_safe_to_replace(self, word: str, position: int, total_words: int) -> bool:
        """Check if it's safe to replace this word"""
        word_lower = word.lower().strip(".,!?;:\"'")
        
        if word_lower in BLACKLIST_WORDS:
            return False
        
        if position > 0 and word[0].isupper():
            return False
        
        if word_lower not in self.synonym_dict:
            return False
        
        return True
    
    def synonym_replacement(
        self, 
        text: str, 
        n: int = 1,
        preserve_first_word: bool = True
    ) -> str:
        """
        Replace n words with context-aware synonyms
        
        Args:
            text: Input text
            n: Max words to replace (actual may be less)
            preserve_first_word: Don't replace question word
        """
        words = text.split()
        if len(words) == 0:
            return text
        
        new_words = words.copy()
        
        # Find safe replaceable words
        replaceable_indices = []
        start_idx = 1 if preserve_first_word else 0
        
        for i in range(start_idx, len(words)):
            if self.is_safe_to_replace(words[i], i, len(words)):
                replaceable_indices.append(i)
        
        if not replaceable_indices:
            return text
        
        # Replace at most n words
        n = min(n, len(replaceable_indices))
        if n == 0:
            return text
        
        replace_indices = random.sample(replaceable_indices, n)
        
        for idx in replace_indices:
            word = words[idx]
            word_lower = word.lower().strip(".,!?;:\"'")
            
            if word_lower in self.synonym_dict:
                synonym = random.choice(self.synonym_dict[word_lower])
                
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                
                # Preserve punctuation
                punct = ""
                if word[-1] in ".,!?;:\"'":
                    punct = word[-1]
                    synonym = synonym + punct
                
                new_words[idx] = synonym
        
        return " ".join(new_words)
    
    def augment_question_only(
        self, 
        question: str,
        num_variations: int = 2
    ) -> List[str]:
        """
        Generate multiple augmented variations
        
        Strategy:
        - Variation 1: Replace 1 word
        - Variation 2: Replace 2 words
        - Variation 3: Replace 1-2 words (different random)
        """
        augmented = []
        attempts = 0
        max_attempts = num_variations * 3
        
        while len(augmented) < num_variations and attempts < max_attempts:
            attempts += 1
            n = random.randint(1, 2)  # Replace 1-2 words
            aug = self.synonym_replacement(question, n=n)
            
            # Only add if it's different and not duplicate
            if aug != question and aug not in augmented:
                augmented.append(aug)
        
        return augmented[:num_variations]
    
    def augment_example(
        self,
        example: Dict,
        num_augmentations: int = 2
    ) -> List[Dict]:
        """Augment single example - only question, preserve context & answer!"""
        augmented_questions = self.augment_question_only(
            example["question"],
            num_variations=num_augmentations
        )
        
        augmented_examples = []
        for i, aug_q in enumerate(augmented_questions):
            aug_example = example.copy()
            aug_example["question"] = aug_q
            aug_example["id"] = f"{example['id']}_aug{i+1}"
            augmented_examples.append(aug_example)
        
        return augmented_examples


def augment_viquad_local(
    input_path: str,
    output_path: str,
    num_augmentations: int = 2,
    test_mode: bool = False
):
    """
    Augment ViQuAD dataset locally
    
    Args:
        input_path: Path to normalized ViQuAD
        output_path: Output path
        num_augmentations: Augmentations per example (1-2 recommended)
        test_mode: If True, only process 100 examples for testing
    """
    logger.info("="*70)
    logger.info("ViQuAD Data Augmentation - Context-Aware Synonym Replacement")
    logger.info("="*70)
    
    # Load dataset
    logger.info(f"Loading dataset from: {input_path}")
    dataset = load_from_disk(input_path)
    
    logger.info(f"Original sizes:")
    logger.info(f"  Train: {len(dataset['train']):,}")
    logger.info(f"  Validation: {len(dataset['validation']):,}")
    logger.info(f"  Test: {len(dataset['test']):,}")
    logger.info("")
    
    # Initialize augmenter
    augmenter = ContextAwareQADataAugmenter(VIETNAMESE_SYNONYMS)
    
    # Augment training data
    original_examples = list(dataset['train'])
    
    if test_mode:
        logger.info("TEST MODE: Processing only 100 examples")
        original_examples = original_examples[:100]
    
    logger.info(f"Augmenting {len(original_examples):,} examples...")
    logger.info(f"Augmentations per example: {num_augmentations}")
    logger.info("")
    
    augmented_data = []
    
    for example in tqdm(original_examples, desc="Augmenting"):
        # Keep original
        augmented_data.append(example)
        
        # Add augmented versions
        aug_examples = augmenter.augment_example(
            example,
            num_augmentations=num_augmentations
        )
        augmented_data.extend(aug_examples)
    
    # Create augmented dataset
    augmented_dataset = DatasetDict({
        "train": Dataset.from_list(augmented_data),
        "validation": dataset["validation"],
        "test": dataset["test"]
    })
    
    # Save
    logger.info(f"\nSaving to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    augmented_dataset.save_to_disk(output_path)
    
    # Statistics
    logger.info("\n" + "="*70)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("="*70)
    
    for split in augmented_dataset.keys():
        orig_size = len(dataset[split])
        aug_size = len(augmented_dataset[split])
        increase = aug_size - orig_size
        pct = (increase / orig_size * 100) if orig_size > 0 else 0
        logger.info(f"{split:12s}: {orig_size:6,} → {aug_size:6,} (+{increase:6,}, +{pct:.1f}%)")
    
    logger.info("="*70)
    logger.info(f"Output: {output_path}")
    logger.info("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment ViQuAD with context-aware synonyms")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/processed/viquad_normalized",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/processed/viquad_augmented",
        help="Output dataset path"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=2,
        help="Augmentations per example (1-2 recommended)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - only process 100 examples"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_path = str(project_root / args.input)
    output_path = str(project_root / args.output)
    
    augment_viquad_local(
        input_path=input_path,
        output_path=output_path,
        num_augmentations=args.num_augmentations,
        test_mode=args.test
    )


if __name__ == "__main__":
    main()
