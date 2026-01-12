from datasets import load_dataset
import os
import pathlib

project_root = pathlib.Path(__file__).parent.parent
datasets_dir = project_root / "datasets" / "raw"
os.makedirs(datasets_dir, exist_ok=True)

# 1. SQuAD 2.0 
print("\nDownload SQuAD 2.0")
squad = load_dataset('squad_v2')
squad.save_to_disk(datasets_dir / "squad")
print("Download SQuAD complete ")

# 2. UIT-ViQuAD 2.0 
print("\nDownload ViQuAD 2.0")
viquad = load_dataset('taidng/UIT-ViQuAD2.0')
# Xóa trường uit_id 
viquad = viquad.remove_columns(['uit_id'])
viquad.save_to_disk(datasets_dir / "viquad")
print("Download ViQuAD complete ")


# 3. XQuAD 
print("\nDownload XQuAD")
xquad_vi = load_dataset('xquad', 'xquad.vi')
xquad_en = load_dataset('xquad', 'xquad.en')
xquad_vi.save_to_disk(datasets_dir / "xquad_vi")
xquad_en.save_to_disk(datasets_dir / "xquad_en")
print("Download XQuAD complete ")

