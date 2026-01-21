# Trích xuất video (audio -> text)
import whisper
import os
import json
import pathlib
import warnings
from src.utils.helpers import get_data_dir, get_project_root

warnings.filterwarnings("ignore")

project_root = get_project_root()
output_base_dir = get_data_dir("output/transcripts")
os.makedirs(output_base_dir, exist_ok=True)

def extract_video_text(video_path, output_dir=None):
    
    if output_dir is None:
        output_txt_dir = output_base_dir / "txt"
        output_json_dir = output_base_dir / "json"
    else:
        output_dir = pathlib.Path(output_dir)
        output_txt_dir = output_dir / "txt"
        output_json_dir = output_dir / "json"
    
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    
    model = whisper.load_model("medium")
    result = model.transcribe(
        str(video_path),
        task="transcribe",
        fp16=False,
        verbose=True
    )
    name = pathlib.Path(video_path).stem
    txt_path = output_txt_dir / f"{name}.txt"
    json_path = output_json_dir / f"{name}.json"
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    return result["text"]

if __name__ == "__main__":
    demo_video = get_data_dir("input/videos/mp4/video_demo.mp4")
    extract_video_text(demo_video, output_base_dir)
    # pass
    