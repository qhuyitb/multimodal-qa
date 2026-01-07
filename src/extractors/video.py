import whisper
import os
import json
import pathlib
import warnings
warnings.filterwarnings("ignore")
output_dir = "data/output/transcripts"
os.makedirs(output_dir, exist_ok=True)

def extract_video_text(video_path, output_dir):
    model = whisper.load_model("medium")
    result = model.transcribe(
        video_path,
        task="transcribe",
        fp16 =False,
        verbose=True
    )
    name = pathlib.Path(video_path).stem
    txt_path = f"{output_dir}/{name}.txt"
    json_path = f"{output_dir}/{name}.json"
    
    with open(txt_path, "w") as f:
        f.write(result["text"])
    with open(json_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    return result["text"]

if __name__ == "__main__":
    pass
    