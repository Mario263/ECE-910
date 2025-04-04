import cv2
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import argparse
import torch
import numpy as np
from PIL import Image
import json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor ONCE to avoid reloading
model_id = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_id)
model = SiglipVisionModel.from_pretrained(model_id).to(device).half().eval()  # Use fp16 precision to reduce memory usage



def write_sets_to_json_incremental(sets_list, output_file):
    """
    Writes a list of sets to a JSON file incrementally.
    It writes each set one by one without keeping everything in memory.

    :param sets_list: Iterable (or generator) of sets containing numbers or strings.
    :param output_file: Output JSON filename.
    """
    with open(output_file, "w") as file:
        file.write("[\n")  # Start JSON array
        first_entry = True
        
        for s in sets_list:
            if not first_entry:
                file.write(",\n")  # Add comma between entries
                
            json.dump(sorted(list(s)), file)  # Convert set to a list and write
            first_entry = False
        
        file.write("\n]")  # Close JSON array

@torch.no_grad()
def extract_frames_to_pil(video_path, start_frame, end_frame, resize=(128, 128)):
    """ Extracts frames from a video using OpenCV. Faster than ffmpeg and resize to reduce memory usage. """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, resize)  # Resize frame to reduce memory usage
        frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

@torch.no_grad()
def generate_frame_vectors(frames, batch_size=8):
    """ Generate feature vectors for a batch of frames. """
    frame_vectors = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        try:
            inputs = processor(images=batch, return_tensors="pt").to(device, non_blocking=True)
            inputs = {k: v.half() for k, v in inputs.items()}  # Convert inputs to fp16
            outputs = model(**inputs)
            frame_vectors.append(outputs.pooler_output.detach().cpu())  # Move to CPU after computation
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM! Reducing batch size...")
            torch.cuda.empty_cache()  # Free memory
            return generate_frame_vectors(frames, batch_size=max(1, batch_size // 2))  # Try smaller batch
    
    return torch.cat(frame_vectors, dim=0)

def get_key_shots_from_shot(video_path, shots_file, threshold=0.95):
    df = pd.read_csv(shots_file)
    
    for index, row in tqdm(df.iterrows(), desc="Processing Shots", total=len(df)):
        tqdm.write(f"{index}")
        start, end = row["Start"], row["End"]
        frames = extract_frames_to_pil(video_path, start, end)
        frame_vectors = []
        try:
            frame_vectors = generate_frame_vectors(frames)  # Compute all embeddings at once
        except:
            tqdm.write(f"skipping {index+1}")
            continue
        key_shots = set()
        key_shot_dict = dict()
        first = 0
        for i in range(1, len(frames)):
            similarity = torch.nn.functional.cosine_similarity(frame_vectors[first], frame_vectors[i], dim=0).item()

            if similarity < threshold:
                key_shots.add(start + i)
                first = i
        key_shots.add(start+(len(frames)-1))
        key_shot_dict["index"]= index
        key_shot_dict["frames"]= list(key_shots)
        yield key_shot_dict

        
def write_data_incrementally_to_json(data_iterable, filename="processed_data_1.json"):
    with open(filename, 'w') as f:
        f.write('[\n')  # Start the JSON array
        first = True
        for data_dict in tqdm(data_iterable,desc="Writing Data to File"):
            tqdm.write(str(data_dict))
            if not first:
                f.write(',\n')  # Add a comma between JSON objects
            json.dump(data_dict, f)
            first = False
        f.write('\n]')  # End the JSON array

if __name__ == "__main__":
    print(f"Using Device: {device}")
    parser = argparse.ArgumentParser(description="Key Shot Generator")
    parser.add_argument('--video', type=str, required=True, help="Path to video")
    parser.add_argument('--threshold', type=float, default=0.95, help="Cosine Similarity Threshold")
    parser.add_argument('--shots', type=str, required=True, help="Path to shots CSV file")
    
    args = parser.parse_args()
    
    write_data_incrementally_to_json(get_key_shots_from_shot(args.video, args.shots, args.threshold),"key_shots.json")
        
