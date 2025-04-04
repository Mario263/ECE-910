import os
import pandas as pd
import argparse
import ffmpeg
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import ffmpeg

def write_data_incrementally_to_json(data_iterable, filename="limited_processed_data.json"):
    with open(filename, 'w') as f:
        f.write('[\n')  # Start the JSON array
        first = True
        count = 0  # Track how many items are written
        
        for data_dict in tqdm(data_iterable, desc="Writing Data to File"):
            if not first:
                f.write(',\n')  # Add a comma between JSON objects
            json.dump(data_dict, f)
            first = False
            count += 1

        f.write('\n]')  # End the JSON array
    
    print(f"📌 Total items written to JSON: {count}")
        
def extract_frames_to_pil(video_path, start_frame, end_frame):
    probe = ffmpeg.probe(video_path)
    fps = eval(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["r_frame_rate"])
    width = int(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["width"])
    height = int(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["height"])
    start_time = start_frame / fps
    end_time = end_frame / fps
    out, _ = (
        ffmpeg
        .input(video_path, ss=start_time, to=end_time)  # Trim video
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")  # Output raw RGB frames
        .run(capture_stdout=True, capture_stderr=True)  # Capture in memory
    )
    frame_size = width * height * 3
    frames = [
        Image.fromarray(np.frombuffer(out[i:i+frame_size], np.uint8).reshape((height, width, 3)), 'RGB')
        for i in range(0, len(out), frame_size)
    ]
    return frames

def extratct_keyshots(directory):
    keyshots = []

    if not os.path.exists(directory):
        print(f"❌ Missing directory: {directory}")
        return []

    print(f"✅ Checking directory: {directory}")  # Debugging output

    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            keyshots.append(os.path.join(directory, file))

    if not keyshots:
        print(f"⚠️ No images found in: {directory}")  # Debugging output
    else:
        # Sort keyshots numerically based on their filename
        keyshots.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

        print(f"✅ Found and sorted images in {directory}: {keyshots}")  # Debugging output

    return keyshots
    
def process_data(data_file, user_prompt="What is this?"):
    data_file_df = pd.read_csv(data_file)
    
    print("CSV Columns:", data_file_df.columns)  
    print("First 5 rows of CSV:\n", data_file_df.head())  # Print first few rows

    found_data = False  # Track if we yield any data
    
    for index, row in tqdm(data_file_df.iterrows(), desc="Creating Data Set"):
        frames = []
        clip_dir = f"dyna/clip-{index}"

        if not os.path.exists(clip_dir):
            print(f"Skipping missing directory: {clip_dir}")
            continue

        try:
            frames = extratct_keyshots(clip_dir)  
        except Exception as e:
            print(f"Error processing {clip_dir}: {e}")
            continue

        if len(frames) == 0:
            print(f"Skipping index {index} - No frames found.")
            continue

        
        row_action = row.get(" Action", "No Action Found").strip()  

        toks = "<image>" * (len(sorted(frames[:10])))
        prompt = f"<|im_start|>user{toks}\n{user_prompt}<|im_end|><|im_start|>assistant {row_action} <|im_end|>"

        data_dict = {"prompt": prompt, "frames": frames[:10]}

        print(f"Writing data for clip-{index}: {data_dict}")  # Debug output
        found_data = True
        yield data_dict
    
    if not found_data:
        print("No data was yielded from process_data(). Check earlier logs.")
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Video DataSet Preparator")
    parser.add_argument('--data', type=str, help="CSV data path")
    args = parser.parse_args()
    data_path = args.data
    prompt = """
    Classify the given video into the following actions:
    1. Rescue
    2. Escape 
    3. Capture 
    4. Heist
    5. Fight
    6. Pursuit
    7. None of the Above - For scenes that do not fall into any of the aforementioned categories.
    Your Reply can include multiple categories if possible. For example, a scene can have Rescue and Escape.
    """
    write_data_incrementally_to_json(process_data(data_path,prompt))
    
