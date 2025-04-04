import os
import pandas as pd
import argparse
import ffmpeg
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import cv2

def extract_frame(video_path, frame_number):
    """
    Extracts a specific frame from a video given its frame number.
    
    :param video_path: Path to the video file.
    :param frame_number: Frame number to extract.
    :return: PIL Image of the extracted frame, or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Move to the desired frame
    
    ret, frame = cap.read()  # Read the frame
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
        return Image.fromarray(frame)
    else:
        print(f"Error: Could not extract frame {frame_number}")
        return None

def load_from_json(json_file):
    """
    Loads a JSON file containing a list of sets (stored as lists).
    
    :param json_file: Path to the JSON file.
    :return: List of sets.
    """
    data = None
    with open(json_file, "r") as file:
        data = json.load(file)
        
    return data

def write_data_incrementally_to_json(data_iterable, filename="processed_data_Abhi.json"):
    with open(filename, 'w') as f:
        f.write('[\n')  # Start the JSON array
        first = True
        for data_dict in tqdm(data_iterable,desc="Writing Data to File"):
            if not first:
                f.write(',\n')  # Add a comma between JSON objects
            json.dump(data_dict, f)
            first = False
        f.write('\n]')  # End the JSON array
        
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
    
def process_data(video_path,data_file,shots_file,window=10,user_prompt="What is this?"):
    data_file_df = pd.read_csv(data_file)
    shots_file_df = pd.read_csv(shots_file)
    new_df = pd.concat([data_file_df,shots_file_df],axis=1)
    new_df["Action Formatted"] = new_df["Action"].fillna("No Action Found")
    os.makedirs("/storage/abhi/keyshot_frames",exist_ok=True)
    frame_number = 0
    for index, row in tqdm(new_df.iterrows(),desc="Creating Data Set"):
        shot= row["Shot"]
        start =  0 if (shot-window) < 0 else (shot-window)
        end = shot+window
        frames = extract_frames_to_pil(video_path,start,end)
        toks = "<image>" * (len(frames))
        prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant "+row["Action Formatted"]+ "<|im_end|>"
        image_paths = []
        for frame in frames:
            frame.save(f"/storage/abhi/keyshot_frames_abhi/frame_{frame_number}.jpg")
            image_paths.append(f"/storage/abhi/keyshot_frames_abhi/frame_{frame_number}.jpg")
            frame_number = frame_number + 1
        data_dict = dict()
        data_dict["prompt"] = prompt
        data_dict["frames"] = image_paths
        yield data_dict

def process_data_alternative(video_path,data_file,shots_file,user_prompt="What is this?"):
    data_file_df = pd.read_csv(data_file)
    key_shots = load_from_json(shots_file)
    data_file_df["Action Formatted"] = data_file_df["Action"].fillna("No Action Found")
    os.makedirs("/storage/abhi/keyshot_frames_2",exist_ok=True)
    frame_number = 0
    for index, row in tqdm(data_file_df.iterrows(),desc="Creating Data Set"):
        image_paths = []
        key_shot_entry = None
        try:     
            key_shot_entry = key_shots[index]
        except:
            continue
        frames = []
        for frame in key_shot_entry["frames"]:
            frames.append(extract_frame(video_path,frame))
        for frame in frames:
            frame.save(f"/storage/abhi/keyshot_frames_2/frame_{frame_number}.jpg")
            image_paths.append(f"/storage/abhi/keyshot_frames_2/frame_{frame_number}.jpg")
            frame_number = frame_number + 1
        toks = "<image>" * (len(frames))
        prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant "+row["Action Formatted"]+ "<|im_end|>"
        data_dict = dict()
        data_dict["prompt"] = prompt
        data_dict["frames"] = image_paths
        yield data_dict
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Video DataSet Preparator")
    parser.add_argument('--video', type=str, help="Video path")
    parser.add_argument('--data', type=str, help="CSV data path")
    parser.add_argument('--shots',type=str, help="Shots CSV data path")
    parser.add_argument('--window',type=int, help="Number of frames you wanna annotate")
    parser.add_argument("--alternative", action="store_true", 
                    help="Use the alternative method for keyshots")
    args = parser.parse_args()
    video_path = args.video
    data_path = args.data
    shots_path = args.shots
    window = args.window
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
    if not args.alternative:
        write_data_incrementally_to_json(process_data(video_path,data_path,shots_path,window,prompt))
    else:
        write_data_incrementally_to_json(process_data_alternative(video_path,data_path,shots_path,prompt))
    
    
    