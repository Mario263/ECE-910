import pandas as pd
import cv2
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, AutoImageProcessor, SiglipForImageClassification,pipeline
import torch
import argparse
import csv
from tqdm import tqdm
import os
import ffmpeg
import numpy as np
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"




def bilinear_interpolation(image, new_width, new_height):
    """
    Resize an image using bilinear interpolation.
    :param image: NumPy array (H x W x C)
    :param new_width: Target width
    :param new_height: Target height
    :return: Resized image as NumPy array
    """
    height, width, channels = image.shape
    x_scale_factor = width / new_width
    y_scale_factor = height / new_height

    result = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = x * x_scale_factor
            src_y = y * y_scale_factor

            x1 = int(src_x)
            y1 = int(src_y)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            alpha = src_x - x1
            beta = src_y - y1

            for c in range(channels):
                result[y, x, c] = (
                    (1 - alpha) * (1 - beta) * image[y1, x1, c]
                    + alpha * (1 - beta) * image[y1, x2, c]
                    + (1 - alpha) * beta * image[y2, x1, c]
                    + alpha * beta * image[y2, x2, c]
                )
    
    return result

def extract_frame(video_path, frame_number, resize_to=(224,224)):
    """
    Extracts a specific frame from a video given its frame number and resizes it using bilinear interpolation.

    :param video_path: Path to the video file.
    :param frame_number: Frame number to extract.
    :param resize_to: Tuple (width, height) to resize the image to.
    :return: PIL Image of the extracted and resized frame, or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3)

        if resize_to:
            resized = bilinear_interpolation(frame, resize_to[0], resize_to[1])
            resized = np.clip(resized, 0, 255).astype(np.uint8)  # ✅ ensure valid range and dtype
            return Image.fromarray(resized)
        else:
            return Image.fromarray(frame)
    else:
        print(f"Error: Could not extract frame {frame_number}")
        return None


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

def annotate_key_shots(key_shots,video_path,model_id="llava-hf/llava-interleave-qwen-7b-hf",user_prompt = "What is this scene about?"):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,load_in_4bit=True)
    model.to("cuda")
    toks = "<image>"
    prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i in key_shots:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(model.device, model.dtype)
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            output_decoded = processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:]
            yield (pil_image,output_decoded)

def annotate_around_key_shots(key_shots,video_path,model_id="llava-hf/llava-interleave-qwen-7b-hf",processor_id="llava-hf/llava-interleave-qwen-0.5b-hf",user_prompt="What is this scene about?",window=60):
    processor = LlavaProcessor.from_pretrained(processor_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,load_in_4bit=True)
    model.to("cuda")
    for shot in key_shots:
        frames = extract_frames_to_pil(video_path,shot-window,shot+window)
        toks = "<image>" * (len(frames))
        prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
        inputs = processor(text=prompt,images=frames,return_tensors="pt").to(model.device, model.dtype)
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output_decoded = processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:]
        yield (shot,output_decoded)

def annotate_key_shots_siglip(key_shots,video_path,model_id="google/siglip-base-patch16-224"):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224")
    candidate_labels = ["A Rescue Scene", "An Escape Scene", "A Capture Scene", "An Heist Scene",
                    "A Car Chase", "Speed", "A Fight Scene", "A Pursuit Scene",
                    "Not a Rescue scene neither an escape scene nor a capture scene nor a heist scene nor a fight scene nor a pursuit scene"]
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i in key_shots:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            outputs = image_classifier(pil_image, candidate_labels=candidate_labels)
            outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
            yield (i, outputs)

def write_to_csv(image_list, output_path):
    os.makedirs(output_path, exist_ok=True)
    i = 0
    with open(output_path+"_annotated.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Description"])
        for img, desc in tqdm(image_list,desc="Annotating key shots"):
            image_path = os.path.join(output_path, f"Key_shot_{i}.png")
            img.save(image_path)
            writer.writerow([image_path, desc])
            i = i+1
            
def write_to_csv_1(image_list, output_path):
    os.makedirs(output_path, exist_ok=True)
    i = 0
    with open(output_path+"key_shot_annotated_7b.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Key Shot", "Description"])
        for img, desc in tqdm(image_list,desc="Annotating key shots"):
            writer.writerow([img, desc])
            
def annotator(keyshot, video, prompt,model_id="llava-hf/llava-interleave-qwen-0.5b-hf",processor_id="llava-hf/llava-interleave-qwen-0.5b-hf"):
    processor = LlavaProcessor.from_pretrained(processor_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,load_in_4bit=True)
    model.to("cuda")
    for i, keyshot_entry in enumerate(keyshot):
        
        # print(keyshot_entry)
        frames = []
        for frame in keyshot_entry["frames"][:25]:
            frames.append(extract_frame(video,frame))
        # frames = extract_frames_to_pil(video_path,shot-window,shot+window)
        toks = "<image>" * (len(frames))
        prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
        inputs = processor(text=prompt,images=frames,return_tensors="pt").to(model.device, model.dtype)
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        torch.cuda.empty_cache()
        output_decoded = processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:]
        yield (i,output_decoded)
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Key Shot Annotation")
    # parser.add_argument('--csv', type=str, help="CSV Path")
    parser.add_argument('--video', type=str, help="Video Path")
    args = parser.parse_args()
    video_path = args.video
    # csv_path = args.csv
    # key_shots = pd.read_csv(csv_path).values.reshape(1,-1)[0].tolist()
    f = open('key_shots.json',)
    key_shots = json.load(f)
    f.close()
    print(key_shots)
    
    user_prompt = """
    Classify the given video into the following actions:
    1. Rescue
    2. Escape 
    3. Capture 
    4. Heist
    5. Fight
    6. Pursuit
    7. Speed
    8. None of the Above - For scenes that do not fall into any of the aforementioned categories.
    Your Reply can include multiple categories if possible. For example, a scene can have Rescue and Escape.
    """
    # annotator(key_shots, video_path, user_prompt)
    
    image_list = annotator(key_shots, video_path, user_prompt, model_id="llava-hf/llava-interleave-qwen-7b-hf",processor_id="llava-hf/llava-interleave-qwen-0.5b-hf")
    
    # image_list = annotate_key_shots(key_shots=key_shots,video_path=video_path,user_prompt=user_prompt)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    write_to_csv_1(image_list,base_name)
    # image_list = annotate_around_key_shots(processor_id = "llava-hf/llava-interleave-qwen-0.5b-hf",model_id = "abhi-2/checkpoint-156",key_shots=key_shots,video_path=video_path,user_prompt=user_prompt)
    # base_name = os.path.splitext(os.path.basename(video_path))[0]
    # write_to_csv_1(image_list,base_name)
    # image_list = annotate_key_shots_siglip(key_shots=key_shots, video_path=video_path)
    # for entry in image_list:
    #     print(entry 