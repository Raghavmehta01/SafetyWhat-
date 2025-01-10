import cv2
import os
import time
from project_utils.image_utils import crop_image, save_image
from ultralytics import YOLO

VIDEO_PATH = "/Users/raghavmehta/Desktop/Coding/projects py/SafetyWhat/data/test_video.mp4"
CROPPED_IMAGES_DIR = "data/cropped_images/"
FRAME_LIMIT = 100

os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

object_counts = {"person": 0, "car": 0, "motorcycle": 0, "truck": 0}

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    return cap

def process_frame(frame, model, frame_idx):
    start_time = time.time()  # Start the timer before inference

    results = model(frame)
    detections = results[0].boxes.data

    if detections is not None:
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, _, class_id = map(int, det[:6])
            label = results[0].names[class_id]
            
            if label in object_counts:
                object_counts[label] += 1

            cropped_img = crop_image(frame, (x1, y1, x2, y2))
            save_path = os.path.join(CROPPED_IMAGES_DIR, f"{label}_frame{frame_idx}_obj{idx}.jpg")
            save_image(cropped_img, save_path)

    inference_time = time.time() - start_time  # End the timer after inference
    return inference_time  # Return the time taken for inference

def main():
    model = YOLO("yolov5s.pt")
    cap = load_video(VIDEO_PATH)
    frame_idx = 0
    total_inference_time = 0  # Initialize a variable to accumulate total inference time
    processed_frames = 0  # Count how many frames were processed

    while cap.isOpened() and frame_idx < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break

        inference_time = process_frame(frame, model, frame_idx)
        total_inference_time += inference_time  # Add the inference time for the current frame
        processed_frames += 1  # Increment the number of processed frames
        
        frame_idx += 1

    cap.release()

    if processed_frames > 0:
        average_inference_time = total_inference_time / processed_frames  # Calculate the average time per frame
        fps = 1 / average_inference_time  # FPS is the reciprocal of the average time per frame
        print(f"Average Inference Time per Frame: {average_inference_time:.4f} seconds")
        print(f"Inference Speed (FPS): {fps:.2f} FPS")
    
    for obj, count in object_counts.items():
        print(f"{obj.capitalize()}s: {count}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
