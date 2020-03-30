import cv2
import os
import sys
import numpy as np
import multiprocessing as mp
from model_inference import run_retinaface, run_emotion_net
from utils import get_frames, preprocess_input

sys.path.append('./retinaface_tf2')

def process_video(video_path, fps):
    
    video_frames = get_frames(video_path, fps)
        
    face_img_queue = mp.Queue()
    video_preds_queue = mp.Queue()
    
    face_detection_process = mp.Process(target=run_retinaface, args = (video_frames, face_img_queue))
    
    face_detection_process.start()
    video_face_imgs = face_img_queue.get()
    face_detection_process.join()
    
    emotion_detection_process = mp.Process(target=run_emotion_net, args = (video_face_imgs, video_preds_queue))
    
    emotion_detection_process.start()
    video_frame_preds = video_preds_queue.get()
    emotion_detection_process.join()
    
    print('Video Successfully Processed')
    
    pred_dict = to_dict(video_frame_preds)
    r = send_response(pred_dict)
    
    if (r.status_code == 200):
        print("Predictions succesfully posted.")
    
    return video_frame_preds

if __name__ == "__main__":
    
    preds = process_video('./sample_videos/2020-03-09-124959.webm', 1)
    
    
    
    
    
    
    
    
    
    
    
    
    
