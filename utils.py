import cv2
import numpy as np
import requests

def get_frames(video_path, fps):
    
    video_frame_list = []
    frame_count = 0
    
    cap = cv2.VideoCapture(video_path)
    
    frame_skip = (60/fps) - 1
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == False:
            break
        
        if (frame_count % frame_skip) == 0:
            video_frame_list.append(frame)
        
        frame_count += 1
    
    return video_frame_list

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def to_dict(video_frame_preds):
    
    num_frames = len(video_frame_preds)
    video_frame_scores = []
    
    for frame in video_frame_preds:
        frame_score = np.mean(frame, axis=0)
        frame_score = list(frame_score)
        video_frame_scores.append(frame_score)
    
    request_dict = {
        "num_frames": num_frames,
        "emotion_scores": video_frame_scores
    }
    
    return request_dict

def send_response(pred_dict):
    
    r = requests.post('http://localhost:4000/api/predictComplete', json = pred_dict)
    return r