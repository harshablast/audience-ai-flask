import cv2
import os
import sys
import numpy as np
import multiprocessing as mp

sys.path.append('./retinaface_tf2')

def run_retinaface(video_frames, face_img_queue):
    
    import tensorflow as tf
    from retinaface_tf2.modules.models import RetinaFaceModel
    from retinaface_tf2.modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm, pad_input_image, recover_pad_output)
    cfg = load_yaml('./retinaface_tf2/configs/retinaface_mbv2.yaml')
    model = RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.5)
    checkpoint_dir = './retinaface_tf2/checkpoints/'
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
    
    video_face_imgs = []
    
    for frame in video_frames:
        
        print('lol')
        
        frame_face_imgs = []
        
        frame_height_raw, frame_width_raw, _ = frame.shape
        frame_pred = np.float32(frame.copy())
        frame_pred = cv2.cvtColor(frame_pred, cv2.COLOR_BGR2RGB)
        frame_pred, pad_params = pad_input_image(frame_pred, max_steps = max(cfg['steps']))
        
        outputs = model(frame_pred[np.newaxis, ...]).numpy()
        outputs = recover_pad_output(outputs, pad_params)
        
        for prior_index in range(len(outputs)):
            ann = outputs[prior_index]
            x1, y1, x2, y2 = int(ann[0] * frame_width_raw), int(ann[1] * frame_height_raw), \
                       int(ann[2] * frame_width_raw), int(ann[3] * frame_height_raw)
            face_img = frame[y1:y2,x1:x2]
            frame_face_imgs.append(face_img)
            
        video_face_imgs.append(frame_face_imgs)
        
    face_img_queue.put(video_face_imgs)
    
    
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
    
def run_emotion_net(video_face_imgs, video_preds_queue):
    
    from keras.models import load_model
    
    emotion_model = load_model('./fer2013_mini_XCEPTION.119-0.65.hdf5')
    emotion_target_size = emotion_model.input_shape[1:3]
    
    video_frame_preds = []
    
    for frame_imgs in video_face_imgs:
        
        print('lol2')
        
        face_img_list = []
        
        for face_img in frame_imgs:
            
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = cv2.resize(face_img, (emotion_target_size))
            face_img = preprocess_input(face_img)
            face_img_list.append(face_img)
            
        face_img_list = np.array(face_img_list)
        face_img_list = face_img_list.reshape(-1, 48, 48, 1)
        
        frame_preds = emotion_model.predict(face_img_list)
        
        video_frame_preds.append(frame_preds)
    
    video_preds_queue.put(video_frame_preds)

    
    
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
    
    return video_frame_preds

if __name__ == "__main__":
    
    preds = process_video('./sample_videos/2020-03-09-124959.webm', 1)
    
    
    
    
    
    
    
    
    
    
    
    
    
