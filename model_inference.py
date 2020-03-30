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
