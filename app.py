from flask import Flask, request, jsonify
import multiprocessing as mp
import audience_ai

app = Flask(__name__)

@app.route('/')
def base_route():
    return("Hello")

@app.route('/process_video', methods=["POST"])
def process_video():
    
    req_data = request.get_json()
    
    video_path = req_data['video_path']
    fps = req_data['fps']
    
    video_process = mp.Process(target=audience_ai.process_video, args = (video_path, fps))
    video_process.start()
    
    return jsonify({"status": 1, "message": "Video Processing Started"})




    
    
    
    
    
    