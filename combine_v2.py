import tensorflow as tf
from video_test import init_det_and_emb_model, embedding_images, image_recognize, draw_polyboxes
import cv2

import torch
import numpy as np

import sys
import os
import glob
import argparse

from ur_audio_sub import *
from pathlib import Path

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor 
from lib.test.evaluation import Tracker
# sys.path.insert(0, "C:/Users/MyPC/extract-information-from-video/pysot/")
# sys.path.insert(0, "C:/Users/MyPC/extract-information-from-video/GhostFaceNets/")

parser = argparse.ArgumentParser(description='video extract information demo')
parser.add_argument('--config', type=str, help='config file')  # ~tracker param
# parser.add_argument('--snapshot', type=str, help='tracking model') # ~ param model
parser.add_argument('--face_model', type=str, help='face recognize model')
parser.add_argument('--video', type=str, help='input video files') # ~videofile
parser.add_argument("--output", type=str, help='output folder') 
parser.add_argument('--known_user', type=str, help="input image folder")

parser.add_argument('--tracker_name', type=str, default= "mixformer2_vit_online", help='Name of tracking method.')
# parser.add_argument('--tracker_param', type=str, help='Name of parameter file.')
# parser.add_argument('--videofile', type=str, help='path to a video file.')
# parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
parser.set_defaults(save_results=True)

parser.add_argument('--params__model', type=str, default=None, help="Tracking model path.")
parser.add_argument('--params__update_interval', type=int, default=25, help="Update interval of online tracking.")
parser.add_argument('--params__online_size', type=int, default=1)
parser.add_argument('--params__search_area_scale', type=float, default=4.5)
parser.add_argument('--params__max_score_decay', type=float, default=1.0)
parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")

args = parser.parse_args()


DIST_THRESH = 0.1
# MODEL_FILE = "./models/GN_W0.5_S2_ArcFace_epoch16.h5"
args.known_user = "./known_user"
# VIDEO_SOURCE = "video/2_eminem_cut.mp4"
TRACKING_THRESH = 0.7

def get_frames(video_name):
    
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cur_frame_idx = 0
        cap = cv2.VideoCapture(args.video)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame, cur_frame_idx
                cur_frame_idx += 1
            else:
                break
    # else:
    #     images = glob(os.path.join(video_name, '*.jp*'))
    #     images = sorted(images,
    #                     key=lambda x: int(x.split('/')[-1].split('.')[0]))
    #     for img in images:
    #         frame = cv2.imread(img)
    #         yield frame

def video_recognize(image_classes, embeddings, det, face_model, video_source, frames_per_detect=5, dist_thresh=0.1):
    cap = cv2.VideoCapture(str(video_source))
    cur_frame_idx = 0
    bbs = []
    while True:
        grabbed, frame = cap.read()
        if grabbed != True:
            return bbs, cur_frame_idx
            # break
        else:
            if cur_frame_idx % frames_per_detect == 0:
                rec_dist, rec_class, bbs, ccs = image_recognize(image_classes, embeddings, det, face_model, frame)
                # print(f"rec_dist: {rec_dist}")
                if len(rec_dist)>0:
                    if rec_dist[0] > dist_thresh:
                        return bbs, cur_frame_idx 
            # cur_frame_idx = 0

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("s"):
        #     cv2.imwrite("{}.jpg".format(cur_frame_idx), frame)
        # if key == ord("q"):
        #     break

        # draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh)
        # cv2.imshow("", frame)
            cur_frame_idx += 1
        
    cap.release()
    # cv2.destroyAllWindows()

def face_recognize(model_file:str, video_source: str, known_user: str = None, 
                   known_user_force: str = None, embedding_batch_size: int = 4, 
                   ):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    det, face_model = init_det_and_emb_model(model_file)
    if known_user_force != None:
        force_reload = True
        known_user = known_user_force
    else:
        force_reload = False
        known_user = args.known_user
    print(args.known_user)
    if known_user != None and face_model is not None:
        image_classes, embeddings, _ = embedding_images(det, face_model, known_user, embedding_batch_size, force_reload)
        print(embeddings)
        # video_source = int(video_source) if str.isnumeric(video_source) else video_source
        bbs, cur_frame_idx = video_recognize(image_classes, embeddings, det, face_model, video_source)
    return bbs, cur_frame_idx

def get_fps(video: str):
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps

def speech2text(file_path: str, model: str ="medium", language: str= "Vietnamese"):
    model=whisper.load_model(model)
    result = model.transcribe(file_path, language=language, fp16=False)
    return result["text"]

def run_video(tracker_name, tracker_param, idx, tracking_thresh, videofile='', optional_box=None, debug=None,
              save_results=False, tracker_params=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video", tracker_params=tracker_params)
    time, fps = tracker.run_video(videofilepath=videofile, idx= idx, tracking_thresh=tracking_thresh, optional_box=optional_box, debug=debug, save_results=save_results)
    return time, fps

def main():
    bbs, idx = face_recognize(args.face_model,args.video, args.known_user)
    # print(args.video)
    print(f"bbs:{bbs}, idx:{idx}")

    if len(bbs) != 0 and len(bbs[0]) != 0:
        is_face = True
    if args.video:
        video_name = args.video.split('/')[-1].split('.')[0]
    else:
        args.video = 'webcam'
    
    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param) 
    # bbs[0].tolist()
    time, fps = run_video(args.tracker_name, args.config, idx, TRACKING_THRESH, args.video, [455, 102, 178, 259], args.debug,
                args.save_results, tracker_params=tracker_params)

    for idx, interval in enumerate(time):
        # find timestamp of each frame
        start_time = (1/fps)*interval[0]
        end_time = (1/fps)*interval[1]
        output = f"{args.output}/video_{idx}.wav"
        # cut_video((start_time, end_time), output=output)
        # extract audio from file
        video = moviepy.editor.VideoFileClip(args.video).subclip(start_time, end_time)
        audio = video.audio
        audio.write_audiofile(output)
        # subGen_path(output)
        text = speech2text(output)
        with open(f"{args.output}/video_{idx}.txt", "w+", encoding='utf-8') as f:
            f.write(text)

if __name__=='__main__':
    main()

