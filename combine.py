import tensorflow as tf
from video_test import init_det_and_emb_model, embedding_images, image_recognize, draw_polyboxes
import cv2

import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import sys
import os
import glob
import argparse

from ur_audio_sub import *
from pathlib import Path

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor 
# sys.path.insert(0, "C:/Users/MyPC/extract-information-from-video/pysot/")
# sys.path.insert(0, "C:/Users/MyPC/extract-information-from-video/GhostFaceNets/")

parser = argparse.ArgumentParser(description='video extract information demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='tracking model')
parser.add_argument('--face_model', type=str, help='face recognize model')
parser.add_argument('--video', type=str, help='input video files')
parser.add_argument("--output", type=str, help='output folder')
parser.add_argument('--known_user', type=str, help="input image folder")
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
# def get_frame_idx():
#     cur_idx = 0
#     time = []
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             if person in frame and start_idx == None:
#                 start_idx = cur_idx
#             elif person not in frame and start_idx != None:
#                 end_idx = cur_idx
#                 time.append(tuple(start_idx, end_idx))
#                 start_idx = None
#             cur_idx += 1
#         else:
#             if start_idx != None:
#                 end_idx = cur_idx
#                 time.append(tuple(start_idx, end_idx))
#             break
# def cut_video(interval: tuple, output: str):
#     start_time = interval[0]
#     end_time = interval[1]
#     ffmpeg_extract_subclip("video/hayley_william.mp4", start_time, end_time, targetname=output)

# def subGen_path(file_path, model='medium', language='Vietnamese', translate=False):
#   """
#   Input audio file path to generate the text caption 
#   Args:
#       file_path (:obj:`str`, required): Eg: "/content/sample_recording.mp3"
#       model (:obj:`str`, optional): choose 1 of 3 options: 'base', 'medium', 'large'. Default value = 'medium'
#       language (:obj:`str`, optional): Target language of the audio file to be generated caption. Auto detects language by default.
#       translate (:obj:`str`, optional): Translate caption to English. No translation by default.
#   """
#   if language == '':
#     if translate == False:
#       os.system('whisper "{}" --model {}'.format(file_path, model))
#     else:
#       os.system('whisper "{}" --model {} --task translate'.format(file_path, model))
#   else:
#     if translate == False:
#       os.system('whisper "{}" --model {} --language {}'.format(file_path, model, language))
#     else:
#       os.system('whisper "{}" --model {} --language {} --task translate'.format(file_path, model, language))

def speech2text(file_path: str, model: str ="medium", language: str= "Vietnamese"):
    model=whisper.load_model(model)
    result = model.transcribe(file_path, language=language, fp16=False)
    return result["text"]

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    bbs, idx = face_recognize(args.face_model,args.video, args.known_user)
    # print(args.video)
    print(f"bbs:{bbs}, idx:{idx}")

    if len(bbs) != 0 and len(bbs[0]) != 0:
        is_face = True
    # else:
    #     print("this person do not appear in the video.")
    #     exit()
    if args.video:
        video_name = args.video.split('/')[-1].split('.')[0]
    else:
        args.video = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    frame_tracked = False
    # print(get_frames(args.video))
    start_idx = None
    time = []
    # for frame, cur_frame_idx in get_frames(args.video):
    cur_frame_idx = 0
    # print(args.video)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if ret:
            # print(cur_frame_idx)
            if is_face and cur_frame_idx == idx:
                try:
                    # init_rect = cv2.selectROI(video_name, frame, False, False) # return a list 4 coordinate
                    init_rect = bbs[0]

                except:
                    exit()
                tracker.init(frame, init_rect)
                start_idx = cur_frame_idx
                is_face = False
                frame_tracked = True
            elif frame_tracked:
                outputs = tracker.track(frame)
                # print(f"outputs: {outputs}")
                if outputs['best_score'] >= TRACKING_THRESH: # tương đương person in frame
                    if start_idx == None:
                        start_idx = cur_frame_idx
                else: # tương đương person not in frame
                    if start_idx != None:
                        end_idx = cur_frame_idx - 1
                        time.append((start_idx, end_idx))
                        start_idx = None

                if 'polygon' in outputs: 
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else: 
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (0, 255, 0), 3)
                cv2.imshow(video_name, frame)
                cv2.waitKey(40)
            cur_frame_idx +=1
        else:
            if start_idx != None:
                end_idx = cur_frame_idx
                time.append((start_idx, end_idx))
            break
    cap.release()
    # print(f"video intervals: {time}")
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

