import os
import wget
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import torch
import pickle as pkl
from nemo.collections.asr.parts.utils.speaker_utils import get_uniqname_from_filepath
from embedding_sound_cp import (
    embedding_human_voice,
    embedding_sound,
    sound_similarity
)
import argparse
THRESH = 0.1
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-vb", "--voice_embeddings", type=str, description="object voice file for embeddings")
    # parser.add_argument("-")
    # args = parser.parse_args()
    #download and crearte
    ROOT = os.getcwd()
    data_dir = os.path.join(ROOT,'input_data')
    os.makedirs(data_dir, exist_ok=True)
    phone_audio = os.path.join(data_dir,'ielts.wav')
    
    # if not os.path.exists(phone_audio):
    #     an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
    #     phone_audio = wget.download(an4_audio_url, data_dir)
    # if not os.path.exists(phone_rttm):
    #     an4_rttm_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
    #     phone_rttm = wget.download(an4_rttm_url, data_dir)

    #create file input_manifest.json 
    meta = {
        'audio_filepath': phone_audio, 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': 2, 
        'rttm_filepath': None,#phone_rttm, 
        'uem_filepath' : None
    }

    with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
        json.dump(meta,fp)
        fp.write('\n')


    output_dir = os.path.join(ROOT, 'output')
    # pred_rttms_dir = os.path.join(output_dir, "pred_rttms")
    os.makedirs(output_dir,exist_ok=True)

    phone_rttm = os.path.join(output_dir,'pred_rttms/phone.rttm')

    # create config file
    MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        MODEL_CONFIG = wget.download(config_url,data_dir)

    # add information for config file for clustering process
    config = OmegaConf.load(MODEL_CONFIG)

    config.diarizer.manifest_filepath = os.path.join(data_dir, 'input_manifest.json')
    config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
    pretrained_vad = 'vad_multilingual_marblenet'
    pretrained_speaker_model = 'titanet_large'
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [2.0, 1.75, 1.5, 1.25, 1] #[1.5,1.25,1.0,0.75,0.5] 
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [1.0, 0.875, 0.75, 0.625, 0.5]#[0.75,0.625,0.5,0.375,0.1] 
    config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
    config.diarizer.oracle_vad = False # ----> ORACLE VAD 
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05

    from nemo.collections.asr.models import ClusteringDiarizer
    sd_model = ClusteringDiarizer(cfg=config)
    sd_model.diarize()

    database = "./database/"
    human_voice = os.path.join(data_dir, "female_voice.wav")
    human_emb_path = embedding_human_voice(config, audio_filepath = human_voice, offset=0.1, duration=2.8, output = database)
    embeddings_path = embedding_sound(config, phone_audio, phone_rttm, output_dir)
    scores = []
    for path in embeddings_path:
        score = sound_similarity(path, human_emb_path)
        scores.append(score)
    # human = "./database/speaker_outputs/embeddings/TPM_embeddings.pkl"
    # embedding sound
    print(scores)
    with open("result.rttm", 'wb') as nf:
        with open(os.path.join(output_dir, "pred_rttms/phone.rttm"), "r+") as f:
            lines = f.readlines()
            
            for line, score in zip(lines, scores):
                if float(score) >= THRESH:#config.speaker_embeddings.parameters.threshold:
                    new_line = line.split()
                    new_line[7] = get_uniqname_from_filepath(human_voice)
                    line = " "
                    for word in new_line:
                        line += word + " "
                    nf.write(line.encode('utf-8'))
                    nf.write(b'\n')

    
if __name__ == '__main__':
    main()