import os
import wget
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import torch
import pickle as pkl

def main():

    #download and crearte
    ROOT = os.getcwd()
    data_dir = os.path.join(ROOT,'data')
    os.makedirs(data_dir, exist_ok=True)
    an4_audio = os.path.join(data_dir,'an4_diarize_test.wav')
    an4_rttm = os.path.join(data_dir,'an4_diarize_test.rttm')
    if not os.path.exists(an4_audio):
        an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
        an4_audio = wget.download(an4_audio_url, data_dir)
    if not os.path.exists(an4_rttm):
        an4_rttm_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
        an4_rttm = wget.download(an4_rttm_url, data_dir)

    
    meta = {
        'audio_filepath': an4_audio, 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': 2, 
        'rttm_filepath': an4_rttm, 
        'uem_filepath' : None
    }
    with open('data/input_manifest.json','w') as fp:
        json.dump(meta,fp)
        fp.write('\n')

    output_dir = os.path.join(ROOT, 'oracle_vad')
    os.makedirs(output_dir,exist_ok=True)

    
    MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        MODEL_CONFIG = wget.download(config_url,data_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    config.diarizer.manifest_filepath = 'data/input_manifest.json'
    config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
    pretrained_speaker_model = 'titanet_large'
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
    config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
    config.diarizer.oracle_vad = True # ----> ORACLE VAD 
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=config)
    # And lets diarize
    oracle_vad_clusdiar_model.diarize()

    
if __name__ == '__main__':
    main()