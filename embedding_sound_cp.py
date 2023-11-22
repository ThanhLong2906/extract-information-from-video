from typing import Union, Any
import json
import os
import wget
from omegaconf import OmegaConf
import pickle as pkl
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.utils import logging, model_utils
from nemo.collections.asr.models.clustering_diarizer import get_available_model_names
from omegaconf import DictConfig
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_uniqname_from_filepath,
    parse_scale_configs,
)
from tqdm import tqdm
import torch
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

class Embedding_sound():
    def __init__(self, cfg: Union[DictConfig, Any]) -> None:
        if isinstance(cfg, DictConfig):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            # Convert config to support Hydra 1.0+ instantiation
            cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        # Diarizer set up
        self._diarizer_params = self._cfg.diarizer

        self.multiscale_embeddings_and_timestamps = {}
        # speaker_model = self._cfg.diarizer.speaker_embeddings.model_path
        self._init_speaker_model()
        self._speaker_params = self._cfg.diarizer.speaker_embeddings.parameters

    def _init_speaker_model(self, speaker_model=None):
            """
            Initialize speaker embedding model with model name or path passed through config
            """
            if speaker_model is not None:
                self._speaker_model = speaker_model
            else:
                model_path = self._cfg.diarizer.speaker_embeddings.model_path
                if model_path is not None and model_path.endswith('.nemo'):
                    self._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=self._cfg.device)
                    logging.info("Speaker Model restored locally from {}".format(model_path))
                elif model_path.endswith('.ckpt'):
                    self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(
                        model_path, map_location=self._cfg.device
                    )
                    logging.info("Speaker Model restored locally from {}".format(model_path))
                else:
                    if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                        logging.warning(
                            "requested {} model name not available in pretrained models, instead".format(model_path)
                        )
                        model_path = "ecapa_tdnn"
                    logging.info("Loading pretrained {} model from NGC".format(model_path))
                    self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                        model_name=model_path, map_location=self._cfg.device
                    )

            # self.multiscale_args_dict = parse_scale_configs(
            #     self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            #     self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            #     self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
            # )
    def _setup_spkr_test_data(self, manifest_file):
        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'trim_silence': False,
            'labels': "UNK",
            'num_workers': self._cfg.num_workers,
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """

        self._out_dir = self._diarizer_params.out_dir

        # self._speaker_dir = os.path.join(self._diarizer_params.out_dir, 'speaker_outputs')
        # self._speaker_dir = self._diarizer_params.out_dir

        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        self.embeddings = {}
        self._speaker_model.eval()
        self.time_stamps = {}

        all_embs = torch.empty([0])
        for test_batch in tqdm(
            self._speaker_model.test_dataloader(),
            desc=f'[{scale_idx+1}/{num_scales}] extract embeddings',
            leave=True,
            disable=not self.verbose,
        ):
            test_batch = [x.to(self._speaker_model.device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
            del test_batch

        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
                if uniq_name in self.embeddings:
                    self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
                else:
                    self.embeddings[uniq_name] = all_embs[i].view(1, -1)
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                self.time_stamps[uniq_name].append([start, end])
                label = dic['label']
                uniq_id = dic['uniq_id']

        if self._speaker_params.save_embeddings:
            embedding_dir = os.path.join(self._out_dir, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir, exist_ok=True)

            prefix = get_uniqname_from_filepath(manifest_file)
            if uniq_id:
                prefix += "_"+str(uniq_id)
            name = os.path.join(embedding_dir, prefix)
            self._embeddings_file = name + f'_embeddings.pkl'
            pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
            logging.info("Saved embedding files to {}".format(embedding_dir))
        return self._embeddings_file
    @property
    def verbose(self) -> bool:
        return self._cfg.verbose


def embedding_human_voice(config: dict, audio_filepath: str, offset: float, duration:float, label: str='U', uniq_id: str = "unk", output: str = None):
    # create output folder
    os.makedirs(output, exist_ok=True)
    # check if embedding exists
    name = get_uniqname_from_filepath(audio_filepath)
    pkl_file = os.path.join(output, f"embeddings/{name}_embeddings.pkl")
    if os.path.exists(pkl_file):
        print(f"{name} embedding existed. Embedding located in {pkl_file}")
        return pkl_file
    # create manifest file
    meta = {
        'audio_filepath': audio_filepath, 
        'offset': offset, 
        'duration': duration, # write it manually 
        'label': label, 
        'uniq_id': uniq_id
    }
    manifest_filepath = os.path.join(output,get_uniqname_from_filepath(audio_filepath) + ".json") 

    with open(manifest_filepath,'w+') as fp:
        json.dump(meta,fp)
        fp.write('\n')

    # embedding
    # ROOT = os.getcwd()
    # data_dir = os.path.join(ROOT,'data')
    # MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
    # if not os.path.exists(MODEL_CONFIG):
    #     config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    #     MODEL_CONFIG = wget.download(config_url,data_dir)

    # config = OmegaConf.load(MODEL_CONFIG)
    config.diarizer.out_dir = output
    embedding = Embedding_sound(config)
    embedding_path = embedding._extract_embeddings(manifest_file=manifest_filepath, scale_idx=0, num_scales=1)
    return embedding_path

def embedding_sound(config: dict, audio_filepath: str, rttm_filepath: str, output: str = None):
    embeddings_path = []
    if output:
        os.makedirs(output, exist_ok=True)
    if rttm_filepath.endswith(".rttm"):
        with open(rttm_filepath, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                offset = float(line.split()[3])
                duration = float(line.split()[4])
                name = line.split()[7]
                # create output folder
                
                # check if embedding exists
                # name = get_uniqname_from_filepath(audio_filepath)
                # pkl_file = os.path.join(output, f"speaker_outputs/embeddings/{name}_embeddings.pkl")
                # if os.path.exists(pkl_file):
                #     print(f"{name} embedding existed. Embedding located in {pkl_file}")
                #     exit()
                # create manifest file
                meta = {
                    'audio_filepath': audio_filepath, 
                    'offset': offset, 
                    'duration': duration, # write it manually 
                    'label': 'U', 
                    'uniq_id': idx
                }
                
                manifest_filepath = os.path.join(output,name + ".json") 

                with open(manifest_filepath,'w+') as fp:
                    json.dump(meta,fp)
                    fp.write('\n')

                config.diarizer.out_dir = output
                embedding = Embedding_sound(config)
                embed_path = embedding._extract_embeddings(manifest_file=manifest_filepath, scale_idx=0, num_scales=1)
                embeddings_path.append(embed_path)
    return embeddings_path

def sound_similarity(audio_sound_emb: str, human_sound_emb: str):
    '''
        audio_sound_emb: link to pickle file of audio sound segment
        human_sound_emb: link to pickle file of human sound 
        return: cosin similarity of 2 sound
    '''
    audio_emb = next(iter(pkl.load(open(audio_sound_emb, "rb")).values()))
    human_emb = next(iter(pkl.load(open(human_sound_emb, "rb")).values()))
    cos = torch.nn.CosineSimilarity()
    return cos(audio_emb, human_emb)

# def main():

#     #download and crearte
#     ROOT = os.getcwd()
#     data_dir = os.path.join(ROOT,'data')

#     output_dir = os.path.join(ROOT, 'oracle_vad')
#     os.makedirs(output_dir,exist_ok=True)

#     MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
#     if not os.path.exists(MODEL_CONFIG):
#         config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
#         MODEL_CONFIG = wget.download(config_url,data_dir)

#     config = OmegaConf.load(MODEL_CONFIG)
#     # config.diarizer.out_dir = output_dir
#     database = "./database/"
#     human_emb_path = embedding_human_voice(config, audio_filepath = "./data/TPM.wav", offset=0.1, duration=2.8, label = "speaker0", uniq_id = 1, output = database)
#     # embeddings_path = embedding_sound(config, os.path.join(data_dir, "file.wav"), os.path.join(data_dir, "file.rttm"))
#     # scores = []
#     # for path in embeddings_path:
#     #     score = sound_similarity(path, human_emb_path)
#     #     scores.append(score)
#     # # human = "./database/speaker_outputs/embeddings/TPM_embeddings.pkl"
#     # # embedding sound
#     # with open("new_file.rttm", 'wb') as nf:
#     #     with open("file.rttm", "r+") as f:
#     #         lines = f.readlines()
#     #         for line, score in zip(lines, scores):
#     #             if score >= config.speaker_embeddings.parameters.threshold:
#     #                 line.split()[7] = get_uniqname_from_filepath(human_emb_path)
#     #                 line = ' '.join(line)
#     #                 nf.write(line.encode('utf-8'))
#     #                 nf.write(b'\n')

# if __name__=='__main__':
#     main()