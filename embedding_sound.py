from typing import Union, Any
import json
import os
import pickle as pkl
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.utils import logging
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
    def __init__(self,speaker_model=None) -> None:
        self.multiscale_embeddings_and_timestamps = {}
        self._init_speaker_model(speaker_model)
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
            'labels': None,
            'num_workers': self._cfg.num_workers,
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """
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

        if self._speaker_params.save_embeddings:
            embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir, exist_ok=True)

            prefix = get_uniqname_from_filepath(manifest_file)
            name = os.path.join(embedding_dir, prefix)
            self._embeddings_file = name + f'_embeddings.pkl'
            pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
            logging.info("Saved embedding files to {}".format(embedding_dir))