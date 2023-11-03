from speechbrain.pretrained import EncoderASR

model = EncoderASR.from_hparams(source="dragonSwing/wav2vec2-base-vn-270h", savedir="pretrained_models/asr-wav2vec2-vi")
model.transcribe_file('dragonSwing/wav2vec2-base-vn-270h/example.mp3')