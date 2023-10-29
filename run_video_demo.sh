python combine_v2.py \
    --config 288_depth8_score \
    --params__model ./mixformer-models/mixformerv2_base.pth.tar \
    --face_model ./models/GN_W0.5_S2_ArcFace_epoch16.h5 \
    --video ./video/thuy_minh_3.mp4 \
    --output ./target \
    --known_user ./known_user 