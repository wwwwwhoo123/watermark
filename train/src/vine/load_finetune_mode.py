import sys
import torch
import argparse
from train.src.wm_model.network import WatermarkEncoder, WatermarkDecoder, WatermarkDecoder256, WatermarkEncoder256
from train.src.wm_model.adaptor0 import Adaptor

p = "src/"
sys.path.append(p)
from vine_turbo import initialize_vae_no_lora, VAE_encode, VAE_decode

set_skip_config = "1:0.0,2:0.0,3:0.0,4:0.0"


def load_finetuned_decoder(base_model_path, finetuned_path):
    print(f"loading: {base_model_path}")
    vae = initialize_vae_no_lora(path=base_model_path)

    print(f"loading: {finetuned_path}")
    checkpoint = torch.load(finetuned_path, map_location="cpu")

    if "state_dict" in checkpoint:
        decoder_state_dict = checkpoint["state_dict"]
    else:
        decoder_state_dict = checkpoint

    vae_state_dict = vae.state_dict()

    for key, value in decoder_state_dict.items():
        if key in vae_state_dict:
            vae_state_dict[key] = value

    vae.load_state_dict(vae_state_dict)

    return vae


def load_finetuned_models(base_model_path, vae_checkpoint_path, wm_checkpoint_path):
    print(f"loading: {base_model_path}")
    vae = initialize_vae_no_lora(path=base_model_path)

    print(f"loading: {vae_checkpoint_path}")
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
    decoder_state_dict = vae_checkpoint["vae_decoder_state_dict"]

    vae_state_dict = vae.state_dict()
    for key, value in decoder_state_dict.items():
        if key in vae_state_dict:
            vae_state_dict[key] = value
    vae.load_state_dict(vae_state_dict)

    print(f"loading: {wm_checkpoint_path}")
    wm_checkpoint = torch.load(wm_checkpoint_path, map_location="cpu")
    wm_encoder = WatermarkEncoder()
    wm_decoder = WatermarkDecoder()

    wm_encoder.load_state_dict(wm_checkpoint["wm_encoder_state_dict"])
    wm_decoder.load_state_dict(wm_checkpoint["wm_decoder_state_dict"])

    return vae, wm_encoder, wm_decoder


def load_finetuned_models_adaptor(base_model_path, vae_checkpoint_path, wm_checkpoint_path, adaptor_checkpoint_path):
    print(f"loading: {base_model_path}")
    vae = initialize_vae_no_lora(path=base_model_path)

    print(f"loading: {vae_checkpoint_path}")
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
    decoder_state_dict = vae_checkpoint["vae_decoder_state_dict"]

    vae_state_dict = vae.state_dict()
    for key, value in decoder_state_dict.items():
        if key in vae_state_dict:
            vae_state_dict[key] = value
    vae.load_state_dict(vae_state_dict)

    print(f"loading: {wm_checkpoint_path}")
    wm_checkpoint = torch.load(wm_checkpoint_path, map_location="cpu")
    wm_encoder = WatermarkEncoder()
    wm_decoder = WatermarkDecoder()

    wm_encoder.load_state_dict(wm_checkpoint["wm_encoder_state_dict"])
    wm_decoder.load_state_dict(wm_checkpoint["wm_decoder_state_dict"])

    print(f"loading: {adaptor_checkpoint_path}")
    adaptor_checkpoint = torch.load(adaptor_checkpoint_path, map_location="cpu")
    adaptor = Adaptor(4, 64)
    adaptor.load_state_dict(adaptor_checkpoint["adaptor_state_dict"])

    return vae, wm_encoder, wm_decoder, adaptor


def load_finetuned_models_adaptor256(base_model_path, vae_checkpoint_path, wm_checkpoint_path, adaptor_checkpoint_path):
    print(f"loading: {base_model_path}")
    vae = initialize_vae_no_lora(path=base_model_path)

    print(f"loading: {vae_checkpoint_path}")
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
    decoder_state_dict = vae_checkpoint["vae_decoder_state_dict"]

    vae_state_dict = vae.state_dict()
    for key, value in decoder_state_dict.items():
        if key in vae_state_dict:
            vae_state_dict[key] = value
    vae.load_state_dict(vae_state_dict)

    print(f"loading: {wm_checkpoint_path}")
    wm_checkpoint = torch.load(wm_checkpoint_path, map_location="cpu")
    wm_encoder = WatermarkEncoder256()
    wm_decoder = WatermarkDecoder256()

    wm_encoder.load_state_dict(wm_checkpoint["wm_encoder_state_dict"])
    wm_decoder.load_state_dict(wm_checkpoint["wm_decoder_state_dict"])

    print(f"loading: {adaptor_checkpoint_path}")
    adaptor_checkpoint = torch.load(adaptor_checkpoint_path, map_location="cpu")
    adaptor = Adaptor(4, 64)
    adaptor.load_state_dict(adaptor_checkpoint["adaptor_state_dict"])

    return vae, wm_encoder, wm_decoder, adaptor
