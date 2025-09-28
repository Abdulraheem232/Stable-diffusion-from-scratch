from clip import Clip
from encoder import Vae_encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict_weights = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = Vae_encoder().to(device)
    encoder.load_state_dict(state_dict_weights['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict_weights['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict_weights['diffusion'], strict=True)

    clip = Clip().to(device)
    clip.load_state_dict(state_dict_weights['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }