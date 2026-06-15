import torch
import yaml
import kornia
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import torchvision.transforms as transforms
from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.utils import *
from loss.pytorch_ssim import ssim
from logger_config.logger import *
from train.src.vine.vine_turbo import VAE_encode, VAE_decode
from train.src.vine.load_finetune_mode import load_finetuned_decoder, load_finetuned_models_adaptor, load_finetuned_models_adaptor256
import json


def save_watermark_image(watermark_tensor, path):
    watermark_tensor = watermark_tensor.clone().detach().cpu()
    watermark_tensor = (watermark_tensor * 0.5 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(watermark_tensor[0])
    img.save(path)


def calculate_nc(original, decoded):
    original = original.view(original.size(0), -1)
    decoded = decoded.view(decoded.size(0), -1)
    orig_mean = original.mean(dim=1, keepdim=True)
    dec_mean = decoded.mean(dim=1, keepdim=True)
    orig_std = original.std(dim=1, keepdim=True)
    dec_std = decoded.std(dim=1, keepdim=True)
    numerator = ((original - orig_mean) * (decoded - dec_mean)).sum(dim=1)
    denominator = (orig_std * dec_std) * (original.size(1) - 1)
    nc = numerator / denominator
    return nc.mean().item()


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def get_text_embedding(prompts, pipeline):
    torch.manual_seed(42)
    with torch.no_grad():
        prompt = prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def main():
    torch.manual_seed(42)
    device = torch.device('cuda')
    with open('./example/config/config_list_adaptor.yaml', 'r') as file:
        cfgs = yaml.safe_load(file)

    vae, wm_encoder, wm_decoder, adaptor = load_finetuned_models_adaptor(cfgs['model_id'], cfgs['vae_path'], cfgs['wm_path'], cfgs['adaptor_path'])
    vae_encoder = VAE_encode(vae)
    vae_decoder = VAE_decode(vae)
    vae_encoder.to(device)
    vae_decoder.to(device)
    wm_encoder.to(device)
    wm_decoder.to(device)
    adaptor.to(device)

    watermark = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])(Image.open(cfgs['watermark'])).unsqueeze(0).to(device)
    watermark_latent = wm_encoder(watermark)

    wm_path = cfgs['save_img']
    csv_file_path = os.path.join(wm_path, 'results.csv')
    csv_header = ['Image Name', 'Enhance PSNR', 'Enhance SSIM', 'NC', 'alpha']
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    if not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0:
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerow(csv_header)

    with open(r'E:\AProject\awhoo\sd3\prompt_list\prompt.json', 'r') as f:
        prompts = json.load(f)

    id = "stabilityai/stable-diffusion-3-medium-diffusers"
    text_encoder = T5EncoderModel.from_pretrained(
        id,
        subfolder="text_encoder_3",
        # load_in_8bit=True,
        device_map="auto",
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        id,
        text_encoder_3=text_encoder,
        transformer=None,
        vae=None,
        device_map="balanced",
    )

    text_embeddings_list = []
    for item in tqdm(prompts):
        caption = item['caption']
        embeds = get_text_embedding(caption, pipeline)
        text_embeddings_list.append((item['image_id'], *embeds))

    del text_encoder
    del pipeline
    flush()

    pipe = WMDetectStableDiffusionPipeline.from_pretrained(
        cfgs['model_id'],
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for (image_id, text_embeddings, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) in tqdm(text_embeddings_list):

        with torch.no_grad():
            latent = pipe(
                prompt_embeds=text_embeddings,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type='latent',
            ).images[0].unsqueeze(0)

            image_ori = vae_decoder(latent, direction="a2b")

            alpha = adaptor(latent[:, 0:4, :, :], watermark_latent)

            latent[:, 0:4, :, :] = latent[:, 0:4, :, :] + alpha * watermark_latent
            image_wm = vae_decoder(latent, direction="a2b")

            re_watermark = wm_decoder(image_wm)

            ssim_value = ssim(image_ori, image_wm).item()
            psnr_value = compute_psnr(image_ori, image_wm)
            nc_value = calculate_nc(watermark, re_watermark)

        for subfolder, img in [('generate', image_ori), ('marked', image_wm)]:
            save_dir = os.path.join(wm_path, subfolder)
            os.makedirs(save_dir, exist_ok=True)
            save_img(os.path.join(save_dir, f"{image_id}.png"), img, pipe)

        wm_out_dir = os.path.join(wm_path, 'watermark')
        os.makedirs(wm_out_dir, exist_ok=True)
        save_watermark_image(re_watermark, os.path.join(wm_out_dir, f"{image_id}.png"))

        with open(csv_file_path, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([image_id, psnr_value, ssim_value, nc_value, alpha.item()])


if __name__ == "__main__":
    main()
