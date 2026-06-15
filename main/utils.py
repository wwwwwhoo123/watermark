from PIL import Image
import math
import os
from train.src.utils.pytorch_ssim import ssim
import matplotlib.pyplot as plt

from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import torch


def show_images_side_by_side(images, titles=None, figsize=(8, 4)):
    """
    Display a list of images side by side.
    
    Args:
    images (list of numpy arrays): List of images to display.
    titles (list of str, optional): List of titles for each image. Default is None.
    """
    num_images = len(images)

    if titles is not None:
        if len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images.")

    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis('off')

        if titles is not None:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()
    return


def show_latent_and_final_img(latent: torch.Tensor, img: torch.Tensor, pipe):
    with torch.no_grad():
        latents_pil_img = pipe.numpy_to_pil(pipe.decode_latents(latent.detach()))[0]
        pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    show_images_side_by_side([latents_pil_img, pil_img], ['Latent', 'Generated Image'])
    return


def save_img(path, img: torch.Tensor, pipe):
    pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    pil_img.save(path)
    return


def get_img_tensor(img_path, device):
    img_tensor = pil_to_tensor(Image.open(img_path).convert("RGB")) / 255
    return img_tensor.unsqueeze(0).to(device)


def create_output_folder(cfgs):
    parent = os.path.join(cfgs['save_img'], cfgs['dataset'])
    wm_path = os.path.join(parent, cfgs['method'], cfgs['case'])

    special_model = ['CompVis']
    for key in special_model:
        if key in cfgs['model_id']:
            wm_path = os.path.join(parent, cfgs['method'], '_'.join([cfgs['case'][:-1], key + '/']))
            break

    os.makedirs(wm_path, exist_ok=True)
    ori_path = os.path.join(parent, 'OriImgs/')
    os.makedirs(ori_path, exist_ok=True)
    return wm_path, ori_path


# Metrics for similarity
def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.) - 10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()


def load_img(img_path, device):
    img = Image.open(img_path).convert('RGB')
    x = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
    return x


def eval_psnr_ssim_msssim(ori_img_path, new_img_path, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)


def eval_lpips(ori_img_path, new_img_path, metric, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return metric(ori_x, new_x).item()


def get_init_latent(img_tensor, pipe, text_embeddings, pooled_prompt_embeds, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        pooled_prompt_embeds=pooled_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    return reversed_latents


import time
import gc
from transformers import T5EncoderModel
from diffusers import StableDiffusion3Pipeline


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


def get_text_embeddings(prompts):
    torch.manual_seed(42)
    id = "stabilityai/stable-diffusion-3-medium-diffusers"
    text_encoder = T5EncoderModel.from_pretrained(
        id,
        subfolder="text_encoder_3",
        # load_in_8bit=True,
        device_map="auto"
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        id,
        text_encoder_3=text_encoder,
        transformer=None,
        vae=None,
        device_map="balanced",
    )
    with torch.no_grad():
        prompt = prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None)

    del text_encoder
    del pipeline
    flush()

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # pipeline = StableDiffusion3Pipeline.from_pretrained(
    #     id,
    #     text_encoder=None,
    #     text_encoder_2=None,
    #     text_encoder_3=None,
    #     tokenizer=None,
    #     tokenizer_2=None,
    #     tokenizer_3=None,
    #     torch_dtype=torch.float16
    # ).to("cuda")
    # pipeline.set_progress_bar_config(disable=True)
    #
    # for _ in range(3):
    #     _ = pipeline(
    #         prompt_embeds=prompt_embeds.half(),
    #         negative_prompt_embeds=negative_prompt_embeds.half(),
    #         pooled_prompt_embeds=pooled_prompt_embeds.half(),
    #         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.half(),
    #     )
    # start = time.time()
    # for _ in range(10):
    #     _ = pipeline(
    #         prompt_embeds=prompt_embeds.half(),
    #         negative_prompt_embeds=negative_prompt_embeds.half(),
    #         pooled_prompt_embeds=pooled_prompt_embeds.half(),
    #         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.half(),
    #     )
    # end = time.time()
    # avg_inference_time = (end - start) / 10
    #
    # print(f"Average prompt encoding time: {avg_prompt_encoding_time:.3f} seconds.")
    # print(f"Average inference time: {avg_inference_time:.3f} seconds.")
    # print(f"Total time: {(avg_prompt_encoding_time + avg_inference_time):.3f} seconds.")
    # print(
    #     f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB"
    # )
    #
    # image = pipeline(
    #     prompt_embeds=prompt_embeds.half(),
    #     negative_prompt_embeds=negative_prompt_embeds.half(),
    #     pooled_prompt_embeds=pooled_prompt_embeds.half(),
    #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.half(),
    # ).images[0]
    # image.save("output_8bit.png")


def get_irf_latent(img_latents, pipe, text_embeddings, pooled_prompt_embeds, num_steps, gamma=0.0):
    # RF inversion from the given image
    # img_latents = pipe.get_image_latents(img_tensor, sample=False)
    irf_latents = pipe.interpolated_inversion(
        latents=img_latents,
        DTYPE=torch.float32,
        prompt_embeds=text_embeddings,
        pooled_prompt_embeds=pooled_prompt_embeds,
        gamma=gamma,
        num_steps=num_steps,
    )
    return irf_latents, img_latents
