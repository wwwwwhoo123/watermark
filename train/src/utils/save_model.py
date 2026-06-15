import torch
from torchvision import transforms


def save_checkpoint(vae, optimizer, epoch, loss, args, is_best=False, is_final=False):
    full_state_dict = vae.state_dict()
    decoder_state_dict = {k: v for k, v in full_state_dict.items() if "decoder" in k}

    checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "state_dict": decoder_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    if is_best:
        save_path = f"{args.output_dir}/vae_decoder_best.ckpt"
    elif is_final:
        save_path = f"{args.output_dir}/vae_decoder_final.ckpt"
    else:
        save_path = f"{args.output_dir}/vae_decoder_epoch_{epoch}.ckpt"

    torch.save(checkpoint, save_path)

    return save_path


def save_checkpoint_(vae, wm_encoder, wm_decoder, optimizer, epoch, loss, args, is_best=False, is_final=False):
    full_state_dict = vae.state_dict()
    decoder_state_dict = {k: v for k, v in full_state_dict.items() if "decoder" in k}

    vae_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "vae_decoder_state_dict": decoder_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    wm_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "wm_encoder_state_dict": wm_encoder.state_dict(),
        "wm_decoder_state_dict": wm_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    if is_best:
        vae_save_path = f"{args.output_dir}/vae_decoder_best.ckpt"
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_best.ckpt"
    elif is_final:
        vae_save_path = f"{args.output_dir}/vae_decoder_final.ckpt"
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_final.ckpt"
    else:
        vae_save_path = f"{args.output_dir}/vae_decoder_epoch_{epoch}.ckpt"
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_epoch_{epoch}.ckpt"

    torch.save(vae_checkpoint, vae_save_path)
    torch.save(wm_checkpoint, wm_save_path)

    return vae_save_path, wm_save_path


def save_checkpoint_adaptor(vae, wm_encoder, wm_decoder, adaptor, optimizer, epoch, loss, args, is_best=False, is_final=False):
    full_state_dict = vae.state_dict()
    decoder_state_dict = {k: v for k, v in full_state_dict.items() if "decoder" in k}

    vae_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "vae_decoder_state_dict": decoder_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    wm_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "wm_encoder_state_dict": wm_encoder.state_dict(),
        "wm_decoder_state_dict": wm_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    adaptor_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "adaptor_state_dict": adaptor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    if is_best:
        vae_save_path = f"{args.output_dir}/vae_decoder_best.ckpt"
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_best.ckpt"
        adaptor_save_path = f"{args.output_dir}/adaptor_best.ckpt"
    elif is_final:
        vae_save_path = f"{args.output_dir}/vae_decoder_final.ckpt"
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_final.ckpt"
        adaptor_save_path = f"{args.output_dir}/adaptor_final.ckpt"
    else:
        vae_save_path = f"{args.output_dir}/vae_decoder_epoch_{epoch}.ckpt"
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_epoch_{epoch}.ckpt"
        adaptor_save_path = f"{args.output_dir}/adaptor_epoch_{epoch}.ckpt"

    torch.save(vae_checkpoint, vae_save_path)
    torch.save(wm_checkpoint, wm_save_path)
    torch.save(adaptor_checkpoint, adaptor_save_path)

    return vae_save_path, wm_save_path, adaptor_save_path


def save_adaptor(wm_encoder, wm_decoder, adaptor, optimizer, epoch, loss, args, is_best=False, is_final=False):
    wm_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "wm_encoder_state_dict": wm_encoder.state_dict(),
        "wm_decoder_state_dict": wm_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    adaptor_checkpoint = {
        "epoch": epoch,
        "global_step": epoch * 1000,
        "adaptor_state_dict": adaptor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    if is_best:
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_best.ckpt"
        adaptor_save_path = f"{args.output_dir}/adaptor_best.ckpt"
    elif is_final:
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_final.ckpt"
        adaptor_save_path = f"{args.output_dir}/adaptor_final.ckpt"
    else:
        wm_save_path = f"{args.output_dir}/wm_encoder_decoder_epoch_{epoch}.ckpt"
        adaptor_save_path = f"{args.output_dir}/adaptor_epoch_{epoch}.ckpt"

    torch.save(wm_checkpoint, wm_save_path)
    torch.save(adaptor_checkpoint, adaptor_save_path)

    return wm_save_path, adaptor_save_path

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


def save_watermark_image(watermark_tensor, path):
    watermark_tensor = watermark_tensor.clone().detach().cpu()
    watermark_tensor = (watermark_tensor * 0.5 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(watermark_tensor[0])
    img.save(path)
