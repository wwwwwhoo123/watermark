import os
import sys
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils.compare import save_image_comparison
from utils.save_model import save_checkpoint_, calculate_nc, save_watermark_image
from utils.dataset import ImageDataset

p = "src/"
sys.path.append(p)
from train.src.vine.vine_turbo import initialize_vae_no_lora, VAE_encode, VAE_decode
from utils.loss import *
from wm_model.network import WatermarkEncoder, WatermarkDecoder
from network.Noise import Noise
from utils.wm_transform1 import WatermarkAugmenter


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ========================================load watermark==============================================#
    watermark_augmenter = WatermarkAugmenter(
        watermark_path="watermark/watermark128.png",
        batch_size=2,
        device=device
    )

    # ===========================================load VAE================================================#
    print(f"original model path: {args.model_path}")
    vae = initialize_vae_no_lora(path=args.model_path)
    vae.to(device)
    vae_encoder = VAE_encode(vae)
    vae_decoder = VAE_decode(vae)
    vae_encoder.to(device)
    vae_decoder.to(device)
    # ================================load WMencoder and Noise layer=====================================#
    wm_encoder = WatermarkEncoder().to(device)
    wm_decoder = WatermarkDecoder().to(device)
    noise_layers = ["Combined([Jpeg(30),GN(0.05),GF(1),Crop(0.5,0.5),Contrast(0.5),Bright(0.5)])"]
    attack = Noise(noise_layers).to(device)

    # ======================================Frozen vae_encoder===========================================#
    for name, param in vae.named_parameters():
        if 'encoder' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(args.finetune)

    # =======================================get trainable params========================================#
    vae_params = [p for p in vae.parameters() if p.requires_grad]
    wm_params = list(wm_encoder.parameters()) + list(wm_decoder.parameters())
    print(f"Total params: {sum(p.numel() for p in vae_params + wm_params)}")

    optimizer = optim.Adam([
        {'params': vae_params, 'lr': args.learning_rate},
        {'params': wm_params, 'lr': args.wm_lr}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    mse_loss = nn.MSELoss()
    prep_loss = LossPrep(device)

    # ===========================================data loading============================================#
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_data_dir = os.path.join(os.path.dirname(args.data_dir), "val")
    val_dataset = ImageDataset(val_data_dir, transform=transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    generate_dir = os.path.join(args.output_dir, "generate")
    watermark_dir = os.path.join(args.output_dir, "watermark")
    os.makedirs(generate_dir, exist_ok=True)
    os.makedirs(watermark_dir, exist_ok=True)

    # ==============================================training=============================================#
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_loss_img = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        watermark_augmenter.update_epoch(epoch)

        for batch_idx, (images, filenames) in enumerate(progress_bar):
            images = images.to(device)
            watermark = watermark_augmenter.get_augmented_watermark()
            optimizer.zero_grad()

            #### embedding watermark ####
            with torch.no_grad():
                latents = vae_encoder(images, direction="a2b")

            watermark_latent = wm_encoder(watermark)
            latents_wm = latents.clone()
            latents_wm[:, 0:4, :, :] = latents[:, 0:4, :, :] + args.alpha * watermark_latent

            reconstructed = vae_decoder(latents_wm, direction="a2b")
            attacked_reconstructed = attack([reconstructed, images])

            re_watermark = wm_decoder(reconstructed)
            attacked_re_watermark = wm_decoder(attacked_reconstructed)

            #### loss function ####
            loss1 = mse_loss(reconstructed, images)
            loss2 = prep_loss(images, reconstructed)
            loss3 = color_consistency_loss(images, reconstructed)

            loss_wm = mse_loss(watermark, attacked_re_watermark)
            loss_wm1 = mse_loss(watermark, re_watermark)

            nc_value = calculate_nc(watermark, re_watermark)

            loss_img = args.lambda_1 * loss1 + args.lambda_2 * loss2 + args.lambda_3 * loss3
            loss_water = args.lambda_wm * loss_wm + args.lambda_wm * loss_wm1
            loss = loss_img + loss_water

            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(vae_params + wm_params, args.clip_grad)

            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            avg_loss = epoch_loss / (batch_idx + 1)
            current_loss_img = loss_img.item()
            epoch_loss_img += current_loss_img
            avg_loss_img = epoch_loss_img / (batch_idx + 1)

            progress_bar.set_postfix({"loss": avg_loss, "loss_img": avg_loss_img, "nc_value": nc_value})

            if (batch_idx + 1) % args.save_interval == 0:
                with torch.no_grad():
                    sample_images = images[:min(4, images.shape[0])].cpu()
                    sample_reconstructed = reconstructed[:min(4, images.shape[0])].cpu()
                    comparison_path = os.path.join(generate_dir, f"epoch_{epoch + 1}_batch_{batch_idx + 1}.png")
                    save_image_comparison(sample_images, sample_reconstructed, comparison_path)

                    watermark_save_path = os.path.join(watermark_dir, f"epoch_{epoch + 1}_watermark.png")
                    save_watermark_image(attacked_re_watermark, watermark_save_path)

        avg_loss = epoch_loss / len(dataloader)

        progress_bar.close()

        # ============================================val===========================================#

        if (epoch + 1) % 2 == 0:
            val_nc_total = 0.0
            val_batches = 0

            with torch.no_grad():
                val_progress_bar = tqdm(val_dataloader, desc=f"Val {epoch + 1}/{args.epochs}")
                for val_images, _ in val_progress_bar:
                    val_images = val_images.to(device)
                    watermark = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])])(
                        Image.open("watermark/watermark128.png")
                    ).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device)
                    noise_layers_val = ["Combined([Jpeg(30)])"]
                    attack_val = Noise(noise_layers_val).to(device)

                    latents = vae_encoder(val_images, direction="a2b")
                    watermark_latent = wm_encoder(watermark)
                    latents_wm = latents.clone()
                    latents_wm[:, 0:4, :, :] = latents[:, 0:4, :, :] + args.alpha * watermark_latent

                    reconstructed = vae_decoder(latents_wm, direction="a2b")
                    attacked_reconstructed = attack_val([reconstructed, val_images])
                    attacked_re_watermark = wm_decoder(attacked_reconstructed)

                    nc_value = calculate_nc(watermark, attacked_re_watermark)
                    val_nc_total += nc_value
                    val_batches += 1

                none = wm_decoder(val_images)
                none_nc = calculate_nc(watermark, none)
                avg_val_nc = val_nc_total / len(val_dataloader)
                print(f"Validation {epoch + 1} completed. Avg val NC: {avg_val_nc:.6f}. NO_wm:  {none_nc:.6f}")
                val_watermark_save_path = os.path.join(watermark_dir, f"epoch_{epoch + 1}_val.png")
                save_watermark_image(none, val_watermark_save_path)

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint_(vae, wm_encoder, wm_decoder, optimizer, epoch + 1, avg_loss, args, is_best=True)

        if args.save_every_epoch:
            save_checkpoint_(vae, wm_encoder, wm_decoder, optimizer, epoch + 1, avg_loss, args)

        save_checkpoint_(vae, wm_encoder, wm_decoder, optimizer, args.epochs, avg_loss, args, is_final=True)


def save_config(args):
    config_file = os.path.join(args.output_dir, "config.txt")
    with open(config_file, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Config has been saved to {config_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training wm_encoder and wm_decoder")
    parser.add_argument("--data_dir", type=str, default="train/dataset/train")
    parser.add_argument("--output_dir", type=str, default="train/result/stage1")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--watermark_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--wm_lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float)

    parser.add_argument("--lambda_1", type=float, default=2.0)
    parser.add_argument("--lambda_2", type=float, default=0.1)
    parser.add_argument("--lambda_3", type=float, default=0.001)
    parser.add_argument("--lambda_wm", type=float, default=1.0)
    parser.add_argument("--lambda_penalty", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--finetune", type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    save_config(args)
    train(args)
