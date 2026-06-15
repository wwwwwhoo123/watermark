import argparse
from tqdm import tqdm
import warnings

from generation.generate import calculate_nc
from train.src.vine.load_finetune_mode import load_finetuned_models

warnings.filterwarnings("ignore")
from main.utils import *
from main.wmattacker import *
from main.attdiffusion import ReSDPipeline


def robust_test(device, model_id, watermark, output_path, attack):
    result_dir = output_path
    image_dir = f'{result_dir}/marked'
    save_dir = f'{result_dir}/attack'
    os.makedirs(save_dir, exist_ok=True)

    start_idx = 0
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    txt_file_path = os.path.join(result_dir, 'attack_results.txt')

    if attack:
        print('###############  Init Attackers   ###############')
        att_pipe = ReSDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float16,
            revision="fp16")
        att_pipe.set_progress_bar_config(disable=True)
        att_pipe.to(device)

        attackers = {
            'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
            'jpeg_attacker_10': JPEGAttacker(quality=10),
            'brightness_0.5': BrightnessAttacker(brightness=0.5),
            'contrast_0.5': ContrastAttacker(contrast=0.5),
            'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
            'Gaussian_blur_9': GaussianBlurAttacker(kernel_size=9, sigma=1),
            'bm3d': BM3DAttacker(),
            'crop': CropAttacker(),
        }

        for imagename in tqdm(image_files[start_idx:], desc="Processing Images", unit="image"):
            wm_img_path = os.path.join(image_dir, f"{imagename.split('.')[0]}.png")
            if not os.path.exists(wm_img_path):
                print(f'###########   Watermarked image not found for {imagename}. Skipping...  ############')
                continue
            # print('##############  Start Attacking...  ##############')
            for attacker_name, attacker in attackers.items():
                os.makedirs(os.path.join(save_dir, attacker_name), exist_ok=True)
                attacked_img_path = os.path.join(save_dir, attacker_name, os.path.basename(wm_img_path))
                attacker.attack([wm_img_path], [attacked_img_path])

    else:
        attackers = {
            'diff_attacker_60'
            'jpeg_attacker_10': (),
            'brightness_0.5': (),
            'contrast_0.5': (),
            'Gaussian_noise': (),
            'Gaussian_blur_9': (),
            'bm3d': (),
            'crop': (),

        }
        (vae,
         wm_encoder,
         wm_decoder) = load_finetuned_models(
            model_id,
            'train/src/result/stage2/vae_decoder_best.ckpt',
            'train/src/result/stage2/wm_encoder_decoder_best.ckpt')
        wm_decoder.to(device)

        with open(txt_file_path, 'w') as txtfile:
            header = ['Image Name'] + list(attackers.keys())
            txtfile.write('\t'.join(header) + '\n')

        accuracy_results = {attacker_name: [] for attacker_name in attackers.keys()}
        for imagename in tqdm(image_files[start_idx:], desc="Attacking Images", unit="image"):

            wm_img_path = os.path.join(image_dir, f"{imagename.split('.')[0]}.png")

            if not os.path.exists(wm_img_path):
                print(f'###########   Watermarked image not found for {imagename}. Skipping...  ############')
                continue

            results = []
            for attacker_name, attacker in attackers.items():
                os.makedirs(os.path.join(save_dir, attacker_name), exist_ok=True)
                attacked_img_path = os.path.join(save_dir, attacker_name, os.path.basename(wm_img_path))
                img_tensor = pil_to_tensor(Image.open(attacked_img_path).convert("RGB")) / 255
                img_tensor = img_tensor.unsqueeze(0).to(device)
                re_watermark = wm_decoder(img_tensor)
                NC = calculate_nc(watermark, re_watermark)

                results.append(str(NC))
                accuracy_results[attacker_name].append(NC)

            with open(txt_file_path, 'a') as txtfile:
                row = [imagename] + results
                txtfile.write('\t'.join(row) + '\n')

        for attacker_name, acc_list in accuracy_results.items():
            if acc_list:
                avg_accuracy = sum(acc_list) / len(acc_list)
                accuracy_results[attacker_name] = avg_accuracy
                print(f"{attacker_name}: {avg_accuracy:.2f}")
            else:
                accuracy_results[attacker_name] = None
                print(f"{attacker_name}: No results recorded.")

        with open(txt_file_path, 'a') as txtfile:
            txtfile.write("\nAverage Accuracy per Attacker:\n")
            for attacker_name, avg_accuracy in accuracy_results.items():
                if avg_accuracy is not None:
                    txtfile.write(f"{attacker_name}: {avg_accuracy:.2f}\n")
                else:
                    txtfile.write(f"{attacker_name}: No results recorded.\n")


def main():
    parser = argparse.ArgumentParser(description="robust test.")
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--output_path', type=str, default='./example/gen_adapter')

    args = parser.parse_args()
    model_id = args.model_id
    output_path = args.output_path
    attack = True
    # load model
    device = torch.device('cuda')
    # ==================================watermark==================================== #
    watermark = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])(
        Image.open("watermark/watermark128.png")
    ).unsqueeze(0).to(device)

    robust_test(device, model_id, watermark, output_path, attack)


if __name__ == '__main__':
    main()
