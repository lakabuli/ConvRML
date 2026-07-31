"""
Run inference for a trained model and calculate reconstruction metrics.

If `infer.save_metrics` is set to `True` in the config file, the script saves
the computed metrics in a dictionary. 

If `infer.save_ids` is not empty, the script saves the reconstructed images for 
the specified image IDs.
"""

import argparse
import os
import re
import torch
import numpy as np
import kornia.geometry.transform as transform

from omegaconf import OmegaConf
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from dataset import get_loader
from models.convnext import ConvRecon
from models.recon_transformer import Recon_Transformer
from models.swin_transformer import SwinRecon

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
HOMOGRAPHY_DIR = os.path.join(ABSOLUTE_PATH, "homography_matrices")
rml_homography_matrix_path = os.path.join(HOMOGRAPHY_DIR, "GT2RML_homography_4x_2026_detached_numpy.npy") # path to rml homography matrix
diffuser_homography_matrix_path = os.path.join(HOMOGRAPHY_DIR, "GT2DC_homography_4x_2026_detached_numpy.npy") # path to diffuser homography matrix

# Extract image ID number from image name. The PLD images have multiple numbers in the name which must be filtered.
def _extract_image_id(name: str):
    """
    Extract the dataset image id from `img_name`.

    Common patterns in this repo:
    - Mirflickr: "im64" -> 64
    - RML/Diffuser processed TIFFs: "4x_img_64_cam_1" -> 64  (NOT 4)
    """
    # Prefer explicit tokens over "first integer" to avoid matching the "4" in "4x_..."
    patterns = [
        r"(?:^|_)img_(\d+)(?:_|$)",  # ...img_64...  (RML/Diffuser)
        r"(?:^|_)im(\d+)(?:_|$)",    # im64 or ..._im64... (Mirflickr)
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))

    # Fallback: last resort (kept for robustness, but prefer naming patterns above)
    m = re.search(r"\d+", name)
    return int(m.group(0)) if m else None

# Calculate the confidence interval for our metrics
def confidence_interval_list(data_list, confidence_interval=0.95):
    error_lo = np.percentile(data_list, 100 * (1 - confidence_interval) / 2)
    error_hi = np.percentile(data_list, 100 * (1 - (1 - confidence_interval) / 2))
    mean = np.mean(data_list)
    return error_lo, error_hi, mean


def parse_args():
    parser = argparse.ArgumentParser(description="Infer Lensless Model")
    parser.add_argument("--config", type=str, default="config_convnext.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.infer.gpu_visible_id)
    gpu_num = config.infer.gpu_num

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_num)
    else:
        device = torch.device("cpu")

    loss_type = "mse" if config.model.alpha == 0 else "mse_lpips"
    if config.dataset.name != "mirflickr":
        run_name = (
            f"{config.model.type}_{config.dataset.size}_{config.dataset.name}_{loss_type}_"
            f"{config.dataloader.batch_size}_x{config.dataset.downsize_factor}_downsize"
            f"{'_full_fov_loss' if config.modes.full_fov_loss else ''}"
        )
    else:
        run_name = f"{config.model.type}_{config.dataset.name}_{loss_type}_{config.dataloader.batch_size}"

    # Where to save the inference images/metrics
    save_dir = os.path.join(config.infer.save_infer_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_id_list_raw = getattr(config.infer, "save_ids", None) or []
    save_id_list = [int(x) for x in save_id_list_raw]

    if config.model.type == "convnext":
        model = ConvRecon(
            config.model.n_channels,
            config.model.output_height,
            config.model.output_width,
            model_size=config.model.size,
        )
    elif config.model.type == "swin":
        model = SwinRecon(
            n_channels=config.model.n_channels,
            img_size=(config.model.output_height, config.model.output_width),
            patch_size=config.model.patch_size,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads_swin,
        )
    elif config.model.type == "basic_transformer":
        model = Recon_Transformer(
            config.model.output_height,
            config.model.output_width,
            config.model.patch_size,
            config.model.n_channels,
            config.model.num_heads_vit,
            config.model.num_blocks,
            config.model.embed_dim,
            config.model.ffn_multiplier,
            config.model.dropout_rate,
        )
    else:
        raise TypeError(config.model.type, "is not a valid model type.")

    if config.gpu_setup.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Load checkpoint dict (matches train.py saving format)
    load_root = os.path.join(config.checkpoint.save_checkpoint_path, run_name)
    checkpoint_path = os.path.join(load_root, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # Test DataLoader batch size: infer-only (does not change training; train.py still uses dataloader.batch_size)
    infer_bs = OmegaConf.select(config, "infer.batch_size")
    loader_config = (
        OmegaConf.merge(config, OmegaConf.create({"dataloader": {"batch_size": int(infer_bs)}}),)
        if infer_bs is not None
        else config
    )
    _, _, test_loader = get_loader(loader_config)

    # load homography matrices
    dataset = config.dataset.name
    if dataset != 'mirflickr':
        if dataset == "rml":            
            homography_matrix = torch.load(rml_homography_matrix_path, weights_only=True) 
        elif dataset == 'diffuser':
            homography_matrix = torch.load(diffuser_homography_matrix_path, weights_only=True)

        # Invert to get RML/diffuser -> GT warp
        imager_to_gt_homography_matrix = torch.inverse(homography_matrix).to(device)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LPIPS(net_type="alex", normalize=True).to(device)

    psnr_list = []
    ssim_list = []
    mse_list = []
    lpips_list = []
    num_batches = 0
    
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            input, target, img_name = batch
            input, target = input.to(device), target.to(device)
            output = model(input)

            if dataset != 'mirflickr':
                output = transform.warp_perspective(output.float(), imager_to_gt_homography_matrix, 
                                                dsize=(output.shape[2], output.shape[3]))        
                
                # Originally target was in imager space. Needs to be warped back to GT space.
                target = transform.warp_perspective(target.float(), imager_to_gt_homography_matrix,        
                                                    dsize=(target.shape[2], target.shape[3]))
            
            output_uncropped = torch.clamp(output, 0, 1)
            target = torch.clamp(target, 0, 1)

            if dataset != 'mirflickr':
                # standardized crop positions in GT imager space rather than in diffuser or rml imager space
                output = output_uncropped[:,:, 52:266, 129:343]           
                target = target[:,:, 52:266, 129:343]
            else:
                # mirflickr crop positions, only one imager here
                output = output_uncropped[:,:,60:,62:-38]
                target = target[:,:,60:,62:-38]

            # Save only when img_id is listed in infer.save_ids (empty list means don't save anything).
            name = img_name[0]
            img_id = _extract_image_id(name)
            should_save = img_id is not None and img_id in save_id_list
            if should_save:
                if config.infer.save_uncropped:
                    out_uncropped = output_uncropped.squeeze().cpu().detach().numpy()
                    out_uncropped = np.moveaxis(out_uncropped, 0, -1)
                    np.save(os.path.join(save_dir, str(img_id) + "_uncropped.npy"), out_uncropped)

                out_img = output.squeeze().cpu().detach().numpy()
                out_img = np.moveaxis(out_img, 0, -1)
                np.save(os.path.join(save_dir, str(img_id) + ".npy"), out_img)

                out_target = target.squeeze().cpu().detach().numpy()
                out_target = np.moveaxis(out_target, 0, -1)
                np.save(os.path.join(save_dir, str(img_id) + "_gt.npy"), out_target)

            psnr_val = psnr(output, target)
            ssim_val = ssim(output, target)
            mse_val = torch.nn.functional.mse_loss(output, target, reduction="mean")  
            clipped_out = torch.clamp(output, min=0.0, max=1.0)
            lpips_val = lpips(clipped_out, target)

            # Prevent metric-state accumulation across batches
            psnr.reset()
            ssim.reset()
            lpips.reset()

            psnr_list.append(psnr_val.item())
            ssim_list.append(ssim_val.item())
            mse_list.append(mse_val.item())
            lpips_list.append(lpips_val.item())

            num_batches += 1

    assert len(mse_list) == len(test_loader.dataset)
    mean_psnr = sum(psnr_list) / len(test_loader.dataset)
    mean_ssim = sum(ssim_list) / len(test_loader.dataset)
    mean_mse = sum(mse_list) / len(test_loader.dataset)
    mean_lpips = sum(lpips_list) / len(test_loader.dataset)

    confidence_interval_mse = confidence_interval_list(mse_list)
    confidence_interval_lpips = confidence_interval_list(lpips_list)
    confidence_interval_psnr = confidence_interval_list(psnr_list)
    confidence_interval_ssim = confidence_interval_list(ssim_list)

    final_results = {
        "avg_mse": mean_mse,
        "avg_lpips": mean_lpips,
        "avg_psnr": mean_psnr,
        "avg_ssim": mean_ssim,
        "mse_per_batch": mse_list,
        "lpips_per_batch": lpips_list,
        "psnr_per_batch": psnr_list,
        "ssim_per_batch": ssim_list,
        "confidence_interval_mse": confidence_interval_mse,
        "confidence_interval_lpips": confidence_interval_lpips,
        "confidence_interval_psnr": confidence_interval_psnr,
        "confidence_interval_ssim": confidence_interval_ssim}
    
    if config.infer.save_metrics:
        np.save(f"{save_dir}/metrics_list.npy", final_results)

    # Print the results
    print("PSNR: ", mean_psnr)       
    print("SSIM: ", mean_ssim)
    print("MSE: ", mean_mse)
    print("LPIPS: ", mean_lpips)

if __name__ == "__main__":
    main()