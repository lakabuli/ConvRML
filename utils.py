import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as lpips_fn
import tifffile
import numpy as np
import kornia.geometry.transform as transform
import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
HOMOGRAPHY_DIR = os.path.join(ABSOLUTE_PATH, "homography_matrices")
    
# Load an image pair in imager space for wandb visualization
def load_wandb_visualization(config):
    dataset = config.dataset.name
    img_number = config.modes.img_number

    if dataset == "mirflickr":
        root_dir = config.dataset.data_path
        return load_image_pair_mirflickr(root_dir, img_number)
    elif dataset == "rml" or dataset == "diffuser":   
        root_dir = config.dataset.data_path
        return load_image_pair_rml_diffuser(root_dir, img_number, dataset, config)
    else:
        raise ValueError(f"Dataset {dataset} not recognized for wandb visualization.")

def load_image_pair_mirflickr(root_dir, img_number):
    ground = np.load(os.path.join(root_dir, f"ground_truth_lensed_npy/im{img_number}.npy"))[..., ::-1]
    diffuser = np.load(os.path.join(root_dir, f"diffuser_images_npy/im{img_number}.npy"))[..., ::-1]
    
    diffuser = np.clip(np.flipud(diffuser)/0.9, 0,1)     # max of measurements is 0.9. Normalizing to range [0, 1]  
    diffuser = torch.from_numpy(diffuser)
    diffuser = torch.moveaxis(diffuser, -1, 0)        # Move channels to the front
    diffuser = diffuser.unsqueeze(0)   

    ground = np.clip(np.flipud(ground), 0,1) 
    ground = ground[60:,62:-38,:]

    return diffuser, ground

def load_image_pair_rml_diffuser(root_dir,img_number, dataset, config):

    if dataset == "diffuser":
        image = tifffile.imread(os.path.join(root_dir, f"4x_diffuser/4x_img_{img_number}_cam_0.tiff")) 
        target = tifffile.imread(os.path.join(root_dir, f"4x_undistorted_GT2DC/warped_4x_undistorted_img_{img_number}_cam_2.tiff"))

    else:
        image = tifffile.imread(os.path.join(root_dir, f"4x_rml/4x_img_{img_number}_cam_1.tiff"))
        target = tifffile.imread(os.path.join(root_dir, f"4x_undistorted_GT2RML/warped_4x_undistorted_img_{img_number}_cam_2.tiff"))

     # images have four dimensions, remove alpha channel
    if image.shape[-1] == 4:
        image = image[..., :-1]
    if target.shape[-1] == 4:
        target = target[..., :-1]

    target = np.ascontiguousarray(target)
    target = crop_borders(target, config)
    
    image = (image/255).astype(np.float32)
    image = np.clip(image, 0,1)
    image = torch.from_numpy(image)
    image = torch.moveaxis(image, -1, 0)
    image = image.unsqueeze(0)          # Add batch dimension

    return image, target

# Crop the borders of an image in imager space based on the dataset and downsize factor.
def crop_borders(img, config, batch=False):

    dataset = config.dataset.name
    downsize_coeff = config.dataset.downsize_factor

    if batch:
        output, target = img
        assert output.ndim == 4 and target.ndim == 4, \
            f"Expected batched tensors of shape (B,C,H,W), got {output.shape} and {target.shape}"

        if dataset == "mirflickr":
            output = output[:,:,60:,62:-38]
            target = target[:,:,60:,62:-38]
        elif dataset == "diffuser": 
            # crop positions in diffuser imager space   
            if downsize_coeff == 8:             
                output = output[:,:,:134, 56:191]     
                target = target[:,:,:134, 56:191]  
            elif downsize_coeff == 4:
                output = output[:,:,13:289, 104:380]
                target = target[:,:,13:289, 104:380]      
        elif dataset == "rml":
            # crop positions in rml imager space
            if downsize_coeff == 8:
                output = output[:,:,14:134, 61:181]
                target = target[:,:,14:134, 61:181]
            elif downsize_coeff == 4:
                output = output[:,:,31:270, 128:367]
                target = target[:,:,31:270, 128:367]        
        return output, target

    assert img.ndim == 3, f"Expected (H,W,C) or (C,H,W), got {img.shape}"

    if dataset == "mirflickr":
        img = img[60:,62:-38,:]
    elif dataset == "diffuser":
        if downsize_coeff == 8:
            img = img[:134, 56:191,:]  
        elif downsize_coeff == 4:
            img = img[13:289, 104:380, :]
    elif dataset == "rml":
        if downsize_coeff == 8:
            img = img[14:134, 61:181,:]
        elif downsize_coeff == 4:
            img = img[31:270, 128:367, :]
    return img

# Apply a homography to transform an image into imager space or ground truth space
def apply_homography(img, homography_matrix):
    # Convert to tensor and ensure array is stored contiguously for faster operations
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(torch.float32)
    img = img.contiguous()

    img = img.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
    img = img.contiguous() # Ensure contiguous after permute
    img = img[None, ...]  # Add batch dimension

    with torch.no_grad():
        warped_img = transform.warp_perspective(img.float(), homography_matrix, 
                                                dsize=(img.shape[2], img.shape[3])).squeeze()        

        # Normalize to 0-1
        warped_img = warped_img / torch.max(warped_img)

    return warped_img

# Combine perceptual evaluation metric (SSIM or LPIPS) with MSE loss to create a custom loss function
class MSE_Perceptual_Loss(nn.Module):
    def __init__(self, device, alpha=0.5, is_lpips=True):
        super().__init__()
        self.alpha = alpha
        self.is_lpips = is_lpips

        self.mse_metric = nn.MSELoss()
        if alpha > 0:
            #self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)   # normalize=True expects inputs in range [0, 1]

    def forward(self, input, target):      
        mse_val = self.mse_metric(input, target)
        perceptual_loss = 0
        if self.alpha == 0:
            total_loss = mse_val 
            return total_loss
        else:
            if self.is_lpips:
                # LPIPs expects inputs in the range [0, 1]
                input_min = torch.min(input)
                normalized_input = (input - input_min)/ (torch.max(input) - input_min)
                perceptual_loss = lpips_fn(normalized_input, target, net_type='alex', normalize=True)
            else:
                ssim_val = self.ssim_metric(input, target)
                perceptual_loss = 1 - ssim_val          # We are minimizing the loss, so we need to take the complement since a high SSIM value is desirable.

            total_loss = mse_val * (1 - self.alpha) + perceptual_loss * self.alpha
            return total_loss 