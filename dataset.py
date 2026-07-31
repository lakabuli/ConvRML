import torch
import numpy as np
import os
import tifffile
import warnings

from torch.utils.data import Dataset
from natsort import natsorted
from skimage.transform import resize
from utils import apply_homography
from omegaconf import OmegaConf

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
HOMOGRAPHY_DIR = os.path.join(ABSOLUTE_PATH, "homography_matrices")

# Create the Dataset class for the DLMD dataset
class Mirflickr(Dataset):
    def __init__(self, root_path, train_type):
        super().__init__()
        root_dir = root_path
        self.data_dir = os.path.join(root_dir, "diffuser_images_npy")       # USE NUMPY BECAUSE TIFF FOLDERS DON'T HAVE ALL 24999 IMAGES
        self.target_dir = os.path.join(root_dir, "ground_truth_lensed_npy")

        full_data_list = os.listdir(self.data_dir)
        sorted_data_list = natsorted(full_data_list)    # Sort the lists in numerical order

        if train_type == "train":
            self.data_list = sorted_data_list[1000:]       # first image is missing in Mirflickr dataset
            self.target_list = sorted_data_list[1000:]
        else:
            self.data_list = sorted_data_list[:1000]       
            self.target_list = sorted_data_list[:1000]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_name = self.data_list[index][:-4]      # get image name without the .npy extension

        # numpy images were saved in BGR format instead of RGB, so we reverse the channel order
        image = np.load(os.path.join(self.data_dir, self.data_list[index]))[..., ::-1]
        target = np.load(os.path.join(self.target_dir, self.data_list[index]))[..., ::-1]

        image = np.clip(np.flipud(image)/0.9, 0,1)     # max of measurements is 0.9. Normalizing to range [0, 1]                                 
        target = np.clip(np.flipud(target), 0,1) 

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)

        # Move channels to the front
        image = torch.moveaxis(image, -1, 0)
        target = torch.moveaxis(target, -1, 0)

        return image, target, img_name
    
# Create the Dataset class for the PLD dataset
class ScalableDataset(Dataset):
    def __init__(self, root_dir, dataset, height, width, size, downsize_coeff, train_type="train", use_processed=False):
        super().__init__()

        if not use_processed:
            warnings.warn(
                "You are using raw data! Training time may be slower due to dataset preprocessing.",
                UserWarning)

            if dataset == 'rml':       
                self.data_dir = os.path.join(root_dir, "rml")  
                self.target_dir = os.path.join(root_dir, "4x_undistorted_ground_truth")     
                
                if downsize_coeff == 4:
                    self.homography_matrix = torch.load(os.path.join(HOMOGRAPHY_DIR, "GT2RML_homography_4x_2026_detached_numpy.npy"), weights_only=True) 
                else:
                    raise ValueError(f"{dataset} homography matrix for downsize coefficient of {downsize_coeff} was not found.")
                
            elif dataset == 'diffuser':
                self.data_dir = os.path.join(root_dir, "diffuser")  
                self.target_dir = os.path.join(root_dir, "4x_undistorted_ground_truth")         

                if downsize_coeff == 4:
                    self.homography_matrix = torch.load(os.path.join(HOMOGRAPHY_DIR, "GT2DC_homography_4x_2026_detached_numpy.npy"), weights_only=True)
                else:
                    raise ValueError(f"{dataset} homography matrix for downsize coefficient of {downsize_coeff} was not found.")
            else:
                raise ValueError(f"Dataset '{dataset}' does not exist. "
                        f"Available options are: 'mirflickr', 'rml', 'diffuser'.")
        else:
            if dataset == "rml":
                self.data_dir = os.path.join(root_dir, "4x_rml") 
                self.target_dir = os.path.join(root_dir, "4x_undistorted_GT2RML")
            elif dataset == "diffuser":
                self.data_dir = os.path.join(root_dir, "4x_diffuser") 
                self.target_dir = os.path.join(root_dir, "4x_undistorted_GT2DC")            
            else:
                raise ValueError(f"Dataset '{dataset}' does not exist. "
                        f"Available options are: 'mirflickr', 'rml', 'diffuser'.")
            self.homography_matrix = None

        self.use_processed = use_processed
        self.dataset = dataset
        self.downsize_coeff = downsize_coeff
        self.height = height
        self.width = width
        full_data_list_data = self.data_list = [f for f in os.listdir(self.data_dir) if f.endswith('.tiff') or f.endswith('.tif')]  # Filter for only tiff files
        sorted_data_list_data = natsorted(full_data_list_data) # Sort the lists in numerical order

        full_data_list_target = self.target_list = [f for f in os.listdir(self.target_dir) if f.endswith('.tiff') or f.endswith('.tif')]  # Filter for only tiff files
        sorted_data_list_target = natsorted(full_data_list_target) 

        full_len_dataset = len(full_data_list_target)

        if train_type == "train":                  
            self.data_list = sorted_data_list_data[5000:int(full_len_dataset * size)]      
            self.target_list = sorted_data_list_target[5000:int(full_len_dataset * size)]
        elif train_type == "val":
            self.data_list = sorted_data_list_data[1000:5000]
            self.target_list = sorted_data_list_target[1000:5000]
        elif train_type == "test":
            self.data_list = sorted_data_list_data[:1000]        # Use first 1000 images for testing, same as DLMD dataset
            self.target_list = sorted_data_list_target[:1000]
        else:
            raise ValueError("train_type must be 'train', 'val', or 'test'. ")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        img_name = self.data_list[index][:-5]     # get image name without the .tiff extension
        image = tifffile.imread(os.path.join(self.data_dir, self.data_list[index]))     
        target = tifffile.imread(os.path.join(self.target_dir, self.target_list[index]))

        # if images have four dimensions, remove alpha channel
        if image.shape[-1] == 4:
            image = image[..., :-1]
        if target.shape[-1] == 4:
            target = target[..., :-1]

        if np.max(image) > 1.0:
            image = (image/255).astype(np.float32)     # max of measurements is 255. Normalizing to range [0, 1]
        if np.max(target) > 1.0:
            target = (target/255).astype(np.float32)

        # Downsample the image by a factor of downsize_coeff to be around the same size as PLD 
        if image.shape[0] > self.height or image.shape[1] > self.width:         
            image = resize(image, (self.height, self.width), anti_aliasing=True).astype(np.float32) 
        image = np.clip(image, 0, 1).astype(np.float32)
        image = torch.from_numpy(image)
        image = torch.moveaxis(image, -1, 0)        # convert HWC to CHW

        if self.use_processed:
            target = np.clip(target, 0, 1).astype(np.float32)
            target = torch.from_numpy(target)
            target = torch.moveaxis(target, -1, 0)
        else:
            target = resize(target, (self.height, self.width), anti_aliasing=True).astype(np.float32)  
            target = apply_homography(target, self.homography_matrix)           # apply homography to compare image and target in the image space during training

        return image, target, img_name

# Create the dataloader based on which dataset is being used
def get_loader(config): 
    dataset = config.dataset.name
    batch_size = config.dataloader.batch_size
    num_workers = config.gpu_setup.num_workers
    height = config.model.input_height
    width = config.model.input_width
    downsize_coeff = config.dataset.downsize_factor
    use_processed = config.dataset.use_processed
    data_path = config.dataset.data_path
    size = OmegaConf.select(config, "dataset.size", default=0.5) 

    if dataset=="mirflickr":                   
        trainset = Mirflickr(data_path,train_type="train")
        valset = Mirflickr(data_path, train_type="val")
        testset = valset
    else:
        trainset = ScalableDataset(data_path, dataset, height, width, size, downsize_coeff, train_type="train", use_processed=use_processed)
        valset = ScalableDataset(data_path, dataset, height, width, size, downsize_coeff, train_type="val", use_processed=use_processed)
        testset = ScalableDataset(data_path, dataset, height, width, size, downsize_coeff, train_type="test", use_processed=use_processed)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
    

