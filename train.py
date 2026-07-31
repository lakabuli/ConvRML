import os
import argparse
import wandb
import torch
import numpy as np

from omegaconf import OmegaConf
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from models.convnext import ConvRecon
from models.recon_transformer import Recon_Transformer
from models.swin_transformer import SwinRecon
from dataset import get_loader
from utils import MSE_Perceptual_Loss, load_wandb_visualization, crop_borders

entity_name = "" # TODO put in entity name for wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train Lensless Model")
    parser.add_argument('--config', type=str, default='./configs/DLMD_mirflickr/config_convnext.yaml', help='Path to the config file')     
    return parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_setup.gpu_visible_id)
    gpu_number = config.gpu_setup.gpu_num
       
    loss_type = "mse" if config.model.alpha == 0 else "mse_lpips" 

    # Save and run variables
    if config.dataset.name != 'mirflickr': 
        run_name = f"{config.model.type}_{config.dataset.size}_{config.dataset.name}_{loss_type}_{config.dataloader.batch_size}_x{config.dataset.downsize_factor}_downsize{'_full_fov_loss' if config.modes.full_fov_loss else ''}" 
    else:
        run_name = f"{config.model.type}_{config.dataset.name}_{loss_type}_{config.dataloader.batch_size}"    

    save_path = os.path.join(config.checkpoint.save_checkpoint_path, run_name)
    os.makedirs(save_path, exist_ok=True)  

    wandb_id = config.checkpoint.wandb_id if config.checkpoint.load_checkpoint else None
    
    if config.modes.wandb_on:
        run = wandb.init(project=config.checkpoint.project_name, 
                         name=run_name,
                         entity=entity_name, 
                         id=wandb_id,            
                         resume="allow",        
                         config={"learning_rate":config.dataloader.lr,      
                                "architecture": config.model.type,
                                "alpha": config.model.alpha,
                                "loss_type": loss_type,
                                "epochs":config.model.num_epochs,
                                "batch_size":config.dataloader.batch_size,
                                "n_channels": config.model.n_channels,
                                "warmup_epochs":config.dataloader.warmup_epochs})     

    # Get data loaders
    train_loader, val_loader, _ = get_loader(config)  

    # See if gpu is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(gpu_number) 
    else:
        device = torch.device("cpu")
    
    # Initialize model and move to GPU if available
    if config.model.type == 'convnext':
        model = ConvRecon(config.model.n_channels, 
                          config.model.output_height, 
                          config.model.output_width, 
                          model_size=config.model.size) 
    
    elif config.model.type == 'swin':          
        model = SwinRecon(n_channels=config.model.n_channels, 
                          img_size=(config.model.output_height,config.model.output_width), 
                          patch_size=config.model.patch_size, 
                          embed_dim=config.model.embed_dim, 
                          num_heads=config.model.num_heads_swin)
        
    elif config.model.type == 'basic_transformer':
        model = Recon_Transformer(config.model.output_height, 
                                  config.model.output_width, 
                                  config.model.patch_size, 
                                  config.model.n_channels, 
                                  config.model.num_heads_vit, 
                                  config.model.num_blocks, 
                                  config.model.embed_dim, 
                                  config.model.ffn_multiplier, 
                                  config.model.dropout_rate)
    else:
        raise TypeError(config.model.type, "is not a valid model type.")
    
    # Set up the model for multi-GPU training if specified
    if config.gpu_setup.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.dataloader.lr, weight_decay=1e-3)         
    criterion = MSE_Perceptual_Loss(device, alpha=config.model.alpha)       

    if config.checkpoint.load_checkpoint:
        checkpoint = torch.load(os.path.join(config.checkpoint.load_path, 'latest_model.pth'), map_location=device)

        # load all saved variables 
        start_epoch = checkpoint['epoch'] + 1       # start from the next epoch

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}.")
    else:
        best_loss = float('inf')
        start_epoch = 0

    train(model,
        train_loader, 
        val_loader, 
        optimizer,
        criterion,
        device,
        start_epoch,
        best_loss,
        save_path,
        config)
    if config.modes.wandb_on:
        run.finish()


# Includes both training and validation
def train(model, train_loader, val_loader, optimizer, criterion, device, start_epoch, best_loss, save_path, config):
    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/config.dataloader.warmup_epochs, end_factor=1.0, total_iters=config.dataloader.warmup_epochs-1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.model.num_epochs-config.dataloader.warmup_epochs, eta_min=1e-5) 

    # Training metrics
    train_psnr = PSNR(data_range=1.0).to(device)
    train_ssim = SSIM(data_range=1.0).to(device)
    lpips = LPIPS(net_type='alex', normalize=True).to(device)

    # Validation metrics
    val_psnr = PSNR(data_range=1.0).to(device)
    val_ssim = SSIM(data_range=1.0).to(device)

    for epoch in range(start_epoch, config.model.num_epochs):
        
        print(f'Start training epoch {epoch+1}/{config.model.num_epochs}...')
        train_psnr_out, train_mse_loss, train_ssim_out, train_lpips_out, train_loss_out = train_epoch(model, epoch, train_loader, optimizer, criterion, device, train_psnr, train_ssim, lpips, config) 
        val_psnr_out, val_mse_loss, val_ssim_out, val_lpips_out, val_loss_out = validate(model, val_loader, criterion, device, val_psnr, val_ssim, lpips, config)

        # Update wandb visualization with recon image sample
        sample_img, sample_target = load_wandb_visualization(config)        
        with torch.no_grad():
            model.eval()
            sample_out = model(sample_img.to(device))
            sample_out = torch.clamp(sample_out, 0, 1)       # in case pixel values are not in range [0, 1]    
            sample_out = sample_out.squeeze().cpu().numpy()
            sample_out = np.transpose(sample_out, (1, 2, 0))
            sample_out = crop_borders(sample_out, config)
        
        if config.modes.wandb_on:
            wandb.log({"training_mse_loss":train_mse_loss, 
                       "training_psnr": train_psnr_out,
                       "train_ssim": train_ssim_out,
                       "train_lpips": train_lpips_out,
                       "train_loss": train_loss_out,
                       "val_mse": val_mse_loss,
                       "val_psnr": val_psnr_out, 
                       "val_ssim": val_ssim_out,
                       "val_lpips": val_lpips_out,
                       "val_loss": val_loss_out,
                       "epoch":epoch+1, 
                       "learning rate":optimizer.param_groups[-1]['lr'],
                       "reconstructed butterfly": wandb.Image(sample_out),
                       "ground truth butterfly": wandb.Image(sample_target)})
        if epoch < config.dataloader.warmup_epochs:
            linear_warmup.step()
        else:
            scheduler.step()
        
        # save latest model for resuming training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, os.path.join(save_path, 'latest_model.pth'))

        # save best model
        if val_mse_loss < best_loss:
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
            print("New best model at epoch", epoch+1, "\n")
            best_loss = val_mse_loss

        # save intermediate models if save_intermediate is True
        if (epoch + 1) % config.checkpoint.save_freq == 0 and config.checkpoint.save_intermediate:
            torch.save(checkpoint, os.path.join(save_path, f'epoch_{epoch+1}_model.pth'))

        

def train_epoch(model, epoch, train_loader, optimizer, criterion, device, train_psnr, train_ssim, lpips, config, lpips_freq=50): 
    model.train()
    total_loss = 0
    total_mse = 0
    total_lpips = 0.0
    lpips_count = 0

    for step, batch in enumerate(tqdm(train_loader)):
        input, target, _ = batch
        input, target = input.to(device), target.to(device)  
        output = model(input)                                             
        optimizer.zero_grad()

        # Calculate loss on cropped image results
        if not config.modes.full_fov_loss: 
            output, target = crop_borders((output, target), config, batch=True)

        loss = criterion(output.squeeze(), target.squeeze())                                  
        loss.backward()
        optimizer.step()        
        total_loss += loss.item() 
        
        with torch.no_grad():                                  
            output_detached = output.detach()   # detach from computational graph
            target_detached = target.detach()

            train_psnr.update(output_detached, target_detached)        
            train_ssim.update(output_detached, target_detached)    
            total_mse += torch.nn.functional.mse_loss(output_detached, target_detached, reduction="mean").item()      

            # LPIPS only accepts images in range [0, 1] or [-1, 1]
            if step % lpips_freq == 0:
                clipped_out = torch.clamp(output_detached, min=0.0, max=1.0)          
                total_lpips += lpips(clipped_out, target_detached).item()
                lpips_count += 1
        
    avg_loss = total_loss / len(train_loader)          
    avg_mse = total_mse / len(train_loader)
    avg_lpips = total_lpips / lpips_count if lpips_count > 0 else 0.0
    avg_psnr = train_psnr.compute()
    avg_ssim = train_ssim.compute() 
    train_psnr.reset()            # for next epoch
    train_ssim.reset()
    lpips.reset() 
    print(f'Epoch {epoch+1}/{config.model.num_epochs}, Train Loss: {avg_loss}, Train PSNR {avg_psnr}, Train SSIM {avg_ssim}, Train MSE {avg_mse}, Train LPIPS {avg_lpips}')
    return avg_psnr, avg_mse, avg_ssim, avg_lpips, avg_loss  


def validate(model,val_loader, criterion, device, val_psnr, val_ssim, lpips, config):                
    model.eval()
    print("Starting validation...")
    with torch.no_grad():
        total_loss = 0.0
        total_mse = 0
        total_lpips = 0.0

        for step, batch in enumerate(tqdm(val_loader)):
            input, target, _ = batch
            input, target = input.to(device), target.to(device)
            output = model(input)

            if not config.modes.full_fov_loss: 
                output, target = crop_borders((output, target), config, batch=True)

            loss = criterion(output.squeeze(), target)  
            total_loss += loss.item() 

            output_detached = output.detach()
            target_detached = target.detach()

            val_psnr.update(output_detached, target_detached)
            val_ssim.update(output_detached, target_detached)
            total_mse += torch.nn.functional.mse_loss(output_detached, target_detached, reduction="mean").item()      

            clipped_out = torch.clamp(output_detached, min=0.0, max=1.0)
            total_lpips += lpips(clipped_out, target_detached).item()

        avg_loss = total_loss/len(val_loader)                     
        avg_mse = total_mse / len(val_loader)
        avg_lpips = total_lpips / len(val_loader)
        avg_psnr = val_psnr.compute()
        avg_ssim = val_ssim.compute()  
        val_psnr.reset()          
        val_ssim.reset()
        lpips.reset() 
        print(f'Val Loss: {avg_loss}, Val PSNR: {avg_psnr}, Val SSIM: {avg_ssim}, Val MSE: {avg_mse}, Val LPIPS: {avg_lpips} \n')
        return avg_psnr, avg_mse, avg_ssim, avg_lpips, avg_loss

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user! Cleaning up...")
    finally:
        if wandb.run is not None:
            print("Closing wandb run...")
            wandb.finish()