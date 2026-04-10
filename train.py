import torch
from tqdm import tqdm 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import yaml
import argparse
import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from kornia.color import rgb_to_ycbcr

from model.model import MDFN
from data.sr_dataset import SRImageDataset
from utils.utils import *
from loss.content_loss import ContentLoss
from loss.structure_loss import StructureLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def setup_logging(config):
    """Setup logging and directories"""
    # Create directories
    log_dir = config['logging']['log_dir']
    base_checkpoint_dir = config['logging']['checkpoint_dir']
    
    # Create experiment-specific checkpoint directory
    checkpoint_dir = os.path.join(base_checkpoint_dir, config['name'])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{config['name']}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Setup tensorboard
    tensorboard_writer = None
    if config['logging']['tensorboard_enabled']:
        tb_dir = os.path.join(log_dir, 'tensorboard', f"{config['name']}_{timestamp}")
        tensorboard_writer = SummaryWriter(tb_dir)
        logging.info(f"Tensorboard logging enabled. Run: tensorboard --logdir {tb_dir}")
    
    return log_file, tensorboard_writer, checkpoint_dir

def validate_all_datasets(model, val_loaders, psnr_metric, ssim_metric, device, epoch=None, writer=None, config=None):
    """Validate on all validation datasets"""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            PSNR_val = 0
            SSIM_val = 0
            
            for data in val_loader:
                images, labels = data['lq'].to(device), data['gt'].to(device)
                outputs = tile_process(images, model, config)

                # Convert outputs and labels to YCbCr for PSNR/SSIM calculation
                outputs_y = rgb_to_ycbcr(outputs)[:, 0:1, :, :]
                labels_y = rgb_to_ycbcr(labels)[:, 0:1, :, :]

                PSNR_val += psnr_metric(outputs_y, labels_y) / len(val_loader)
                SSIM_val += ssim_metric(outputs_y, labels_y) / len(val_loader)

            results[val_name] = {
                'psnr': float(PSNR_val),
                'ssim': float(SSIM_val)
            }
            
            logging.info(f"Validation {val_name}: PSNR: {PSNR_val:.2f} - SSIM: {SSIM_val:.4f}")
            
            # Log to tensorboard
            if writer is not None and epoch is not None:
                writer.add_scalar(f'Validation_PSNR/{val_name}', PSNR_val, epoch)
                writer.add_scalar(f'Validation_SSIM/{val_name}', SSIM_val, epoch)
    
    return results

def save_checkpoint(model, optimizer, epoch, loss, metrics, checkpoint_dir, config, eval_metric):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config
    }
    
    # Save regular checkpoint with epoch number
    if eval_metric.startswith('epoch_'):
        checkpoint_path = os.path.join(checkpoint_dir, f"{config['name']}_{eval_metric}.pth")
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    else:
        # This is the best model - save both checkpoint and model state
        checkpoint_path = os.path.join(checkpoint_dir, f"{config['name']}_{eval_metric}_best.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save just the model state dict for easy loading (replace previous best)
        best_model_path = os.path.join(checkpoint_dir, f"{config['name']}_best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        
        logging.info(f"Best checkpoint saved: {checkpoint_path}")
        logging.info(f"Best model state saved: {best_model_path}")
    
    return checkpoint_path

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_datasets(config):
    """Create train, validation, and test datasets from config"""
    # Training dataset
    train_opt = config['datasets']['train'].copy()
    train_opt['phase'] = 'train'
    train_opt['scale'] = config['scale']
    trainset = SRImageDataset(train_opt)
    
    # Validation datasets
    val_datasets = {}
    for key in config['datasets']:
        if key.startswith('val_'):
            val_opt = config['datasets'][key].copy()
            val_opt['phase'] = 'val'
            val_opt['scale'] = config['scale']
            val_datasets[key] = SRImageDataset(val_opt)
    
    return trainset, val_datasets

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='/media/mirl/DATA/Projects/mdfn/config/MDFN_2X.yaml', help="Path to the configuration file")
parser.add_argument("-metric", "--eval_metric", default='psnr', help="Evaluation Metric used for training: PSNR or SSIM")

args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Setup logging and directories
log_file, writer, checkpoint_dir = setup_logging(config)

# Set manual seed for reproducibility
if 'manual_seed' in config:
    torch.manual_seed(config['manual_seed'])
    logging.info(f"Manual seed set to: {config['manual_seed']}")

# Log configuration
logging.info("=" * 50)
logging.info(f"Starting training with config: {args.config}")
logging.info(f"Model: {config['name']}")
logging.info(f"Scale factor: {config['scale']}x")
logging.info(f"Evaluation metric: {args.eval_metric}")
logging.info("=" * 50)

# Create datasets
trainset, val_datasets = create_datasets(config)
logging.info(f"Training set size: {len(trainset)}")
for name, dataset in val_datasets.items():
    logging.info(f"{name} size: {len(dataset)}")

# Create model from config
model = MDFN(
    scale_factor=config['network']['scale_factor'],
    channels=config['network']['channels'],
    base_channels=config['network']['base_channels'],
    laplacian_levels=config['network']['laplacian_levels'],
    image_size=config['network']['image_size']
)
image_size = config['network']['image_size']

model.to(device)
summary(model, input_size=(1, 3, image_size, image_size), col_names=("input_size","output_size","num_params","mult_adds"), depth=4)

# Log model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total parameters: {total_params:,}")
logging.info(f"Trainable parameters: {trainable_params:,}")

# Get training parameters from config
training_config = config['training']
batch_size = config['datasets']['train']['batch_size']
max_epochs = training_config['epochs']
early_stopping_patience = training_config['early_stopping_patience']

# Create model save path in checkpoint directory
best_model_path = os.path.join(checkpoint_dir, f"{config['name']}_best_model.pth")

# Create data loaders
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=config['datasets']['train'].get('use_shuffle', True), 
    num_workers=config['datasets']['train'].get('num_worker', 4), 
    pin_memory=True, 
    prefetch_factor=2, 
    persistent_workers=True
)

# Create validation data loaders
val_loaders = {}
for name, dataset in val_datasets.items():
    val_loaders[name] = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure().to(device)

content_loss = ContentLoss()
structure_loss = StructureLoss()

# Create optimizer from config
optimizer = optim.AdamW(
    model.parameters(), 
    lr=training_config['learning_rate'],
    betas=training_config['betas'],
    weight_decay=training_config['weight_decay'],
    amsgrad=training_config['amsgrad']
)

# Add ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=training_config.get('lr_patience', 10),
    verbose=True,
    min_lr=training_config.get('min_lr', 1e-7)
)

logging.info(f"Optimizer: AdamW")
logging.info(f"Learning rate: {training_config['learning_rate']}")
logging.info(f"Weight decay: {training_config['weight_decay']}")
logging.info(f"Betas: {training_config['betas']}")

best_metrics = {'psnr': 0, 'ssim': 0}
training_history = {'loss': [], 'psnr': [], 'ssim': []}
counter = 0
epoch = 0

# Log hyperparameters to tensorboard
if writer:
    writer.add_hparams({
        'learning_rate': training_config['learning_rate'],
        'weight_decay': training_config['weight_decay'],
        'batch_size': batch_size,
        'scale_factor': config['network']['scale_factor'],
        'base_channels': config['network']['base_channels']
    }, {})
logging.info("Starting training...")

while counter < early_stopping_patience and epoch < max_epochs:
    t0 = time.time()
    epoch_psnr = 0
    epoch_ssim = 0
    epoch_loss = 0
    
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{max_epochs}")

        for batch_idx, data in enumerate(tepoch):
            # Extract data in the format returned by SRImageDataset
            inputs, labels = data['lq'].to(device), data['gt'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = content_loss(outputs, labels) + structure_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # Convert outputs and labels to YCbCr for PSNR/SSIM calculation
            outputs_y = rgb_to_ycbcr(outputs)[:, 0:1, :, :]
            labels_y = rgb_to_ycbcr(labels)[:, 0:1, :, :]

            batch_psnr = psnr(outputs_y, labels_y) 
            batch_ssim = ssim(outputs_y, labels_y)

            epoch_loss += loss.item()
            epoch_psnr += batch_psnr.item()
            epoch_ssim += batch_ssim.item()
            
            tepoch.set_postfix_str(f" loss: {loss.item():.4f} - PSNR: {batch_psnr:.4f} - SSIM: {batch_ssim:.4f}")
            
            # Log to tensorboard every N batches
            if writer and batch_idx % config['logging']['log_interval'] == 0:
                global_step = epoch * len(trainloader) + batch_idx
                writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
                writer.add_scalar('Train/PSNR_batch', batch_psnr, global_step)
                writer.add_scalar('Train/SSIM_batch', batch_ssim, global_step)

    # Calculate average metrics for the epoch
    epoch_loss /= len(trainloader)
    epoch_psnr /= len(trainloader)
    epoch_ssim /= len(trainloader)
    
    training_history['loss'].append(epoch_loss)
    training_history['psnr'].append(epoch_psnr)
    training_history['ssim'].append(epoch_ssim)

    train_time = time.time() - t0

    # Validation on all datasets
    t1 = time.time()
    val_results = validate_all_datasets(model, val_loaders, psnr, ssim, device, epoch, writer, config)
    val_time = time.time() - t1
    
    # Log epoch results
    logging.info(f"Epoch {epoch+1}/{max_epochs}:")
    logging.info(f"  Train - Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f}, SSIM: {epoch_ssim:.4f}")
    for val_name, val_metrics in val_results.items():
        logging.info(f"  Val ({val_name}) - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
    logging.info(f"  Time - Train: {train_time:.0f}s, Val: {val_time:.0f}s")
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Train/Loss_epoch', epoch_loss, epoch)
        writer.add_scalar('Train/PSNR_epoch', epoch_psnr, epoch)
        writer.add_scalar('Train/SSIM_epoch', epoch_ssim, epoch)
        writer.add_scalar('Time/Train', train_time, epoch)
        writer.add_scalar('Time/Validation', val_time, epoch)

    # Extract metrics from the first validation dataset or a specified one
    main_val_dataset = config.get('training', {}).get('main_val_dataset', list(val_results.keys())[0])
    main_val_psnr = val_results[main_val_dataset]['psnr']
    main_val_ssim = val_results[main_val_dataset]['ssim']
    
    # Check for improvement and save best model
    current_metric = main_val_psnr if args.eval_metric == 'psnr' else main_val_ssim
    best_metric = best_metrics['psnr'] if args.eval_metric == 'psnr' else best_metrics['ssim']

    # Step the scheduler
    scheduler.step(current_metric)
    
    if current_metric > best_metric:
        best_metrics['psnr'] = main_val_psnr
        best_metrics['ssim'] = main_val_ssim
        
        # Save best model
        save_checkpoint(model, optimizer, epoch, epoch_loss, val_results, checkpoint_dir, config, args.eval_metric)
        logging.info(f"New best {args.eval_metric.upper()} on {main_val_dataset}: {current_metric:.4f}")
        counter = 0
    else:
        counter += 1
        logging.info(f"No improvement for {counter}/{early_stopping_patience} epochs")

    # Save regular checkpoint
    if (epoch + 1) % config['logging']['save_checkpoint_interval'] == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"{config['name']}_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, epoch_loss, val_results, checkpoint_dir, config, f"epoch_{epoch+1}")

    epoch += 1

logging.info('Finished Training')

# Load best model for final evaluation
best_model_path = os.path.join(checkpoint_dir, f"{config['name']}_best_model.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    logging.info(f"Loaded best model from: {best_model_path}")

logging.info("Final evaluation on all validation datasets:")
final_results = validate_all_datasets(model, val_loaders, psnr, ssim, device, config=config)

# Training summary
main_val_name = list(final_results.keys())[0]
final_psnr = final_results[main_val_name]['psnr']
final_ssim = final_results[main_val_name]['ssim']

logging.info("=" * 50)
logging.info("TRAINING COMPLETED")
logging.info("=" * 50)
logging.info(f"Configuration: {args.config}")
logging.info(f"Model: {config['name']}")
logging.info(f"Scale factor: {config['scale']}x")
logging.info(f"Total epochs: {epoch}")
logging.info(f"Evaluation metric: {args.eval_metric}")
logging.info(f"Best {args.eval_metric.upper()}: {best_metrics[args.eval_metric]:.4f}")
logging.info(f"Final Results - PSNR: {final_psnr:.2f} - SSIM: {final_ssim:.4f}")
logging.info(f"Checkpoints saved in: {checkpoint_dir}")
logging.info(f"Training log: {log_file}")

# Close tensorboard writer
if writer:
    writer.close()