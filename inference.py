import argparse
import os
import yaml
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from torchvision.utils import save_image

from model.model import MDFN

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def tile_process(images, model, config):
    _, _, h_old, w_old = images.size()
    window_size = config['tile']['window_size']
    tile_size = config['tile']['tile_size']
    h_pad = (h_old // window_size + 1) * window_size - h_old if h_old % window_size != 0 else 0
    w_pad = (w_old // window_size + 1) * window_size - w_old if w_old % window_size != 0 else 0
    
    img = torch.cat([images, torch.flip(images, [2])], 2)[:, :, : h_old + h_pad, :] if h_pad > 0 else images
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad] if w_pad > 0 else img
        
    b, c, h, w = img.size()
    tile = min(tile_size, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    tile_overlap = config['tile']['tile_overlap']
    sf = config['scale']

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile] if h > tile else [0]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile] if w > tile else [0]
    
    E = torch.zeros(b, c, h * sf, w * sf).type_as(img)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            with torch.no_grad():
                out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)
            E[
                ...,
                h_idx * sf : (h_idx + tile) * sf,
                w_idx * sf : (w_idx + tile) * sf,
            ].add_(out_patch)
            W[
                ...,
                h_idx * sf : (h_idx + tile) * sf,
                w_idx * sf : (w_idx + tile) * sf,
            ].add_(out_patch_mask)
            
    output = E.div_(W)
    output = output[..., : h_old * sf, : w_old * sf]
    return output

def main():
    parser = argparse.ArgumentParser(description="MDFN Super-Resolution Inference")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input LR image")
    parser.add_argument("-s", "--scale", type=int, required=True, choices=[2, 4], help="Super-resolution scale factor (2 or 4)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the output HR image")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config_path = f"config/MDFN_{args.scale}X.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    config = load_config(config_path)

    # Initialize model
    model = MDFN(
        scale_factor=config['network']['scale_factor'],
        channels=config['network']['channels'],
        base_channels=config['network']['base_channels'],
        laplacian_levels=config['network'].get('laplacian_levels', 5),
        image_size=config['network'].get('image_size', 64)
    )

    # Load checkpoint
    # Assuming standard checkpoint path, modify if necessary
    checkpoint_path = f"checkpoints/MDFN_{args.scale}X.pth"
    # Fallback to checking the config log dirs or try to find a best model if standard path fails
    best_model_path = os.path.join(config['logging']['checkpoint_dir'], config['name'], f"{config['name']}_best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint_path = best_model_path
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Proceeding with untrained model for testing.")

    model.to(device)
    model.eval()

    # Load image
    print(f"Loading image from {args.input}")
    try:
        img = Image.open(args.input).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Transform image to tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        output_tensor = tile_process(img_tensor, model, config)

    # Save output image
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_image(output_tensor.squeeze(0), args.output)
    print(f"Saved super-resolved image to {args.output}")

if __name__ == "__main__":
    main()
