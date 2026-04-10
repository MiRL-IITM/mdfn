import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms
from torchinfo import summary
from datasets import load_dataset
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import kornia
from tqdm import tqdm
from utils.niqe import calculate_niqe
from thop import profile, clever_format
import os
from model.model import MDFN
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

# Scale factor configuration
SCALE_FACTOR = 4

if SCALE_FACTOR not in [2, 4]:
    raise ValueError(f"Scale factor {SCALE_FACTOR} not supported. Use 2 or 4.")
elif SCALE_FACTOR == 2:
    print("Testing MDFN for 2x super-resolution...")
    WINDOW_SIZE = 16
    TILE_SIZE = 64
    TILE_OVERLAP = 32
    model = MDFN(scale_factor=SCALE_FACTOR, channels=3, base_channels=64, laplacian_levels=5, image_size=64)
else:
    print("Testing MDFN for 4x super-resolution...")
    WINDOW_SIZE = 8
    TILE_SIZE = 32
    TILE_OVERLAP = 16
    model = MDFN(scale_factor=SCALE_FACTOR, channels=3, base_channels=64, laplacian_levels=5, image_size=32)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Spatial: {(count_parameters(model.spatial_block) + count_parameters(model.refinement))/1e6}")
print(f"Laplacian: {(count_parameters(model.spatial_block) + count_parameters(model.laplacian_block) + count_parameters(model.refinement))/1e6}")
print(f"Fourier: {(count_parameters(model.spatial_block) + count_parameters(model.fourier_block) + count_parameters(model.refinement))/1e6}")
print(f"Total trainable parameters: {count_parameters(model)/1e6}")


model.to(device)
summary(model, input_size=(1, 3, 64, 64) if SCALE_FACTOR == 2 else (1, 3, 32, 32), col_names= ("input_size","output_size","num_params","mult_adds"), depth = 4)

def tile_process(images, model, scale):
    _, _, h_old, w_old = images.size()
    h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
    w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
    img = torch.cat([images, torch.flip(images, [2])], 2)[
        :, :, : h_old + h_pad, :
    ]
    img = torch.cat([img, torch.flip(img, [3])], 3)[
        :, :, :, : w_old + w_pad
    ]
    b, c, h, w = img.size()
    tile = min(TILE_SIZE, h, w)
    assert tile % WINDOW_SIZE == 0, "tile size should be a multiple of window_size"
    
    sf = scale
    stride = tile - TILE_OVERLAP
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
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

class SuperResolutionDataset(Dataset):
    def __init__(self, dataset, transform_img=None, transform_target=None):
        self.dataset = dataset
        self.transform_img = transform_img
        self.transform_target = transform_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        lr = "lr"

        img_path = self.dataset[idx][lr]
        image = Image.open(img_path)

        if image.mode == "L":
          image = image.convert('RGB')
        
        if lr == 'lr':

            image_size = image.size

            h , w = SCALE_FACTOR*image_size[0], SCALE_FACTOR*image_size[1]


        target_path = self.dataset[idx]["hr"] 
        target = Image.open(target_path)

        if target.mode == "L":
          target = target.convert('RGB')
        
        if lr == 'lr':
            target_size = target.size

            if target_size[0] != h or target_size[1] != w:
                t  = transforms.Resize([w, h])
                target = t(target)

        

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_target:
            target = self.transform_target(target)

        return image, target

transform_img_set5 = transforms.Compose([transforms.ToTensor()])
transform_target_set5 = transforms.Compose([transforms.ToTensor()])

transform_img_bsd100 = transforms.Compose([transforms.ToTensor()])
transform_target_bsd100 = transforms.Compose([transforms.ToTensor()])

transform_img_urban100 = transforms.Compose([transforms.ToTensor()])
transform_target_urban100 = transforms.Compose([transforms.ToTensor()])

transform_img_set14 = transforms.Compose([transforms.ToTensor()])
transform_target_set14 = transforms.Compose([transforms.ToTensor()])

transform_img_div2k = transforms.Compose([transforms.ToTensor()])
transform_target_div2k = transforms.Compose([transforms.ToTensor()])

transform_img = [transform_img_set5, transform_img_bsd100, transform_img_urban100, transform_img_set14, transform_img_div2k]
transform_target = [transform_target_set5, transform_target_bsd100, transform_target_urban100, transform_target_set14, transform_target_div2k]

# Load datasets based on scale factor
if SCALE_FACTOR == 2:
    dataset_set5      = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
    dataset_bsd100    = load_dataset('eugenesiow/BSD100', 'bicubic_x2', split='validation')
    dataset_Urban100  = load_dataset('eugenesiow/Urban100', 'bicubic_x2', split='validation')
    dataset_set14     = load_dataset('eugenesiow/Set14', 'bicubic_x2', split='validation')
elif SCALE_FACTOR == 4:
    dataset_set5      = load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')
    dataset_bsd100    = load_dataset('eugenesiow/BSD100', 'bicubic_x4', split='validation')
    dataset_Urban100  = load_dataset('eugenesiow/Urban100', 'bicubic_x4', split='validation')
    dataset_set14     = load_dataset('eugenesiow/Set14', 'bicubic_x4', split='validation')
else:
    raise ValueError(f"Scale factor {SCALE_FACTOR} not supported. Use 2 or 4.")

data = [dataset_set5, dataset_bsd100, dataset_Urban100, dataset_set14]
name = ["Set5", "BSD100", "Urban100", "Set14"]

# Model checkpoint paths for different scale factors
checkpoint_paths = {
    2: f'checkpoints/MDFN_2X',
    4: f'checkpoints/MDFN_4X'
}

path = checkpoint_paths[SCALE_FACTOR]
print(f"Loading model for {SCALE_FACTOR}x scale from: {path}")
model.load_state_dict(torch.load(path))

# Calculate FLOPS using thop (after loading checkpoint to avoid state_dict conflicts)
input_tensor = (torch.randn(1, 3, 32, 32) if SCALE_FACTOR == 4 else torch.randn(1, 3, 64, 64)).to(device)
flops, params = profile(model, inputs=(input_tensor,), verbose=False)
flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

print(f"\n{'='*60}")
print(f"MODEL STATISTICS (using THOP)")
print(f"{'='*60}")
print(f"Total Parameters: {params_formatted}")
print(f"Total FLOPS: {flops_formatted} (for 64x64 input)")
print(f"{'='*60}\n")

batch_size  = 1
ssim        = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr        = PeakSignalNoiseRatio().to(device)

model.eval()

# Create output directory for saving SR images
output_dir = f'results/MDFN_{SCALE_FACTOR}x'
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"EVALUATING MDFN MODEL")
print(f"Scale Factor: {SCALE_FACTOR}x")
print(f"Device: {device}")
print(f"Output Directory: {output_dir}")
print(f"{'='*60}\n")

for i, item in enumerate(data):

    valset     = SuperResolutionDataset(item, transform_img[i], transform_target[i])
    testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
    
    print(f"Evaluating on {name[i]} dataset...")
    
    PSNR_rgb = 0
    SSIM_rgb = 0
    NIQE_rgb = 0
    PSNR_ycbcr = 0
    SSIM_ycbcr = 0
    PSNR_y = 0
    SSIM_y = 0
    NIQE_y = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(testloader, desc=f"Testing {name[i]}", unit="batch")):

            # Images and labels are in RGB space
            images, labels = data[0].to(device), data[1].to(device) 
            outputs = tile_process(images, model, scale=SCALE_FACTOR)

            # Save SR images
            dataset_output_dir = os.path.join(output_dir, name[i])
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Save LR, SR, and HR images
            save_image(images, os.path.join(dataset_output_dir, f'{batch_idx:04d}_LR.png'))
            save_image(outputs, os.path.join(dataset_output_dir, f'{batch_idx:04d}_SR.png'))
            save_image(labels, os.path.join(dataset_output_dir, f'{batch_idx:04d}_HR.png'))

            # Compute metrics in RGB space first
            PSNR_rgb += psnr(outputs, labels) / len(testloader)
            SSIM_rgb += float(ssim(outputs, labels)) / len(testloader)
            
            # NIQE for RGB (convert to grayscale)
            outputs_gray = kornia.color.rgb_to_grayscale(outputs)
            outputs_gray_np = outputs_gray.squeeze(0).squeeze(0).cpu().numpy() * 255.0
            NIQE_rgb += calculate_niqe(outputs_gray_np, crop_border=0) / len(testloader)

            # Convert to YCbCr space for additional metrics
            outputs_ycbcr = kornia.color.rgb_to_ycbcr(outputs)
            labels_ycbcr = kornia.color.rgb_to_ycbcr(labels)

            PSNR_ycbcr += psnr(outputs_ycbcr, labels_ycbcr) / len(testloader)
            SSIM_ycbcr += float(ssim(outputs_ycbcr, labels_ycbcr)) / len(testloader)

            # Extract Y Channel for Y-only metrics
            outputs_ychannel = outputs_ycbcr[:, 0:1, :, :]
            labels_ychannel = labels_ycbcr[:, 0:1, :, :]

            PSNR_y += psnr(outputs_ychannel, labels_ychannel) / len(testloader)
            SSIM_y += float(ssim(outputs_ychannel, labels_ychannel)) / len(testloader)
            
            # NIQE for Y channel
            outputs_y_np = outputs_ychannel.squeeze(0).squeeze(0).cpu().numpy() * 255.0
            NIQE_y += calculate_niqe(outputs_y_np, crop_border=0) / len(testloader)


    print(f"\n{'-'*50}")
    print(f"RESULTS FOR {name[i].upper()} DATASET ({SCALE_FACTOR}x)")
    print(f"{'-'*50}")
    print("RGB Space (NIQE on grayscale):")
    print(f"  PSNR: {PSNR_rgb:.2f} dB")
    print(f"  SSIM: {SSIM_rgb:.4f}")
    print(f"  NIQE: {NIQE_rgb:.4f}")
    print("\nYCbCr Space:")
    print(f"  PSNR: {PSNR_ycbcr:.2f} dB")
    print(f"  SSIM: {SSIM_ycbcr:.4f}")
    print("\nY Channel Only:")
    print(f"  PSNR: {PSNR_y:.2f} dB")
    print(f"  SSIM: {SSIM_y:.4f}")
    print(f"  NIQE: {NIQE_y:.4f}")
    print(f"\nImages saved to: {os.path.join(output_dir, name[i])}")
    print(f"{'-'*50}\n")

print(f"\n{'='*60}")
print(f"EVALUATION COMPLETED FOR MDFN {SCALE_FACTOR}x")
print(f"{'='*60}")
print(f"\nModel Complexity:")
print(f"  Parameters: {params_formatted}")
print(f"  FLOPS: {flops_formatted} (64x64 input)")
print(f"\nAll images saved to: {output_dir}")
print(f"{'='*60}")
