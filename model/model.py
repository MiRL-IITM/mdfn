import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianPyramidFeatureExtractor(nn.Module):
    def __init__(self, feature_channels, num_levels=5, size=(32, 32)):
        super(LaplacianPyramidFeatureExtractor, self).__init__()
        self.num_levels = num_levels
        self.size = size
        # Convolutional layers for feature refinement
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        feature_channels, feature_channels, kernel_size=7, padding=3
                    ),
                    nn.BatchNorm2d(feature_channels),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(self.num_levels)
            ]
        )
        self.extractor = nn.Sequential(
            nn.Conv2d(
                self.num_levels * feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
        )

    def forward(self, x):
        laplacian = []
        self.size = x.shape[2:]
        current = x
        # Build Laplacian pyramid
        for i in range(self.num_levels):
            # Downsample
            downsampled = F.interpolate(
                current, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            # Compute Laplacian
            upsampled = F.interpolate(
                downsampled,
                size=current.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            diff = current - upsampled
            # Refine features
            refined = F.adaptive_max_pool2d(self.conv_layers[i](diff), self.size)
            laplacian.append(refined)
            current = downsampled
        # Concatenate all levels
        return self.extractor(torch.cat(laplacian, dim=1))


class FourierFeatureExtractor(nn.Module):

    def __init__(self, channels,  ratio_g_in=0.75, ratio_g_out=0.75, fft_norm='ortho', image_size = 32):
        super().__init__()
        
        # --- Basic parameters ---
        self.h = image_size
        self.w = image_size
        self.fft_norm = fft_norm
        
        # --- Channel Splitting ---
        self.g_in_channels = int(channels * ratio_g_in)
        self.l_in_channels = channels - self.g_in_channels
        self.g_out_channels = int(channels * ratio_g_out)
        self.l_out_channels = channels - self.g_out_channels
        
        # In FFT, we treat real and imaginary parts as separate channels for Conv2d
        # So we double the channel counts for the global branch convolutions.
        g_in_ch_r_i = self.g_in_channels * 2
        g_out_ch_r_i = self.g_out_channels * 2

        # --- Local Branch ---
        self.local_conv = nn.Sequential(
            nn.Conv2d(self.l_in_channels, self.l_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.l_out_channels),
            nn.GELU()
        )

        # --- Global Branch ---
        # The convolutions now operate on the stacked real+imaginary tensor.
        # Following the repo's logic, we add +1 to input channels for the concatenated APE.
        self.range_transform = nn.Conv2d(g_in_ch_r_i + 1, g_out_ch_r_i, kernel_size=3, padding=1, bias=False)
        self.activation = nn.GELU()
        self.inverse_range_transform = nn.Conv2d(g_out_ch_r_i + 1, g_out_ch_r_i, kernel_size=3, padding=1, bias=False)

        # Absolute Position Embedding (APE)
        freq_w = self.w // 2 + 1
        self.ape = nn.Parameter(torch.randn(1, 1, self.h, freq_w))

        # Dynamic Skip Connection
        self.lambda_skip = nn.Parameter(torch.tensor(0.))
        # 1x1 conv for the skip path to match channels if needed
        if self.g_in_channels != self.g_out_channels:
            self.skip_conv = nn.Conv2d(g_in_ch_r_i, g_out_ch_r_i, kernel_size=1, bias=False)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        batch, _, h, w = x.shape

        x_l, x_g = torch.split(x, [self.l_in_channels, self.g_in_channels], dim=1)

        # --- Local Branch ---
        out_l = self.local_conv(x_l)

        # --- Global Branch ---
        
        # 1. Store min/max for Adaptive Clip
        g_min = torch.amin(x_g, dim=(-2, -1), keepdim=True)
        g_max = torch.amax(x_g, dim=(-2, -1), keepdim=True)

        # 2. Apply 2D Real FFT
        ffted = torch.fft.rfft2(x_g, norm=self.fft_norm)
        
        # 3. Stack real and imaginary parts to create a real-valued tensor
        # This is the key step to avoid the complex-type error with nn.Conv2d.
        ffted_stacked = torch.cat([ffted.real, ffted.imag], dim=1)

        # 4. Main Path (Range Transform -> ReLU -> Inverse Transform)
        # Concatenate APE for the first convolution
        main_in = torch.cat([ffted_stacked, self.ape.expand(batch, -1, -1, -1)], dim=1)
        rt_out = self.range_transform(main_in)
        activated = self.activation(rt_out)
        
        # Concatenate APE for the second convolution
        irt_in = torch.cat([activated, self.ape.expand(batch, -1, -1, -1)], dim=1)
        main_out = self.inverse_range_transform(irt_in)
        
        # 5. Skip Path
        skip_out = self.skip_conv(ffted_stacked)

        # 6. Dynamic Skip Connection
        lambda_val = torch.sigmoid(self.lambda_skip)
        ffted_combined = main_out * (1 - lambda_val) + skip_out * lambda_val

        # 7. Unstack the real-valued tensor back into real and imaginary parts
        ffted_real, ffted_imag = torch.chunk(ffted_combined, 2, dim=1)
        output_ffted = torch.complex(ffted_real, ffted_imag)

        # 8. Apply Inverse 2D Real FFT
        out_g_spatial = torch.fft.irfft2(output_ffted, s=(h, w), norm=self.fft_norm)

        # 9. Concatenate local and global outputs
        output = torch.cat((out_l, out_g_spatial), dim=1)

        return output 


class MDFN(nn.Module):
    def __init__(
        self,
        scale_factor=2,
        channels=3,
        base_channels=24,
        laplacian_levels=3,
        image_size=32):
        
        super(MDFN, self).__init__()

        self.scale_factor = scale_factor
        self.base_channels = base_channels
        self.laplacian_levels = laplacian_levels
        self.image_size = image_size

        self.spatial_block = nn.Sequential(
            nn.Conv2d(channels, base_channels // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels // 4, base_channels//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, padding=1),
        )

        self.laplacian_block = LaplacianPyramidFeatureExtractor(
            base_channels, num_levels=self.laplacian_levels
        )

        self.fourier_block = nn.ModuleList(
            [
                FourierFeatureExtractor(channels=base_channels, image_size=image_size)
                for _ in range(self.laplacian_levels * 2)
            ]
        )

        shuffle_channels = 3*base_channels//(self.scale_factor**2)

        self.refinement = nn.Sequential(
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(shuffle_channels, base_channels//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels//4, channels, kernel_size=3, padding=1)
        )
        
        self.final_activation = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        self._initialize_weights(self)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        image = self.upsample(x)
        x = self.spatial_block(x)
        laplacian_features = self.laplacian_block(x)
        fourier_features = laplacian_features
        for block in self.fourier_block:
            fourier_features = block(fourier_features) + fourier_features
        features = torch.cat([x, laplacian_features, fourier_features], dim=1)
        out = self.refinement(features) + image
        return self.final_activation(out)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDFN(channels=3, base_channels=64, laplacian_levels=5, image_size=64).to(device)
    x = torch.randn(1, 3, 64, 64, device=device)
    y = model(x)
    print(y.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"Number of parameters: {total_params:,}")
    print(f"Number of parameters in million: {total_params/1e6}")
