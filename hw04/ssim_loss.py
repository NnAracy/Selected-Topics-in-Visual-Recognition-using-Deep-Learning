import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=1.0, channel=3, size_average=True, K1=0.01, K2=0.03):
        """
        SSIM Loss.
        Args:
            window_size (int): Size of the Gaussian window.
            data_range (float or int): The dynamic range of the input images (e.g., 1.0 for images in [0,1] or 255 for images in [0,255]).
            channel (int): Number of channels in the input images.
            size_average (bool): If True, the loss is averaged over all elements and batch.
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.size_average = size_average
        self.K1 = K1
        self.K2 = K2
        
        # Create a Gaussian window and register it as a buffer
        self.register_buffer('window', create_window(window_size, channel))

    def ssim(self, img1, img2):
        # mu1, mu2 dimensions: [Batch, Channel, Height, Width]
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # Ensure non-negative variances for stability
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        C1 = (self.K1 * self.data_range)**2
        C2 = (self.K2 * self.data_range)**2
        
        term1_num = 2 * mu1_mu2 + C1
        term2_num = 2 * sigma12 + C2
        
        term1_den = mu1_sq + mu2_sq + C1
        term2_den = sigma1_sq + sigma2_sq + C2
        
        ssim_map = (term1_num * term2_num) / (term1_den * term2_den)  # Per pixel SSIM

        if self.size_average:
            return ssim_map.mean()  # Average over all
        else:
            return ssim_map.mean(dim=[1, 2, 3])  # Average over channel

    def forward(self, img1, img2):
        ssim_val = self.ssim(img1, img2)
        loss = 1.0 - ssim_val  # SSIM loss

        return loss