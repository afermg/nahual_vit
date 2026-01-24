"""From https://huggingface.co/CaicedoLab/MorphEm"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as v2
from transformers import AutoModel


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


# Load model
device = "cuda"
model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
model.to(device).eval()

# Define transforms
transform = v2.Compose([
    SaturationNoiseInjector(),
    PerImageNormalize(),
    v2.Resize(size=(224, 224), antialias=True),
])

# Generate random batch (N, C, H, W)
batch_size = 2
num_channels = 3
images = torch.randint(
    0, 256, (batch_size, num_channels, 512, 512), dtype=torch.float32
)

print(f"Input shape: {images.shape} (N={batch_size}, C={num_channels}, H=512, W=512)")
print()

# Bag of Channels (BoC) - process each channel independently
with torch.no_grad():
    batch_feat = []
    images = images.to(device)

    for c in range(images.shape[1]):
        # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
        single_channel = images[:, c, :, :].unsqueeze(1)

        # Apply transforms
        single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

        # Extract features
        output = model.forward_features(single_channel)
        feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
        batch_feat.append(feat_temp)

# Concatenate features from all channels
features = np.concatenate(batch_feat, axis=1)

print(f"Output shape: {features.shape}")
print(f"  - Batch size (N): {features.shape[0]}")
print(f"  - Feature dimension (C * feature_dim): {features.shape[1]}")

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as v2
from transformers import AutoModel


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


# Load model
device = "cuda"
model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
model.to(device).eval()

# Define transforms
transform = v2.Compose([
    SaturationNoiseInjector(),
    PerImageNormalize(),
    v2.Resize(size=(224, 224), antialias=True),
])

# Generate random batch (N, C, H, W)
batch_size = 2
num_channels = 3
images = torch.randint(
    0, 256, (batch_size, num_channels, 512, 512), dtype=torch.float32
)

print(f"Input shape: {images.shape} (N={batch_size}, C={num_channels}, H=512, W=512)")
print()

# Bag of Channels (BoC) - process each channel independently
with torch.no_grad():
    batch_feat = []
    images = images.to(device)

    for c in range(images.shape[1]):
        # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
        single_channel = images[:, c, :, :].unsqueeze(1)

        # Apply transforms
        single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

        # Extract features
        output = model.forward_features(single_channel)
        feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
        batch_feat.append(feat_temp)

# Concatenate features from all channels
features = np.concatenate(batch_feat, axis=1)

print(f"Output shape: {features.shape}")
print(f"  - Batch size (N): {features.shape[0]}")
print(f"  - Feature dimension (C * feature_dim): {features.shape[1]}")

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as v2
from transformers import AutoModel


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


# Load model
device = "cuda"
model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
model.to(device).eval()

# Define transforms
transform = v2.Compose([
    SaturationNoiseInjector(),
    PerImageNormalize(),
    v2.Resize(size=(224, 224), antialias=True),
])

# Generate random batch (N, C, H, W)
batch_size = 2
num_channels = 3
images = torch.randint(
    0, 256, (batch_size, num_channels, 512, 512), dtype=torch.float32
)

print(f"Input shape: {images.shape} (N={batch_size}, C={num_channels}, H=512, W=512)")
print()

# Bag of Channels (BoC) - process each channel independently
with torch.no_grad():
    batch_feat = []
    images = images.to(device)

    for c in range(images.shape[1]):
        # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
        single_channel = images[:, c, :, :].unsqueeze(1)

        # Apply transforms
        single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

        # Extract features
        output = model.forward_features(single_channel)
        feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
        batch_feat.append(feat_temp)

# Concatenate features from all channels
features = np.concatenate(batch_feat, axis=1)

print(f"Output shape: {features.shape}")
print(f"  - Batch size (N): {features.shape[0]}")
print(f"  - Feature dimension (C * feature_dim): {features.shape[1]}")

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as v2
from transformers import AutoModel


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


# Load model
device = "cuda"
model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
model.to(device).eval()

# Define transforms
transform = v2.Compose([
    SaturationNoiseInjector(),
    PerImageNormalize(),
    v2.Resize(size=(224, 224), antialias=True),
])

# Generate random batch (N, C, H, W)
batch_size = 2
num_channels = 3
images = torch.randint(
    0, 256, (batch_size, num_channels, 512, 512), dtype=torch.float32
)

print(f"Input shape: {images.shape} (N={batch_size}, C={num_channels}, H=512, W=512)")
print()

# Bag of Channels (BoC) - process each channel independently
with torch.no_grad():
    batch_feat = []
    images = images.to(device)

    for c in range(images.shape[1]):
        # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
        single_channel = images[:, c, :, :].unsqueeze(1)

        # Apply transforms
        single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

        # Extract features
        output = model.forward_features(single_channel)
        feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
        batch_feat.append(feat_temp)

# Concatenate features from all channels
features = np.concatenate(batch_feat, axis=1)

print(f"Output shape: {features.shape}")
print(f"  - Batch size (N): {features.shape[0]}")
print(f"  - Feature dimension (C * feature_dim): {features.shape[1]}")

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as v2
from transformers import AutoModel


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


# Load model
device = "cuda"
model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
model.to(device).eval()

# Define transforms
transform = v2.Compose([
    SaturationNoiseInjector(),
    PerImageNormalize(),
    v2.Resize(size=(224, 224), antialias=True),
])

# Generate random batch (N, C, H, W)
batch_size = 2
num_channels = 3
images = torch.randint(
    0, 256, (batch_size, num_channels, 512, 512), dtype=torch.float32
)

print(f"Input shape: {images.shape} (N={batch_size}, C={num_channels}, H=512, W=512)")
print()

# Bag of Channels (BoC) - process each channel independently
with torch.no_grad():
    batch_feat = []
    images = images.to(device)

    for c in range(images.shape[1]):
        # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
        single_channel = images[:, c, :, :].unsqueeze(1)

        # Apply transforms
        single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

        # Extract features
        output = model.forward_features(single_channel)
        feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
        batch_feat.append(feat_temp)

# Concatenate features from all channels
features = np.concatenate(batch_feat, axis=1)

print(f"Output shape: {features.shape}")
print(f"  - Batch size (N): {features.shape[0]}")
print(f"  - Feature dimension (C * feature_dim): {features.shape[1]}")
