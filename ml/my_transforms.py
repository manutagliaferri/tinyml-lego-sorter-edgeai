import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        # 1. Create the mask
        mask = tensor > 0
        
        # 2. Optimization: Generate noise ONLY for the non-zero pixels
        # mask.sum() gives us the exact count of 'Lego' pixels
        noise = torch.randn(mask.sum()) * self.std + self.mean
        
        # 3. Apply noise using boolean indexing
        # This selects the pixels, adds noise, and puts them back in place
        tensor[mask] += noise
        
        # 4. Clamp to ensure we remain valid images
        return torch.clamp(tensor, 0., 1.)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class AddBlockGaussianNoise(object):
    def __init__(self, mean=0., std=0.1, block_size=4):
        self.mean = mean
        self.std = std
        self.block_size = block_size
    
    def __call__(self, tensor):
        # tensor shape: [C, H, W] where C=1 for grayscale
        C, H, W = tensor.shape
        
        # 1. Create the mask for non-zero (Lego) pixels
        mask = tensor > 0
        
        # 2. Calculate dimensions for noise blocks
        h_blocks = (H + self.block_size - 1) // self.block_size
        w_blocks = (W + self.block_size - 1) // self.block_size
        
        # 3. Generate noise at block resolution (one value per 4x4 block)
        noise_blocks = torch.randn(C, h_blocks, w_blocks) * self.std + self.mean
        
        # 4. Upsample noise blocks to match image size using nearest neighbor
        # This replicates each noise value across its 4x4 block
        noise_full = F.interpolate(
            noise_blocks.unsqueeze(0),  # Add batch dimension
            size=(H, W),
            mode='nearest'
        ).squeeze(0)  # Remove batch dimension
        
        # 5. Apply noise only to non-zero pixels (Lego regions)
        tensor[mask] += noise_full[mask]
        
        # 6. Clamp to valid range
        return torch.clamp(tensor, 0., 1.)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(mean={self.mean}, std={self.std}, block_size={self.block_size})')

# Custom wrapper to apply transforms
class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)