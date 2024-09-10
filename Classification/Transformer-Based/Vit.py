import torch.nn as nn
import torch


class Patch_Embed(nn.Module):
    """Patch embed an image to p x p patches
    
    Args:
    -----
        p (int) -   patch size
        proj_dim(int) - dimension to which the patches will be projected
        max_patches - maximum number of patches an image can have
        
    Returns:
    ----------
        A flattened patch-embedded tensor with positional encoding.
    """
    
    def __init__(self , p , proj_dim , max_patches):
        super(Patch_Embed).__init__()
        
        self.patch_size = p
        self.proj_dim = proj_dim
        self.linear = nn.Linear(p*p*3 , proj_dim)
        self.positional_embed = nn.Parameter(torch.randn(1 , max_patches , proj_dim)) # can be initalized with zeros instead
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        
    def forward(self , x ):
        b , c , h , w = x.shape
        
        assert h % self.patch_size == 0 and w % self.patch_size == 0, "Image size must be divisible by the patch size."

        num_patches = h*w // self.patch_size**2
        
        x = x.reshape(b , num_patches , self.patch_size*self.patch_size*c)
        x = self.linear(x)
        x = x.to(self.device)
        
        x = x + self.positional_embed[: ,  : num_patches ,  :].to(x.device)
            
        return x
 
        
        
        
        
        