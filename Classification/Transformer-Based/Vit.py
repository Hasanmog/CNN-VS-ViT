import torch.nn as nn
import torch
from torch.nn.init import xavier_uniform_

class Patch_Embed(nn.Module):
    """Patch embed an image to p x p patches
    
    Arguments:
    -------------
        p (int): patch size
        proj_dim(int): dimension to which the patches will be projected
        max_patches(int): maximum number of patches an image can have     
    Returns:
    ----------
        A flattened patch-embedded tensor with positional encoding.
    """
    
    def __init__(self , p , proj_dim , max_patches):
        super(Patch_Embed , self).__init__()
        
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
        
        x = x + self.positional_embed[: ,  : num_patches ,  :].to(x.device)  # to only work with the actual number of patches and not the maximum number
        return x    # [batch_size, num_patches, proj_dim]
    
 
class MultiHeadAttention(nn.Module):
    """ Attention mechanism block built from scratch
    
    Arguments:
    -------------
        input_dim(int): Dimension of the input embeddings (typically the patch embedding dimension).
        out_dim(int): Dimension of the query, key, and value projections.
        num_heads(int): number of attention heads ~ must be divisible by the out_dim of the patch embedding
   Returns:
    --------
        attention_output: Tensor of shape [batch_size, num_patches, out_dim]
        The output of the multi-head attention mechanism.
    """
    def __init__(self , input_dim , out_dim , num_heads):
        super(MultiHeadAttention , self).__init__()
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.heads_dim = out_dim // num_heads    # each head will have out_dim // num_heads dimension
        self.w_q = nn.Parameter(torch.empty(input_dim , out_dim))
        self.w_k = nn.Parameter(torch.empty(input_dim , out_dim))
        self.w_v = nn.Parameter(torch.empty(input_dim , out_dim))
        self.num_heads = num_heads 
        self.final_linear = nn.Linear(out_dim, out_dim)
        
        xavier_uniform_(self.w_q)
        xavier_uniform_(self.w_k)
        xavier_uniform_(self.w_v)
        
        self.dk = torch.tensor(self.heads_dim, dtype=torch.float32) # dimensionality of the key , which is the proj_dim of the Patch_Embed , and the heads_dim in multihead attention
        
    def forward(self , x):
        
        batch_size , num_patches , _ = x.shape
        
        Q = x @ self.w_q# matrix multiplication
        V = x @ self.w_v
        K = x @ self.w_k
        
        Q = Q.view(batch_size , num_patches , self.num_heads , self.heads_dim).transpose(1 , 2) # transpose to work with heads in parallel 
        K = K.view(batch_size , num_patches , self.num_heads , self.heads_dim).transpose(1, 2 )
        V = V.view(batch_size , num_patches , self.num_heads , self.heads_dim).transpose(1 , 2) 
        
        
        scores = Q @ K.transpose(-2, -1) / torch.sqrt(self.dk)
        attention_weights = torch.softmax(scores, dim=-1)
        
        output = attention_weights @ V  # [batch_size, num_heads, seq_length, head_dim]
         
        attention_output = output.transpose(1, 2).contiguous() # [batch_size, seq_length , num_heads , head_dim]
        attention_output = attention_output.view(batch_size , num_patches , -1) # [batch_size , num_patches , num_heads * head_dim] == [batch_size , num_patches , out_dim]
        
        attention_output = self.final_linear(attention_output)     
            
        return attention_output  
        
        
class Transformer_Encoder(nn.Module):
    """
    Multi-layer Transformer Encoder with multiple attention heads and MLPs.
    
    Arguments:
    -------------
        embed_dim (int): Dimension of the input embeddings (typically the patch embedding dimension).
        num_heads (int): Number of attention heads. Must be divisible by the embed_dim.
        num_layers (int): Number of transformer encoder layers.
        mlp_dim (int): Dimension of the feed-forward MLP inside each encoder layer.
    
    Returns:
    ------------
        Tensor of shape [batch_size, num_patches + 1, embed_dim]:
        The output after passing the input through multiple transformer encoder layers. The output includes both the class token and patch embeddings.
        
    """
    def __init__(self , embed_dim , num_heads , num_layers , mlp_dim ):
        super(Transformer_Encoder , self).__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(embed_dim, embed_dim, num_heads),
                'layernorm1': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, mlp_dim),
                    nn.ReLU(),
                ),
                'layernorm2': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])
    def forward(self, x):
        
        for layer in self.layers:
            x_norm = layer['layernorm1'](x)
            attention_output = layer['attention'](x_norm)
            x = attention_output + x  # Residual connection

            
            x_norm = layer['layernorm2'](x)
            mlp_output = layer['mlp'](x_norm)
            x = mlp_output + x  # Residual connection
        
        return x
        
        
        
        
class VisionTransformer(nn.Module):
    """
    Full Vision Transformer model with patch embedding, multi-layer transformer encoder, class token, and a final classification head.
    
    Arguments:
    -------------
        patch_size (int): Size of the patches to divide the image into (e.g., 16x16).
        embed_dim (int): Dimension to which the patches will be projected (typically 768 for ViT).
        max_patches (int): Maximum number of patches that an image can have.
        num_heads (int): Number of attention heads. Must be divisible by the embed_dim.
        num_layers (int): Number of transformer encoder layers.
        mlp_dim (int): Dimension of the feed-forward MLP inside each transformer encoder layer.
        num_classes (int): Number of output classes for the classification task.
    
    Returns:
    ------------
        Tensor of shape [batch_size, num_classes]: The final classification logits for each class.
   
    """
    def __init__(self,
                 patch_size , 
                 embed_dim ,
                 max_patches, 
                 num_heads , 
                 num_layers , 
                 mlp_dim , 
                 num_classes):
        super(VisionTransformer , self).__init__()
        self.patch_embed = Patch_Embed(patch_size , embed_dim , max_patches)
        self.encoder = Transformer_Encoder(embed_dim , num_heads , num_layers , mlp_dim)
        
        self.class_token = nn.Parameter(torch.zeros(1 , 1 , embed_dim))
        
        self.prediction_head = nn.Linear(embed_dim , num_classes)
        
    def forward(self , x):
        
        b , c ,h , w = x.shape
        
        patches = self.patch_embed(x)
        
        class_token_expanded = self.class_token.expand(b, -1, -1) # to match batch size
        x = torch.cat((class_token_expanded, patches), dim=1)
        
        attention = self.encoder(x)
        
        class_token_final = attention[:, 0]
        logits = self.prediction_head(class_token_final)
        
        return logits
        
        
        
        
        