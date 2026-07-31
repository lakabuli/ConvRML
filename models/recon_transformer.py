import torch
from torch import nn
import math

class Recon_Transformer(nn.Module):
    def __init__(self, height, width, patch_size, n_channels, n_heads=4, n_blocks=6, embed_dim=128, ffn_multiplier=2, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.positional_encoding = Patch_Embedding(height, width, patch_size, n_channels, embed_dim, dropout_rate=0.0)                                    
        self.encoder = Encoder(embed_dim, ffn_multiplier, n_heads, n_blocks, dropout_rate)
        self.decoder = Decoder(embed_dim, n_channels, height, width, patch_size)
        self.apply(init_weights)
    
    def forward(self, x):                                                       
        encoder_in = self.positional_encoding(x)         # (batch_size, seq_len, embed_dim) tensor
        encoder_out = self.encoder(encoder_in)        # output = (batch_size, num_patches, embed_dim)
        result = self.decoder(encoder_out)
        return result
    
# Initialize weights to very small numbers close to 0, instead of pytorch's default initalization. 
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0) 

    elif isinstance(m, Patch_Embedding):
        torch.nn.init.trunc_normal_(m.embedding, mean=0.0, std=0.02)
    
# list of all encoder blocks
class Encoder(nn.Module):
    def __init__(self, embed_dim, ffn_multiplier, n_heads, n_blocks, dropout_rate):
        super().__init__()

        self.encoder_layers = nn.ModuleList([Encoder_Block(embed_dim, ffn_multiplier, n_heads, dropout_rate) for _ in range(n_blocks)])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)     # call on individual encoder block one at a time.
        return x
    
# list of all decoder blocks
class Encoder_Block(nn.Module):
    def __init__(self, embed_dim, ffn_multiplier, n_heads, dropout_rate):
        super().__init__()

        self.attention = Multi_Headed_Attention(n_heads, embed_dim)    
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)     

        self.ffn = Position_wise_ffn(embed_dim, ffn_multiplier)        # transformer paper has multiplier = 4, simplified model has 2
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))        # now most people do residual after dropout + attention
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x
    
# Doing basic CNN upsampling
class Decoder(nn.Module):
    def __init__(self, embed_dim, end_num_channels, height, width, patch_size):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2, padding=1)     # same as lensless imaging transformer
        self.conv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2, padding=1)        # had bias=False in their decoder
        self.conv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(embed_dim // 8, end_num_channels, kernel_size=2, stride=2, padding=1)

        self.batch1 = nn.BatchNorm2d(embed_dim // 2)
        self.batch2 = nn.BatchNorm2d(embed_dim // 4)
        self.batch3 = nn.BatchNorm2d(embed_dim // 8)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.height, self.width = height, width
        self.num_patch_height = height // patch_size[0]
        self.num_patch_width = width // patch_size[1]

    def forward(self, x):
        x = x.transpose(1, 2)       # did this in patch embedding for memory efficiency (lensless transformer doesn't do this in the first place)
        x = torch.unflatten(x, 2, (self.num_patch_height, self.num_patch_width))       # undo the flatten operation in embedding layer (flattens patched image to one long vector)
        x = self.relu1(self.batch1(self.conv1(x)))
        x = self.relu2(self.batch2(self.conv2(x)))
        x = self.relu3(self.batch3(self.conv3(x)))
        x = self.conv4(x)
        x = nn.functional.interpolate(x, (self.height, self.width), mode='bilinear', align_corners=False)      # Get image to desired size
        return x



### Helper classes ###

class Multi_Headed_Attention(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads

        # If doesn't divide evenly, concatenation won't work.
        assert embed_dim % n_heads == 0
        self.d_key = embed_dim // n_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)     
        self.Wk = nn.Linear(embed_dim, embed_dim)     
        self.Wv = nn.Linear(embed_dim, embed_dim)

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.Wq(x)      # dimensions = (batch_size, seq_len, embed_dim) for Q, K, V
        k = self.Wk(x)
        v = self.Wv(x)

        # 1 - split into heads
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_key)     # (batch size, sequence len, num heads, d_key)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_key)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_key)

        # 2 - swap n_heads and sequence length dimensions to split each batch along each head
        q = q.transpose(1, 2)             # (batch size, n_heads, sequence len, d_key)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3 - Do self-attention                                                                           
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))       # want last two dimensions to get swapped
        probabilities = torch.softmax(attention_weights, dim=-1)         # gets the probabilities along last dimension. For 2d the result of softmax is a (d_v, 1) vector.
        output = torch.matmul(probabilities, v) 

        # 4 - Reverse step 2 then concatenate heads
        output = output.transpose(1, 2).contiguous()            # Need contiguous because .view() changes the way the tensor is stored (not stored consecutively in memory anymore)
        output = output.view(output.size(0), -1, self.d_key * self.n_heads)
        output = self.projection(output)
        return output  


# separate image into patches and do positional embedding + patch embedding for ViT
class Patch_Embedding(nn.Module):
    def __init__(self, height, width, patch_size, n_channels, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.patch_height, self.patch_width = patch_size
        assert height % self.patch_height == 0
        assert width % self.patch_width == 0

        self.embed_dim = embed_dim          # same as embedding dimension 
        self.num_patches = (height // self.patch_height) * (width // self.patch_width)
        
        # Note: positional embedding in ViT does not use sine/cosine
        self.conv = torch.nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)         # no overlapping means stride needs to be same as patch size
        self.embedding = nn.Parameter(torch.zeros([1, self.num_patches, self.embed_dim], requires_grad=True))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)        # (batch size, channels, height, width) => (batch size, embed_dim, height/patch size, width/patch size)
        x = torch.flatten(x, 2)         # result = (batch size, embed_dim, num_patches). 
        x = x.transpose(1, 2)           # switch to having num_patch tensors of dim embed_dim (for efficient memory storage since embedding dimension is usually smaller than sequence length/number of patches)

        x = x + self.embedding
        x = self.dropout(x)
        return x

# called "position wise" because it already includes positional embeddings from previous layers
class Position_wise_ffn(nn.Module):                           # 2 fully connected dense layers  https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c 
    def __init__(self, embed_dim, ffn_multiplier):                       # feed forward just means no recurrent relations
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * ffn_multiplier)
        self.linear2 = nn.Linear(embed_dim * ffn_multiplier, embed_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)         
        x = self.gelu(x)
        x = self.linear2(x)
        return x    