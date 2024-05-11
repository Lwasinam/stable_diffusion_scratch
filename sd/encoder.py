import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch Size, Channel, Height, Width, ) -> (Batch Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Heigth , Widht) doesnt change
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Heigth , Widht) doesnt change
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Heigth , Widht) -> (Batch Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), 

            # (Batch_Size, 128, Heigth/2 , Width /2) -> (Batch Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Heigth/2 , Width /2) -> (Batch Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Heigth , Widht) -> (Batch Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Heigth/4 , Width /4) -> (Batch Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Heigth/4 , Width /4) -> (Batch Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Heigth/4 , Widht/4) -> (Batch Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

           
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),
             
            # (Batch_Size, 512, Heigth/8 , Width /8) -> (Batch Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # size remains saome 
            VAE_AttentionBlock(512),

             # (Batch_Size, 512, Heigth/8 , Width /8) -> (Batch Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Heigth/8, Width /8) -> (Batch Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            nn.SiLU(), 

            # (Batch_Size, 512, Heigth/8, Width /8) -> (Batch Size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

             # (Batch_Size, 8, Heigth/8, Width /8) -> (Batch Size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)



        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel ,Height, Widht)
        # noise (Batch_size, Channel, Out_channels, Height/8, Widht/8)
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (Padding_Left, Padding_right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0,1,0,1))
            x = module(x)
        # (Batch_size, 8 ,Height, Width/8, Height/8) -> two tensors (Batch_size, 4, Heigth/8, Widht/8)
        mean, log_variance = torch.chunk(x,2, dim=1)   
        log_variance = torch.clamp(log_variance, -30, 20)  

        variance = log_variance.exp()    
        stdev = variance.sqrt()

        # z =  N(0,1 ) -> N(mean, variance)
        # x = mean +stdev * z
        #trasnform x into a distribution of noise??
        x = mean + stdev * noise
        x *= 0.18215

        return x

