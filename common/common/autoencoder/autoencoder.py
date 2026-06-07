import torch
import torch.nn as nn
from typing import List, Tuple

class Encoder(nn.Module):
    def __init__(self, conv_channel_sizes: List[int], lin_channel_sizes: List[int], input_shape: Tuple[int, int] = (28, 28)):
        super().__init__()
        conv_layers = []
        for i in range(1, len(conv_channel_sizes)):
            conv_layers.append(nn.Conv2d(
                in_channels=conv_channel_sizes[i-1],
                out_channels=conv_channel_sizes[i],
                kernel_size=3,
                stride=2,
                padding=1
            ))
            conv_layers.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*conv_layers)

        # Dynamically calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, conv_channel_sizes[0], *input_shape)
            dummy_output = self.conv(dummy_input)
            self.spatial_shape = dummy_output.shape[2:] # (H_out, W_out)
            self.flat_features = dummy_output.numel() # Total flattened features

        # Build linear layers
        lin_layers = []
        in_features = self.flat_features
        
        for i, out_features in enumerate(lin_channel_sizes):
            lin_layers.append(nn.Linear(in_features, out_features))
            if i < len(lin_channel_sizes) - 1:
                lin_layers.append(nn.LeakyReLU(0.2))
            in_features = out_features
            
        self.fc = nn.Sequential(*lin_layers)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, conv_channel_sizes: List[int], lin_channel_sizes: List[int], spatial_shape: Tuple[int, int], flat_features: int):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.unflat_channels = conv_channel_sizes[-1]

        rev_linear_channel_sizes = list(reversed(lin_channel_sizes))
        lin_layers = []

        in_features = rev_linear_channel_sizes[0]
        
        for out_features in rev_linear_channel_sizes[1:]:
            lin_layers.append(nn.Linear(in_features, out_features))
            lin_layers.append(nn.LeakyReLU(0.2))
            in_features = out_features

        # Final linear layer to project back to the total flattened spatial size
        lin_layers.append(nn.Linear(in_features, flat_features))
        lin_layers.append(nn.LeakyReLU(0.2))
        
        self.fc = nn.Sequential(*lin_layers)

        # Build conv layers
        rev_channels = list(reversed(conv_channel_sizes))
        conv_layers = []
        for i in range(1, len(rev_channels)):
            conv_layers.append(nn.ConvTranspose2d(
                in_channels=rev_channels[i-1],
                out_channels=rev_channels[i],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ))
            if i < len(rev_channels) - 1:
                conv_layers.append(nn.LeakyReLU(0.2))
        conv_layers.append(nn.Sigmoid())

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.unflat_channels, *self.spatial_shape)
        return self.conv(x)


class Autoencoder(nn.Module):
    def __init__(self, conv_channel_sizes: List[int], lin_channel_sizes: List[int], input_shape: Tuple[int, int] = (28, 28)):
        super().__init__()
        self.encoder = Encoder(conv_channel_sizes, lin_channel_sizes, input_shape)
        self.decoder = Decoder(conv_channel_sizes, lin_channel_sizes, self.encoder.spatial_shape, self.encoder.flat_features)

    def forward(self, x): 
        return self.decoder(self.encoder(x))