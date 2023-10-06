import torch
from torch import nn
from functools import reduce
from operator import mul


class AutoEncoder(nn.Module):
    '''
    For MNISTs images
    input_size = [1, 28, 28]
    '''
    def __init__(self, input_size, hidden_size, code_length):
        super().__init__()

        c, h, w = input_size

        # Encoding layers
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=hidden_size//2, stride=2, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_size//2, out_channels=hidden_size, stride=2, kernel_size=3, bias=False, padding=1)

        self.deepest_shape = (hidden_size, h // 4, w // 4)
        self.flatten_dim = reduce(mul, self.deepest_shape)
        self.act = nn.ReLU()

        self.linear1 = nn.Linear(in_features=self.flatten_dim, out_features=code_length)

        # Decoding layers
        self.linear2 = nn.Linear(in_features=code_length, out_features=self.flatten_dim)
        self.convt1 = nn.ConvTranspose2d(in_channels=hidden_size, out_channels=hidden_size//2, stride=2, kernel_size=3, padding=1,
                                         output_padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=hidden_size//2, out_channels=c, stride=2, kernel_size=3, padding=1,
                                         output_padding=1)

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)

        return _x, emb

    def decoder(self, emb):
        _x = self.act(self.linear2(emb))
        _x = _x.view(-1, *self.deepest_shape)
        _x = self.act(self.convt1(_x))
        _x = self.convt2(_x)
        return _x

    def encoder(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(torch.flatten(h, 1))
        emb = self.linear1(h)
        return emb

class AutoEncoder_RGB32(nn.Module):
    """
    For RGB image sets
    input_size = [3, 32, 32]
    """

    def __init__(self, input_size, hidden_size, code_length):
        super().__init__()

        c, h, w = input_size

        hidden_channels = [hidden_size // (2 ** i) for i in range(3)]
        hidden_channels.reverse()

        # Encoding layers
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=hidden_channels[0], stride=2, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], stride=2, kernel_size=3,
                               bias=False, padding=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2], stride=2, kernel_size=3,
                               bias=False, padding=1)
        self.deepest_shape = (hidden_channels[-1], h // (2 ** len(hidden_channels)), w // (2 ** len(hidden_channels)))
        self.flatten_dim = reduce(mul, self.deepest_shape)
        self.act = nn.LeakyReLU()

        self.linear1 = nn.Linear(in_features=self.flatten_dim, out_features=code_length)

        # Decoding layers
        self.linear2 = nn.Linear(in_features=code_length, out_features=self.flatten_dim)
        self.convt1 = nn.ConvTranspose2d(in_channels=hidden_channels[2], out_channels=hidden_channels[1], stride=2,
                                         kernel_size=3, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=hidden_channels[1], out_channels=hidden_channels[0], stride=2,
                                         kernel_size=3, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=hidden_channels[0], out_channels=c, stride=2, kernel_size=3, padding=1,
                                         output_padding=1)

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)
        return _x, emb

    def decoder(self, emb):
        _x = self.act(self.linear2(emb))
        _x = _x.view(-1, *self.deepest_shape)
        _x = self.act(self.convt1(_x))
        _x = self.act(self.convt2(_x))
        _x = self.convt3(_x)
        return _x

    def encoder(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.act(torch.flatten(h, 1))
        emb = self.linear1(h)
        return emb

class AutoEncoder_RGB256(nn.Module):
    """
    For RGB images
    input_size = [3, 256, 256]
    """

    def __init__(self, input_size, hidden_size, code_length):
        super().__init__()

        c, h, w = input_size

        hidden_channels = [hidden_size//(2**i) for i in range(6)]
        hidden_channels.reverse()

        # Encoding layers
        in_channels = c
        modules = []
        for h_dim in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encode = nn.Sequential(*modules)
        self.act = nn.LeakyReLU()

        self.deepest_shape = (hidden_channels[-1], h // (2 ** len(hidden_channels)), w // (2 ** len(hidden_channels)))
        self.flatten_dim = reduce(mul, self.deepest_shape)

        self.linear1 = nn.Linear(in_features=self.flatten_dim, out_features=code_length)

        # Decoding layers
        self.linear2 = nn.Linear(in_features=code_length, out_features=self.flatten_dim)
        modules = []
        hidden_channels.reverse()
        for i in range(len(hidden_channels) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i],
                                       hidden_channels[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )
        self.decode = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[-1],
                               c,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            )

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)

        return _x, emb

    def decoder(self, emb):
        _x = self.act(self.linear2(emb))
        _x = _x.view(-1, *self.deepest_shape)
        _x = self.decode(_x)
        _x = self.final_layer(_x)
        return _x

    def encoder(self, x):
        h = self.encode(x)
        h = self.act(torch.flatten(h, 1))
        emb = self.linear1(h)
        return emb