import torch 
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_disc_block(im_dim, hidden_dim*4),
            self.get_disc_block(hidden_dim*4, hidden_dim*2),
            self.get_disc_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
    def get_disc_block(self, in_features, out_features):
        return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LeakyReLU(0.2)
            )
    def forward(self, img):
        return self.disc(img)


class Generator(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128, im_dim=784):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.get_gen_block(z_dim, hidden_dim),
            self.get_gen_block(hidden_dim, hidden_dim*2),
            self.get_gen_block(hidden_dim*2, hidden_dim*4),
            self.get_gen_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid()
        )

    def get_gen_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
    def forward(self, noise):
        return self.gen(noise)
