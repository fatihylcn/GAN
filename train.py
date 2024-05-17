import torch 
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator



z_dim = 64 
n_epoch = 20
batch_size = 64
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
img_dim = 1*28*28

criterion = nn.BCEWithLogitsLoss()

disc = Discriminator().to(device)
gen = Generator(z_dim=z_dim).to(device)

disc_optim = torch.optim.Adam(disc.parameters(), lr=lr)
gen_optim = torch.optim.Adam(gen.parameters(), lr=lr)
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataloader = DataLoader(
    MNIST(root=".", download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)


writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0
for epoch in range(n_epoch):
    for batch_idx, (real, label) in enumerate(dataloader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)
        
        noise = torch.randn((cur_batch_size, z_dim), device=device)

        ###### Train Discriminator
        fake = gen(noise)
        fake_pred = disc(fake.detach())
        fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
        real_pred = disc(real)
        real_loss = criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2 
        disc_optim.zero_grad()
        disc_loss.backward(retain_graph=True)
        disc_optim.step()


        ###### Train Generator
        noise2 = torch.randn((cur_batch_size, z_dim), device=device)
        fake2 = gen(noise2)
        fake2_pred = disc(fake2)
        gen_loss = criterion(fake2_pred, torch.ones_like(fake2_pred))
        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        if batch_idx == 0:

            print(
                f"Epoch {epoch} / {n_epoch}\n"
                f"Loss Discriminator {disc_loss:.4f}, Loss Generator {gen_loss:.4f}"
            )

            with torch.no_grad():
                noise_test = torch.randn((cur_batch_size, z_dim), device=device)
                fake = gen(noise_test).reshape(-1, 1, 28, 28)
                real = real.reshape(-1, 1, 28, 28)
                grid_fake = make_grid(fake, normalize=True)
                grid_real = make_grid(real, normalize=True)

                writer_fake.add_image(
                    "Fake Images", grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real Images", grid_real, global_step=step
                )

                step+=1
