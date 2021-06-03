import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from IPython.display import Image
%matplotlib inline

DATA_DIR = 'drive/MyDrive/Colab Notebooks/Berry'
#print(os.listdir(DATA_DIR))

image_size = 64
batch_size = 128 #halom méret
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

#kivágás, átméretezés a programnak 0.5os átlagos szórás
train_dataset = ImageFolder(DATA_DIR, transform=tt.Compose([
    tt.Resize(image_size),  #átméretezés img méretre
    tt.CenterCrop(image_size),  #középkivág img méretre
    tt.ToTensor(),  #átkonvertálja a képeket "számokká" (pixelek szine 0-255ig)
    tt.Normalize(*stats)])) #minden értékből kivonja az átlagot majd elosztja a szórással

# több adat befogadása
# Itt létrehozunk egy train_loader nevű adatbetöltőt 
# amely összekeveri a train_dataset adatait, és 128 minta kötegét adja vissza, amelyeket az ideghálózatok képzésére fog használni.
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)


# betöltött dataset mutatása 
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
"""
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

show_batch(train_dl) 
"""
# erőforrás kiválasztása megfelelő default érték alapján (másolt kódrészlet)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
# itt a vége 


# A GAN diszkriminátora egyszerűen osztályozó. 
# Megpróbálja megkülönböztetni a valós adatokat a generátor által létrehozottaktól.
# A diszkriminátor "megbünteti" a generátort azért, mert valószínűtlen eredményeket produkál.

discriminator = nn.Sequential( #segíthet abban, hogy több modult csoportosítsunk.
    # be: 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # ki: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # ki: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # ki: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # ki: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # ki: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())

discriminator = to_device(discriminator, device)
latent_size = 128

# A generátor megtanul hiteles adatokat generálni. A generált példányok negatív képzési példákká válnak a diszkriminátor számára.

generator = nn.Sequential(
    # bemenet : latent_size x 1 x 1
    # 2D transzponált konvolúciós operátort alkalmaz egy bemeneti képre, amely több bemeneti síkból áll.
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True), # Segítenek a modellnek megismerni a bemenet és a kimenet közötti összetett kapcsolatokat.
    # ki: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # ki: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # ki: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # ki: 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # ki: 3 x 64 x 64
)
xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
fake_images = generator(xb)
#print(fake_images.shape) #alap "zaj"
show_images(fake_images)

generator = to_device(generator, device)

def train_discriminator(real_images, opt_d):
    # Diszkriminátor színátmenet törlés
    opt_d.zero_grad()

    # Átmegy a valós kép a diszkriminátoron
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Hamis képek generálása
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Átmegy a hamis kép a diszkriminátoron
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Diszkriminátor adat fissités
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(opt_g):
    # Generátor színátmenet törlés
    opt_g.zero_grad()
    
    # Hamis képek generálása
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Megpróbáljuk becsapni a generátort
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generaltkep-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Mentes', fake_fname)

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

save_samples(0, fixed_latent)

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # vesztés, értékekre szolgálló tömb
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Optimalizáló létrehozása
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Azt akarjuk, hogy modellünk többször is végigmenjen a teljes adatkészleten, ezért a for ciklust használjuk. 
    # Valahányszor átmegy a teljes képhalmazon, korszaknak hívják. (epoch)
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Diszkriminátor tanulása
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Generátor tanulása
            loss_g = train_generator(opt_g)
            
        # Rögzítse az adatokat
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Adat kiírás
        print("Korszak [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Generált képek mentése
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

# Meghatározza a tanulási arányt (lr), amelyet a hálózati súlyok hozzáigazításához használ.
lr = 0.0002
# Tanulási korszak
epochs = 500

history = fit(epochs, lr)
losses_g, losses_d, real_scores, fake_scores = history

Image('./generated/generaltkep-0065.png')

vid_fname = 'gans_training.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
files.sort()

out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'mp4v'), 1, (530,530))
[out.write(cv2.imread(fname)) for fname in files]
out.release()
