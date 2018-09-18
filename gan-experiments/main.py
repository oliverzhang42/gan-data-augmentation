from CGAN import CGAN
from utils import load_mnist

encoding = 100
generator = [8, 16, 32]
discriminator = [32, 16, 8]
batch_size = 64
results_dir = "results"
checkpoint_dir = None
logs_dir = None
dataset_name = 'mnist'
epochs = 30

c = CGAN(encoding, generator, discriminator, batch_size, results_dir, checkpoint_dir, logs_dir, dataset_name, epochs)
c.train()