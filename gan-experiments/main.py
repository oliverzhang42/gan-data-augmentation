import sys
import os
from CGAN import CGAN
from utils import load_mnist

def check_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

generator = [32, 64, 64]
discriminator = [64, 64, 32]
batch_size = 64
results_dir = sys.argv[1]
check_folder(results_dir)
checkpoint_dir = None
logs_dir = None
dataset_name = 'mnist'
epochs = int(sys.argv[2])

c = CGAN(generator, discriminator, batch_size, results_dir, checkpoint_dir, logs_dir, dataset_name, epochs)
c.train()