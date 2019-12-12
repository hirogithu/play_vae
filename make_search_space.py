"""train VAE, save weight and latent vector"""

# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import print_function
import glob
import time
import random
import os

import torch.utils.data
#from scipy import misc
from torch import optim
from torchvision.utils import save_image
import numpy as np
#import pickle
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from PIL import Image
from imageio import imread

from matplolib.pyplot as plt

from vae import VAE # from net import *

im_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1

def process_batch(batch):
    #data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]
    data = [np.array(Image.fromarray(x).resize([im_size, im_size])).transpose((2, 0, 1)) for x in batch]

    if device == "cuda":
        x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.

    else:
        x = torch.from_numpy(np.asarray(data, dtype=np.float32)) / 127.5 - 1.

    x = x.view(-1, 3, im_size, im_size)

    return x

def plt_rec_kl_loss(rec_loss, kl_loss):
    plt.plot(rec_loss_history, label="rec loss")
    plt.plot(kl_loss_history, label="kl loss")
    plt.title("Loss rec/kl")
    plt.legend()
    plt.show()    


def main():
    batch_size = 128
    z_size = 64
    vae = VAE(zsize=z_size, layer_count=5)
    if device == "cuda":
        vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.0005

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    train_epoch = 20

    #sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)

    dir_ls = glob.glob("train_img/PINS/*")

    rec_loss_history = []
    kl_loss_history = []

    for epoch in range(train_epoch):
        vae.train()

        # with open('data_fold_%d.pkl' % (epoch % 5), 'rb') as pkl:
            #data_train = pickle.load(pkl)
        data_train_ls = []
        for i, dir_ in enumerate(dir_ls[epoch%10*10:(epoch+2)%10*10]):
            data_train_ls.append(glob.glob(dir_+"/*.jpg"))

        data_train = []
        for dir_ in data_train_ls:
            # print(epoch, len(dir_), dir_[0].split("\\")[-1])
            for x in dir_[(epoch//10*10)%len(dir_):(epoch//10*10+20)%len(dir_)]:
                data_train.append(imread(x))

        #print(len(data_train))#, len(data_train[0]))
        # data_train_ls = glob.glob("train_img/PINS/pins_zendaya/*.jpg")
        # data_train = [imread(x) for x in data_train_ls]
        #print("Train set size:", len(data_train))
        #continue

        random.shuffle(data_train)

        batches = batch_provider(data_train, batch_size, process_batch, report_progress=False)

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x in batches:
            vae.train()
            vae.zero_grad()
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            #############################################

            os.makedirs('results/rec', exist_ok=True)
            os.makedirs('results/gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if 0:
            # if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

        print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
            (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))

        rec_loss_history.append(rec_loss)
        kl_loss_history.append(kl_loss)

        if ((epoch + 1) % 10 == 0) and (epoch + 1 != train_epoch):
            torch.save(vae.state_dict(), "VAEmodel.pkl")
        del batches
        del data_train
    
    plt_rec_kl_loss(rec_loss, kl_loss)

    print("Training finish!... save training results")
    torch.save(vae.state_dict(), "VAEmodel.pkl")


if __name__ == '__main__':
    main()
