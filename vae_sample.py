""" load vae weight, get encode img and decode img"""
import glob

import torch
from torchvision.utils import save_image
from dlutils import batch_provider
from imageio import imread

from vae import VAE
from make_search_space import process_batch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
PATH = "VAEmodel.pkl"
im_size = 128


def main():
    batch_size = 128
    z_size = 64
    vae = VAE(zsize=z_size, layer_count=5)
    vae.load_state_dict(torch.load(PATH))
    vae.eval()

    img_path_ls = glob.glob("output_img/face_*.jpg")
    input_data = [imread(path) for path in img_path_ls]
    batches = batch_provider(input_data, batch_size, process_batch, report_progress=False)

    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)

    for x in batches:
        x_rec, _, _ = vae.forward(x)
        print(x_rec.shape)
#        x_rec, _, _ = vae(x)
        resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
        resultsample = resultsample.cpu()
        save_image(resultsample.view(-1, 3, im_size, im_size),
                    'results/rec/sample_encode_.png')
        x_rec = vae.decode(sample1)
        resultsample = x_rec * 0.5 + 0.5
        resultsample = resultsample.cpu()
        save_image(resultsample.view(-1, 3, im_size, im_size),
                    'results/gen/sample_decode_.png')


if __name__ == '__main__':
    main()
