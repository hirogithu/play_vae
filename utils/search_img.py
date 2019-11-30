""" Get Imgae likely input image == min(dist of latent vector)"""

import torch
import pandas as pd
from imageio import imread
from dlutils import batch_provider

from make_search_space import process_batch
from vae import VAE

PATH = "VAEmodel.pkl"

def calic_dist(a, b):
    # cosine similality
    return a - b

def get_latent_data(key_id):
    if key_id == -1:
        #key_latent_data
        data = pd.from_csv("key.csv")
    else:
        #likely_latent_data
        data = pd.from_csv("likely_"+str(key_id)+".csv")
    return data

def encode_img(img):
    """Take img into model, Get latent vector """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    z_size = 64
    vae = VAE(zsize=z_size, layer_count=5)
    vae.load_state_dict(torch.load(PATH))
    vae.eval()
    input_data = [imread(img)]
    batches = batch_provider(input_data, batch_size, process_batch, report_progress=False)
    encoded_vector, _, _ = vae(batches)
    return encoded_vector

def find_img(vector):
    """ Caliculate dist of (vector, key:=[center of gravity] data),
    Search likely data, and Search most likey data."""
    key_id = 0
    key_latent_data = get_latent_data(-1)
    for i, key_data in enumerate(key_latent_data):
        if i == 0:
            dist_min = calic_dist(vector, key_data)
        else:
            dist_i = calic_dist(vector, key_data)
            if min(dist_min, dist_i) == dist_i:
                dist_min = dist_i
                key_id = i

    ls_img = []
    dist_tmp = []
    likely_latent_data = get_latent_data(key_id)
    for i, likely_data in enumerate(likely_latent_data):
        dist_tmp.append((i, calic_dist(vector, likely_data)))
    dist_tmp = sorted(dist_tmp, key=1)
    for k in range(5):
        ls_img.append(likely_latent_data[dist_tmp[k][0]])
    return ls_img

def get_likely_image(img_path):
    """1:ENCODE input, GET the latent vector, 2:SEARCH latent space, 3:GET most likely img."""
    v_enc = encode_img(img_path)
    img_list = find_img(v_enc)
    return img_list
