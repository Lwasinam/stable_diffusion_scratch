from dataset import ImageDataset
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import model_loader
from ddpm import DDPMSampler
from pipeline import get_time_embedding
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
from configuration import get_config
from diffusion import Diffusion 
from encoder import VAE_Encoder
device = 'cuda'
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler



cfg = get_config()
def get_dataset(tokenizer, cfg):
    dataset = load_dataset("diffusers/pokemon-gpt4-captions")

    train_ds = ImageDataset(dataset, tokenizer )

    training_dataloader = DataLoader(train_ds, batch_size = cfg['batch_size'], shuffle = True)
    print('Loaded dataset')
    return training_dataloader

def get_tokenizer():
    tokenizer = CLIPTokenizer('/content/stable_diffusion_scratch/data/vocab.json', merges_file = '/content/stable_diffusion_scratch/data/merges.txt')

    print('Loaded tokenizer')
    return tokenizer


# def get_model():
#     model_file = "../STABLE_DIFFUSION/data/v1-5-pruned-emaonly.ckpt"
#     models = model_loader.preload_models_from_standard_weights(model_file, device)
#     print('Loaded models')
#     return models

def train_model(cfg):
    batch_size = 3
    tokenizer = get_tokenizer()
    training_dataloader = get_dataset(tokenizer, cfg)
    # models = get_model()
    model = Diffusion().to(device)
    # encoder = VAE_Encoder().to(device)
    # clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # encoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).to(device)

    
    seed = 42

    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    sampler = DDPMSampler(generator)    

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg['lr'])
    loss_fn = nn.CrossEntropyLoss().to(device)
    # timesteps = torch.from_numpy(np.arange(0, 1000)[::-1].copy()).to(device)

    for epoch in range(0, 20):
        model.train()
        # clip.eval()
        # encoder.eval()
        batch_iterator = tqdm(training_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            image = batch['image_tensor'].to(device)
            tokens = batch['text_tokens'].to(device)
            # timestep = timesteps[random.randrange(1000)]
            # forward pass

            noise = torch.randn((cfg['batch_size'], 4, 128//8, 128//8), device = device)
            # # print(image.shape)
            # # latents = encoder(image, noise)
            
            # latents = encoder.encode(image).latent_dist.sample(generator)
            # # print(latents.shape) 

            # latents = sampler.add_noise(latents, timestep).to(device)

           
            # time_embedding = get_time_embedding(timestep).to(device)

           
            # context = clip(tokens)
           
           
            # print(latents.shape)

            predicted_noise = model(noise, torch.randn(1,77,512).to(device), torch.rand(1,320).to(device))
            # print(predicted_noise.shape)
            # print(noise.shape)
            loss = loss_fn(predicted_noise.reshape(-1, 16 * 16 *4), noise.reshape(-1, 16* 16*4))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    train_model(cfg)

            


