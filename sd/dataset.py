import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
device = 'cpu'
from pipeline import rescale

class ImageDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = dataset
        self.tokenizer = tokenizer
        super().__init__()
    def __len__(self):
      
        return len(self.data['train'])
    def __getitem__(self, idx):
        image = self.data['train'][idx]['image']
        text = self.data['train'][idx]['text'] 

        

        input_image_tensor = image.resize((512, 512))
        input_image_tensor = np.array(input_image_tensor)

        # (Height, Width, Channel)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1 ))

        input_image_tensor =  input_image_tensor.permute(2, 0,1)


                 # convert the prompt into tokens using the tokenizer
        cond_tokens = self.tokenizer([text], padding = 'max_length', max_length = 77).input_ids
            # (batch_size, seq_len)   
        cond_tokens = torch.tensor(cond_tokens, dtype= torch.long, device = device)
        cond_tokens = cond_tokens.squeeze(0)
        return {
            'image_tensor': input_image_tensor,
            'text_tokens': cond_tokens
        }

   

      
