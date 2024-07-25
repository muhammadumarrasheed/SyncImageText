# clip_dataset.py
import pandas as pd
import os
from PIL import Image
import clip
import torch
from torch.utils.data import Dataset

class ClipDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or clip.load("ViT-B/32")[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        text = self.data_frame.iloc[idx]['post_text']
        img_path = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image_id'] + '.jpg')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Tokenize text and truncate to the maximum sequence length
        text_tokens = clip.tokenize([text], truncate=True)[:,:77].to(self.device)
        text_tokens = text_tokens.squeeze(0)  # Remove batch dimension

        return text_tokens, image
