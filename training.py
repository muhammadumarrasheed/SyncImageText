import torch
from torch.utils.data import DataLoader
import clip
from clip_dataset import ClipDataset
import torch.nn.functional as F

def evaluate(model, data_loader):
    model.eval()
    total_similarity = 0
    count = 0

    with torch.no_grad():
        for texts, images in data_loader:
            # Ensure text input is correctly shaped. If batch size is 1, squeeze out only if necessary.
            if texts.size(0) == 1:
                texts = texts.squeeze(0)
            # Ensure image input is correctly shaped.
            images = images.squeeze(0) if images.size(0) == 1 else images

            text_features = model.encode_text(texts)
            image_features = model.encode_image(images)

            # Normalize features to compute cosine similarity
            text_features = F.normalize(text_features, p=2, dim=-1)
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Compute cosine similarity
            similarity = (text_features * image_features).sum(dim=1)
            total_similarity += similarity.sum().item()  # Sum over the batch
            count += texts.size(0)  # Total number of items

    average_similarity = total_similarity / count
    return average_similarity


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    eval_data = ClipDataset(
        csv_file='dataset/twitter/test_posts.csv',
        img_dir='dataset/twitter/twitter_cleaned/images_test',
        transform=preprocess
    )
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)

    average_similarity = evaluate(model, eval_loader)
    print(f'Average Cosine Similarity: {average_similarity}')

if __name__ == "__main__":
    main()
