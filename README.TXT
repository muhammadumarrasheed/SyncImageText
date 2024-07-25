Sync ImageText - Image and Text Analysis Platform
==================================================

Introduction:
-------------
Sync ImageText is a cutting-edge platform that leverages the power of CLIP (Contrastive Language–Image Pre-training) to evaluate and understand the relationship between textual descriptions and corresponding images. This tool is designed to enhance the way digital content is interacted with, making it more intuitive and effective for users.

Features:
---------
- Image and text analysis using pre-trained CLIP model.
- Cosine similarity calculations to measure the relevance between text and images.
- Batch processing capabilities for efficient data handling.
- Support for various image formats and detailed text inputs.

Getting Started:
----------------
1. Download the dataset using https://drive.google.com/file/d/1maf12DjW4ph-fDLR8KkjgyzTvfrQASfy/view?usp=sharing
2. Ensure Python and necessary libraries (torch, clip, PIL) are installed.
3. Place your dataset in the designated directories as outlined in the project structure.
4. Run the provided scripts to train or evaluate the model with your data.

Project Structure:
------------------
- dataset/
  - twitter/
    - test_posts.csv
    - train_posts_clean.csv
    - twitter_cleaned/
      - images_test/
      - images_train/
- src/
  - clip_dataset.py   # Dataset handler for loading and preprocessing data.
  - evaluation.py     # Script for evaluating the model performance.
  - training.py       # Script for training the model on provided data.

Usage:
------
- Training: Run `training.py` to train the model on your image-text dataset.
- Evaluation: Run `evaluation.py` to calculate the average cosine similarity of the trained model.

Output:
-------
The output will provide insights into the average cosine similarity between the evaluated texts and images, indicating how well the model can relate text descriptions to visual content.

