# EEG Classification - Dataset Setup for Google Colab  

This repository contains code for **EEG classification** using machine learning and deep learning models. The datasets required for training and evaluation are hosted on **Kaggle**, and this guide provides steps to download and set up the data in **Google Colab**.  

---

## Dataset Preparation in Google Colab  

### Step 1: Install Kaggle API and Configure Credentials  

Before downloading the datasets, you need to set up your **Kaggle API key**:  

1. Download `kaggle.json` from your Kaggle account:  
   - Go to [Kaggle](https://www.kaggle.com/).  
   - Click on your profile picture (top-right corner) → **Settings**.  
   - Scroll to **API** and click **"Create New API Token"**.  
   - This downloads a `kaggle.json` file.  

2. Upload `kaggle.json` to Google Colab:  

   ```python```
   from google.colab import files
   files.upload()  # Manually upload kaggle.json

3. Configure Kaggle API:

  # Install kaggle package if not installed
  !pip install -q kaggle
  
  # Move kaggle.json to the right location and set permissions
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json

### Step 2: Download Required Datasets
Once the Kaggle API is set up, run the following commands to download and extract the necessary datasets:

# 1. Download Brain Spectrograms dataset
!kaggle datasets download -d cdeotte/brain-spectrograms -p /content/brain_spectrograms --unzip

# 2. Download Kaggle KL-Divergence scoring dataset
!kaggle datasets download -d cdeotte/kaggle-kl-div -p /content/kaggle_kl_div --unzip

# 3. Download Brain EEG Spectrograms dataset
!kaggle datasets download -d cdeotte/brain-eeg-spectrograms -p /content/brain_eeg_spectrograms --unzip

# 4. Download HMS Harmful Brain Activity Competition dataset
!kaggle competitions download -c hms-harmful-brain-activity-classification -p /content/hms_harmful_activity --unzip
Step 3: Load Datasets into Pandas
After downloading, load the datasets into Pandas for further processing:

import pandas as pd
import numpy as np
import os

# Load CSV files (if applicable)
train_df = pd.read_csv('/content/hms_harmful_activity/train.csv')
test_df = pd.read_csv('/content/hms_harmful_activity/test.csv')

# Load Spectrogram Parquet files
spectrogram_df = pd.read_parquet('/content/brain_spectrograms/sample_spectrogram.parquet')

# Load EEG Spectrograms (NumPy format)
eeg_specs = np.load('/content/brain_eeg_spectrograms/eeg_specs.npy', allow_pickle=True).item()

# Load KL-Divergence scoring function (if needed)
import sys
sys.path.append('/content/kaggle_kl_div')
from kaggle_kl_div import score
Step 4: Verify Data Integrity
Ensure that the datasets have been loaded correctly:

print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("Spectrogram Sample:", spectrogram_df.head())
print("EEG Spectrogram Keys:", eeg_specs.keys())
Dataset Links
These datasets are used in this project and can be found on Kaggle:

🧠 Brain Spectrograms
🔗 View on Kaggle

📊 Kaggle KL-Divergence Scoring Dataset
🔗 View on Kaggle

🧠 Brain EEG Spectrograms
🔗 View on Kaggle

⚡ HMS Harmful Brain Activity Competition
🔗 View on Kaggle


