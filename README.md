# EEG Classification - Dataset Setup for Google Colab  

This repository contains code for **EEG classification** using machine learning and deep learning models. The datasets required for training and evaluation are hosted on **Kaggle**, and this guide provides steps to download and set up the data in **Google Colab**.  

## Dataset Preparation in Google Colab  

### Step 1: Install Kaggle API and Configure Credentials  
Before downloading the datasets, you need to set up your **Kaggle API key**:  

1. Download `kaggle.json` from your Kaggle account:  
   - Go to [Kaggle](https://www.kaggle.com/).  
   - Click on your profile picture (top-right corner) → **Settings**.  
   - Scroll to **API** and click **"Create New API Token"**.  
   - This downloads a `kaggle.json` file.  

2. Upload `kaggle.json` to Google Colab:  

   ```python
   from google.colab import files
   files.upload()  # Manually upload kaggle.json
   ```

3. Configure Kaggle API:  

   ```bash
   !pip install -q kaggle
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 2: Download Required Datasets  
Once the Kaggle API is set up, run the following commands to download and extract the necessary datasets:  

```bash
!kaggle datasets download -d cdeotte/brain-spectrograms -p /content/brain_spectrograms --unzip
!kaggle datasets download -d cdeotte/kaggle-kl-div -p /content/kaggle_kl_div --unzip
!kaggle datasets download -d cdeotte/brain-eeg-spectrograms -p /content/brain_eeg_spectrograms --unzip
!kaggle competitions download -c hms-harmful-brain-activity-classification -p /content/hms_harmful_activity --unzip
```

### Step 3: Load Datasets into Pandas  
After downloading, load the datasets into Pandas for further processing:  

```python
import pandas as pd
import numpy as np
import os

train_df = pd.read_csv('/content/hms_harmful_activity/train.csv')
test_df = pd.read_csv('/content/hms_harmful_activity/test.csv')
spectrogram_df = pd.read_parquet('/content/brain_spectrograms/sample_spectrogram.parquet')
eeg_specs = np.load('/content/brain_eeg_spectrograms/eeg_specs.npy', allow_pickle=True).item()

import sys
sys.path.append('/content/kaggle_kl_div')
from kaggle_kl_div import score
```

### Step 4: Verify Data Integrity  
Ensure that the datasets have been loaded correctly:  

```python
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("Spectrogram Sample:", spectrogram_df.head())
print("EEG Spectrogram Keys:", eeg_specs.keys())
```

## Dataset Links  
These datasets are used in this project and can be found on Kaggle:  

1. 🧠 **Brain Spectrograms** - [View on Kaggle](https://www.kaggle.com/datasets/cdeotte/brain-spectrograms)  
2. 📊 **Kaggle KL-Divergence Scoring Dataset** - [View on Kaggle](https://www.kaggle.com/datasets/cdeotte/kaggle-kl-div)  
3. 🧠 **Brain EEG Spectrograms** - [View on Kaggle](https://www.kaggle.com/datasets/cdeotte/brain-eeg-spectrograms)  
4. ⚡ **HMS Harmful Brain Activity Competition** - [View on Kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)  

## About This Repository  
This repository is dedicated to **EEG classification** using machine learning and deep learning techniques. The goal is to develop a **robust classification model** for identifying harmful brain activity patterns. Contributions and suggestions are welcome! 🚀  

### License  
This project is open-source and available under the **MIT License**.  

