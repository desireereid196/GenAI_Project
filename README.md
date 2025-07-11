<h1>
    <img src="./vtt_logo.png" alt="GHLightLogo" align="left" alt="Sample Image" class="image-left" width="80" height="80" style="padding: 10px;"/>
    Vision-to-Text: An Image Captioning System
</h1>
<br>

## 🔍 Overview

This project bridges **Computer Vision** and **Natural Language Processing**, focusing on:

- Encoder-decoder architecture for image captioning
- Transfer learning with pretrained CNNs
- LSTM-based language modeling
- Embedding-based and generative evaluation metrics

## 📌 Description

**VTT (Vision-to-Text)** is a modular deep learning pipeline for image captioning that translates visual content into coherent, semantically meaningful natural language descriptions. It combines computer vision (CV) and natural language processing (NLP) techniques via an encoder-decoder architecture using ResNet-50 and LSTM networks.

This project supports both the **[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)** and **[Flickr30k](https://www.kaggle.com/datasets/awsaf49/flickr30k-dataset)** datasets and includes tools for:
- Image preprocessing & feature extraction
- Caption tokenization, filtering, and padding
- Training-ready sequence generation
- Metric-based and qualitative evaluation


## 📁 Repository Structure

```bash
GenAI_Project/
├── LICENSE                           # MIT License.
├── README.md                         # Repository overview and setup.
├── pyproject.toml                    # Project configuration file.
├── archive                           # Old stuff.
├── data/                             # 
    ├── raw/                          # Raw data
    └── processed/                    # Cleaned and processed data; tokenizers.
├── documents/                        # Project milestones, research notes, etc.
├── figures/                          # Performance plots.
├── models/                           # Trained models.
├── notebooks/                        # Development and experiment notebooks.
    ├── experiment_1.ipynb            # Baseline Training and Evaluation. (Research Question 1) 
    ├── experiment_2.ipynb            # Error Analysis. (Research Questions 2 and 4)
    ├── experiment_3.ipynb            # Semantic Fidelity Comparison. (Research Question 3)
    └── experiment_4.ipynb            # Generalization. (Research Question 1)
├── outputs/                          # Model runner outputs needed for Milestone 3.
├── scripts/                          # 
    ├── data_runner.py                # Data pipeline script needed for Milestone 2.
    ├── model_runner.py               # Model pipeline script needed for Milestone 3.
    ├── train_model.py                # Model training script.
    ├── preprocess_captions.py        # Caption preprocessing script.
    └── extract_features.py           # Feature extraction script.
└── src/                              # Contains the core source code.
    └── vtt/                          # The main package for the project.
        ├── __init__.py               #
        ├── config.py                 # Configuration file for project.
        ├── utils.py                  # Shared helper and utility functions.
        ├── captions/                 #
            ├── __init__.py           #
            ├── cleaning.py           # Load and clean captions.
            ├── vocabulary.py         # Count word frequencies and filter captions.
            ├── tokenization.py       # Fit tokenizer, convert captions to sequences, etc.
            ├── padding.py            # Pad caption sequences.
            └── io.py                 # Save and load padded sequences.
        └── features/                 #
            ├── __init__.py           #
            ├── preprocessing.py      # Image preprocessing.
            ├── extractor.py          # ResNet feature extraction.
            └── batch_runner.py       # Batch processing and saving.
        ├── models/                   #
            ├── __init__.py           #
            ├── architecture.py       # Model building.
            ├── data_loader.py        # Data pipeline setup.
            └── trainer.py            # Training orchestration.

```

## 🛠 Setup

Follow the steps below to set up the project locally for development, experimentation, or training.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/GenAi_Project.git
cd GenAI_Project
```

### 2. Create a Virtual Environment 

We recommend using Python 3.10+ with TensorFlow 2.9+ support.

```bash
conda create -n genai_project python=3.10
conda activate genai_project
```

### 3. Install the Package and Dependencies

Install the `vtt` package in editable (`-e`) mode so you can make changes to the source code and test them without reinstalling.

```bash
# Ensure you are at the top-level of the GenAI_Project repository
pip install -e .
```

This installs the package locally while keeping it linked to the source code.

### 4. Verify the Installation

Verify that the `vtt` package was installed and is importable.

```bash
# Ensure your virtual environment is active
python -c "import vtt; print('vtt imported successfully')"
```

## 📄 License
MIT License — feel free to use, share, and modify.

## 🤝 Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 🧠 Project Maintainers
- [Curtis Neiderer](mailto:neiderer.c@northeastern.edu)
- [Divya Maheshkumar](maheshkumar.d@northeastern.edu)
- [Desiree Reed](reed.des@northeastern.edu)
- [Minal Ahir](ahir.m@northeastern.edu")
- [Arundhati Ubhad]("ubhad.a@northeastern.edu")
- Contributors welcome!