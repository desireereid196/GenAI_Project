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
├── LICENSE                               # MIT License.
├── README.md                             # Repository overview and setup.
├── pyproject.toml                        # Project configuration file.
├── archive                               # Old stuff.
├── data/                                 # 
    ├── raw/                              # Raw data.
    └── processed/                        # Cleaned and processed data; tokenizers.
├── documents/                            # Project milestones, research notes, etc.
├── figures/                              # Performance plots.
├── models/                               # Trained models.
├── notebooks/                            # Development and experiment notebooks.
    ├── experiment_1.ipynb                # Baseline Training and Evaluation. 
    ├── experiment_2.ipynb                # Error Analysis. 
    ├── experiment_3.ipynb                # Semantic Fidelity Comparison. 
    └── experiment_4.ipynb                # Generalization. 
├── outputs/                              # Model runner outputs needed for Milestone 3.
├── scripts/                              # 
    ├── data_runner.py                    # Data pipeline script needed for Milestone 2.
    ├── model_runner.py                   # Model pipeline script needed for Milestone 3.
    ├── train_model.py                    # Model training script.
    ├── preprocess_captions.py            # Caption preprocessing script.
    └── extract_features.py               # Feature extraction script.
└── src/                                  # Contains the core source code.
    └── vtt/                              # The main package for the project.
        ├── __init__.py                   #
        ├── config.py                     # Project configuration and dependencies.
        ├── utils.py                      # Shared helper and utility functions.
        ├── data/                         #
            ├── __init__.py               #
            ├── caption_preprocessing.py  # Caption cleaning/tokenization.
            ├── image_preprocessing.py    # Image feature extraction.
            └── data_loader.py            # tf.data.Dataset loaders.
        ├── evaluation/                   #
            ├── __init__.py               #
            ├── evaluation.py             # Caption evaluation logic.
            └── metrics.py                # BLEU, METEOR, ROUGE, BERTScore, CLIPScore, etc.
        ├── models/                       #
            ├── __init__.py               #
            ├── architecture.py           # Model architecture definitions.
            ├── train.py                  # Model training logic.       
            └── predict.py                # Caption generation from trained model.
```

## 🛠 Setup

Follow the steps below to set up the project locally for development, experimentation, or training.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/GenAi_Project.git
cd GenAI_Project
```

### 2. Install Git LFS

Install Git large file storage so large files (i.e., `NPZ` data files) are stored as pointers and not full file objects in the repository.

```bash
git lfs install
```

### 3. Create a Virtual Environment

We recommend using Python 3.10+ with TensorFlow 2.9+ support.

```bash
conda create -n genai_project python=3.10
conda activate genai_project
```

### 4. Install the Package and Dependencies

Install the `vtt` package in editable (`-e`) mode so you can make changes to the source code and test them without reinstalling.

```bash
# Ensure you are at the top-level of the GenAI_Project repository
pip install -e .
```

This installs the package locally while keeping it linked to the source code.

### 5. Verify the Installation

Verify that the `vtt` package was installed and is importable.

```bash
# Ensure your virtual environment is active
python -c "import vtt; print('vtt imported successfully')"
```

### 6. Install the Pre-Commit Hooks

This command sets up the necessary Git hooks (like pre-commit, pre-push, etc.) in the .git/hooks/ directory of your local clone, pointing them to the pre-commit framework.

```bash
pre-commit install
```

Important Note: Collaborators only need to run pre-commit install once per local clone of the repository.

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