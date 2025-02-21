# MiniSlot

A mini implementation of Object-Centric Learning with Slot Attention.

## Overview

This repository implements a simplified version of the Slot Attention module and its integration into an autoencoder for unsupervised object discovery.

## Folder Structure

- `config/`: Configuration files.
- `data/`: Scripts or instructions to generate/load synthetic data.
- `models/`: Model implementations (encoder, slot attention, decoder, autoencoder).
- `utils/`: Utility functions (e.g., positional encoding, visualization helpers).
- `experiments/`: Notebooks or scripts for evaluation and visualization.
- `train.py`: Training script.
- `evaluate.py`: Evaluation script.

## Setup

1. Clone repository and install dependencies:
   ```bash
   git clone https://github.com/sabrikhalil/MiniSlot
   pip install -r requirements.txt
   ```

2. Download_dataset: 
   ```bash 
   python data/download_dataset.py
   ``` 
    
3. Run the training script:
   ```bash
   python train.py
   ```

4. Evaluate the model:
   ```bash
   python evaluate.py
   ```


