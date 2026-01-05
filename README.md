The system processes each image through four parallel forensic branches followed by attention-based fusion to classify them

**1. Semantic Branch (CLIP)**

- Uses a frozen CLIP (ViT-B/16) image encoder
- Captures high-level semantic and logical inconsistencies
- Helps detect images that “look right locally but feel wrong globally”

**2. Frequency Branch**

- Computes FFT-based radial power spectra
- Models unnatural frequency distributions introduced by generative models

**3. Noise Residual Branch**

- Uses SRM-style high-pass filters
- Extracts sensor-level noise residuals
- Highly effective at detecting diffusion artifacts and missing camera noise

**4. Statistics & Color Branch**

- Handcrafted forensic features:
- Color moments (RGB, YCbCr)
- Patch-level statistics
- Patch self-similarity
- Designed to capture non-physical color relationships and texture repetition 

**Fusion**

- Outputs from all branches are combined using a learned attention mechanism
- Final decision is produced from the weighted forensic representation

**Data Pipeline**

- To avoid shortcut learning and improve real-world robustness
- No resizing of original images
- Random 256×256 crops during training
- Universal JPEG recompression during training
- 5-crop evaluation (corners + center) during validation and testing
- Final prediction is obtained by averaging logits across crops

**Dataset**

- Large-scale dataset with 100,000+ images
- Balanced real vs AI-generated samples
- Explicit train / validation / test splits
- Test set is never used during training or validation

**Results**

_Validation Performance_ 
- ROC-AUC: 0.9968

_Test Performance_
- ROC-AUC: 0.9974
- Accuracy: 97.2%


**File Structure**

    image-authenticity-analysis/
    │
    ├── src/                        # Core model and dataset code
    │   ├── dataset.py              # Dataset loading & preprocessing logic
    │   ├── model_clip.py           # CLIP semantic branch
    │   ├── model_frequency.py      # Frequency-domain forensic branch
    │   ├── model_noise.py          # Noise residual (SRM-based) branch
    │   ├── model_stats_n_color.py  # Statistical & color forensic branch
    │   ├── model_fusion.py         # Attention-based fusion module
    │   └── model_full.py           # Full end-to-end forensic model
    │
    ├── scripts/                    # Training, evaluation, analysis scripts
    │   ├── training_full.py        # Full end-to-end training script
    │   ├── test_evaluation.py      # Final evaluation on test split
    │   
    ├── data/
    │   ├── raw/
    │   │   ├── real/               # Real images
    │   │   └── fake/               # AI-generated images
    │   └── metadata.csv            # maps images tio train/val/test split + labels
    │
    ├── reports/
    │   ├── best_model.pth          # Best trained model checkpoint
    │   
    ├── README.md
    └── requirements.txt
    

**HOW TO RUN**

1. Create environment

        conda create -n img_auth_det python=3.11
        conda activate img_auth_det

2. Install Depencencies

        pip install -r requirements.txt
   
3. Dataset Preparation

     - link :  https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset/data
     - place real images to data/raw/real
     - place fake images to data/raw/fake
  
   _data/metadata.csv is already included in the repo and maps the images to these paths automatically_
  
4. Run
   
   1. - Train the model

            cd scripts
            python training_full.py

      - this saves the best model to reports/best_model.pth
  
   2. Evaluate on test set

           cd scripts
           python test_evaluation.py

    
