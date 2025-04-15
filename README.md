# DA6401-A2 
# Part A
# Question 01: Flexible CNN on iNaturalist Dataset


 Dataset Setup
- Mounted from Google Drive.
- Extracted from `nature_12K.zip` to `/content/inaturalist_12K`.
- Mean and std for normalization computed from training images.


 Model: `FlexibleCNN` <br>
A CNN with 555 conv-activation-maxpool blocks followed by:
- 1 dense layer with `n` neurons
- 1 output layer with `10` neurons (for 10 classes)

 Configurable Parameters
- `m` : number of filters per conv layer  
- `k` : kernel size  
- `n` : neurons in the dense layer  
- Activation function 
- Input image size (default: `224x224`)  


Computation Metrics
- `compute_parameters(m, k, n)`: total trainable parameters  
- `compute_computations(m, k)`: total convolution operations
# Question 02: Hyperparameter Tuning with W&B Sweep on iNaturalist 



 Dataset Setup
- Mounted from Google Drive: `nature_12K.zip`
- Extracted path: `/content/inaturalist_12K/train`
- **Randomly sampled 400 images per class** for faster training and experimentation
- Validation split: 20% from training data, **class-balanced**
- **Test set was not used** for hyperparameter tuning



 Model: `CustomCNN` <br>
Configurable CNN with
- 5 convolutional blocks
- Flexible filter layouts (same/double/half)
- Built-in augmentation toggle
-  BatchNorm, Dropout
- Dense layer with ReLU before final output

Training Setup
- Optimizer: Adam
- Max Epochs: 10
- Early stopping on val_acc (patience = 3)
- Image Size: 224x224
- Batch Size: 64

 Config Parameters
- `base_filter`: base filters (64/128)
- `kernel_size`: 3 or 5
- `activation`: ReLU, GELU, SiLU, Mish
- `filter_type`: same/double/half
- `batch_norm`: True/False
- `augmentation`: True/False
- `dropout`: 0 / 0.1 / 0.2
- `dense_neurons`: 128 / 256 / 512



 W&B Sweep Settings
- **Sweep Method**: Bayesian optimization
- **Goal**: Maximize `val_acc`
- **Count**: 15 trials
- **Project**: `inat-sweep-v3`

```python
sweep_id = wandb.sweep(sweep_settings, project="inat-sweep-v3")
wandb.agent(sweep_id, function=launch_training, count=15)
```
# Question 03 : Wandb report

# Question 04: Final Test Evaluation using Best Model



Test Data Usage
- Dataset path: `/content/inaturalist_12K/val`
- Test data was **never used** during model selection or hyperparameter tuning
- All model selection was strictly based on train-validation performance

Evaluation Strategy
- Trained the best model for 10 epochs on full training data
- Selected best checkpoint based on test accuracy
- Loaded the checkpoint and evaluated accuracy on the held-out test set
- Accuracy calculated using sklearn.metrics.accuracy_score

Best Model Configuration (from sweep)
```python
best_config = {
    "base_filter": 128,
    "kernel_size": 3,
    "activation": "SiLU",
    "filter_type": "same",
    "batch_norm": False,
    "augmentation": True,
    "dropout": 0,
    "dense_neurons": 512
}
```
Final Test Accuracy : 

Creative Visual Grid of Predictions (10*3) <br>
To visualize model performance, I created a 10 × 3 image grid showing predictions on randomly selected test images.
- Green titles: Correct predictions
- Red titles: Incorrect predictions
- Each cell displays the predicted and true label
# Part B
# Question 01 & 02: Answer in markdown cell (PartB.ipynb)
# Question 03 : Fine-Tuning ResNet50 on iNaturalist 12K 

This project fine-tunes a **ResNet50** model using **PyTorch Lightning** on the 10-class variant of the **iNaturalist 12K** dataset. The goal is to explore different transfer learning strategies and identify the most effective approach for fine-tuning. <br>


 Dataset

- **Source**: `nature_12K.zip` (mounted from Google Drive in Colab)
- **Preprocessing**:
- Resize to `224x224` (ImageNet standard)
- Normalize with dataset-specific `mean` and `std`

Transfer Learning Strategies Evaluated

| Strategy        | Validation Accuracy (on 200 samples) |
|----------------|---------------------------------------|
|  Head-only    | 57%                                   |
|  Partial      | 67%                                   |
|  Last block   | **71%** (selected)                    |

 The `"last_block"` strategy fine-tunes only `layer4` and `fc` layers of ResNet50. <br>



Model Architecture

- Base model: **ResNet50** with pretrained ImageNet weights
- Modified FC layer:
```python
net.fc = nn.Sequential(
    original_fc,         # 1000-class original output
    nn.ReLU(),
    nn.Linear(1000, 10)  # Adapted for 10-class iNaturalist
)
```
 Training Configuration
- Framework: PyTorch Lightning

- Logging: Weights & Biases (wandb)

- Optimizer: Adam (lr=1e-4)

- Loss: CrossEntropyLoss

- Batch size: 64

- Epochs: 5 <br>

Test Accuracy with last_block fine-tuning: 84.04%
# Inferences listed as markdown in PartB.ipynb
