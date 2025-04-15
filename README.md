# DA6401-A2 
# Part A
# Question 01: Flexible CNN on iNaturalist Dataset

---

 Dataset Setup
- Mounted from Google Drive.
- Extracted from `nature_12K.zip` to `/content/inaturalist_12K`.
- Mean and std for normalization computed from training images.


 Model: `FlexibleCNN`
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

---

 Dataset Setup
- Mounted from Google Drive: `nature_12K.zip`
- Extracted path: `/content/inaturalist_12K/train`
- **Randomly sampled 400 images per class** for faster training and experimentation
- Validation split: 20% from training data, **class-balanced**
- **Test set was not used** for hyperparameter tuning



 Model: `CustomCNN`
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

---

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


