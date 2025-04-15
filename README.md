# DA6401-A2
# Question 01: Flexible CNN on iNaturalist Dataset

This experiment builds a configurable CNN using PyTorch Lightning to classify images from the iNaturalist 12K dataset.

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

This experiment involves training a configurable CNN using PyTorch Lightning on the iNaturalist 12K dataset with **hyperparameter tuning via Weights & Biases (W&B) sweep**.

---

 Dataset Setup
- Mounted from Google Drive: `nature_12K.zip`
- Extracted path: `/content/inaturalist_12K/train`
- **Randomly sampled 400 images per class** for faster training and experimentation
- Validation split: 20% from training data, **class-balanced**
- **Test set was not used** for hyperparameter tuning



 Model: `CustomCNN`
Configurable CNN with:
- 5 convolutional blocks
- Flexible filter layouts (same/double/half)
- Optional BatchNorm, Dropout
- Dense layer with ReLU before final output

 Config Parameters
- `base_filter`: base #filters (64/128)
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



