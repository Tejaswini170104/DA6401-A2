# DA6401-A2
# Question 01: Flexible CNN on iNaturalist Dataset

This project builds a configurable CNN using PyTorch Lightning to classify images from the iNaturalist 12K dataset.

---

## Dataset Setup
- Mounted from Google Drive.
- Extracted from `nature_12K.zip` to `/content/inaturalist_12K`.
- Mean and std for normalization computed from training images.

---

## Model: `FlexibleCNN`
A CNN with 555 conv-activation-maxpool blocks followed by:
- 1 dense layer with `n` neurons
- 1 output layer with `10` neurons (for 10 classes)

### Configurable Parameters
- `m` : number of filters per conv layer  
- `k` : kernel size  
- `n` : neurons in the dense layer  
- Activation function 
- Input image size (default: `224x224`)  

---

## Computation Metrics
- `compute_parameters(m, k, n)`: total trainable parameters  
- `compute_computations(m, k)`: total convolution operations  


