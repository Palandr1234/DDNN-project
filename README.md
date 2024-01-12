This repository is an PyTorch implementation of the DnCNN image denoising using R2R scheme for unsupervised learning

## 1.Training the model

### (1).Dependencies

- [PyTorch](http://pytorch.org/)(<0.4)
- [torchvision](https://github.com/pytorch/vision)
- OpenCV for Python
- [HDF5 for Python](http://www.h5py.org/)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### (2). Training models on AWGN noise

Train R2R model for AWGN noise level $\sigma =25$:

```
python3 train_AWGN.py --prepare_data --noiseL 25 --val_noiseL 25 --training R2R
```
## 2. Testing the model
Test R2R model for AWGN noise level $\sigma = 25$

```
python3 test_AWGN.py --test_noiseL 25 --training R2R
```
