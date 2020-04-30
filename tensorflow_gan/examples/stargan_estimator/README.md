## StarGAN Estimator

Note: it is based on stargan estimator doc and refactor part of code. Add evaluation.py to load checkpoint and transform data.

### env version:
tensorflow = "1.15.0"
tensorflow_datasets = "2.1.0"
tensorflow_probability = "0.8.0"

### How to run

0. Please do NOT
```
pip install tensorflow_gan
```
Instead, glone this repo to set up python path as instructions.

1.  Run the setup instructions in [tensorflow_gan/examples/README.md](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/README.md#steps-to-run-an-example)
2.  Trainning: Run:

```
python stargan_estimator/train.py
```
After training is done, it will generate `train_result.json`. It contains train parameters which can be used in the `evaluation.py`.
Please load `train_result.json` in `evaluation.py`.
```
python stargan_estimator/evaluation.py
```

### Description

An estimator implementation of StarGAN that abstracts away the training
procedure. And add evalaution function to load checkpoint and transform images.

<img src="images/stargan_estimator.png" title="StarGAN Estimator" width="500" />
