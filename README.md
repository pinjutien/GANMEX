# GANMEX: One-vs-One Attributions Guided by GAN-generated Counterfactual Explanation Baselines

GANMEX is designed for 1) generating counterfactual explanations using Generative Adversarial Networks (GAN) and 2) producing class-targeted baselines for attribution methods such as integraded gradient (IG), DeepLIF, and DeepSHAP. GANMEX was built upon on the [TF-GAN](https://github.com/tensorflow/gan) implementation of [StarGAN](https://arxiv.org/abs/1711.09020) with the to-be-explained classifier incorporated as part of the adversarial networks. Such design allows the counterfactuals to be generated with respect to the trained classifier's perspective and therefore provide reliable explanations to the trained model.

![alt text](https://github.com/pinjutien/GANMEX/blob/master/figures/result1.jpg?raw=true)


## Requirements

Requires python 3.6 and tensorflow 1. Additional packages can be installed by the following:

```pip install -r requirements.txt```

## Classifier Training

Run `examples/classification/train_tfds.py`. The training script will produce a keras model `model_keras.h5` as well as a model file converted into tensorflow estimator for GAN training.

## GANMEX Training

Run `examples/stargan_estimator/train.py`. Indicate the path to the classifier files (including the converted estimator file) in the `cls_model` arguement. If `cls_model` is set to `None`, then a new class discriminator will be trained from scratch.



## Reference
Please refer to our paper for more detailed explanations of the methodologies:

Sheng-Min Shih, Pin-Ju Tien, Zohar Karnin. [GANMEX: One-vs-One Attributions using GAN-based Model Explainability](https://arxiv.org/abs/2011.06015)
 
----------------------------------
## TensorFlow-GAN (TF-GAN) original README.md

TF-GAN is a lightweight library for training and evaluating [Generative
Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661).


* Can be installed with `pip` using `pip install tensorflow-gan`, and used
with `import tensorflow_gan as tfgan`
* [Well-tested examples](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/examples/)
* [Interactive introduction to TF-GAN](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/colab_notebooks/tfgan_tutorial.ipynb) in colaboratory

## Structure of the TF-GAN Library

TF-GAN is composed of several parts, which are designed to exist independently:

*   [Core](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/python/train.py):
    the main infrastructure needed to train a GAN. Set up training with
    any combination of TF-GAN library calls, custom-code, native TF code, and other frameworks
*   [Features](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/python/features/):
    common GAN operations and
    normalization techniques, such as instance normalization and conditioning.
*   [Losses](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/python/losses/):
    losses and
    penalties, such as the Wasserstein loss, gradient penalty, mutual
    information penalty, etc.
*   [Evaluation](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/python/eval/):
    standard GAN evaluation metrics.
    Use `Inception Score`, `Frechet Distance`, or `Kernel Distance` with a
    pretrained Inception network to evaluate your unconditional generative
    model. You can also use your own pretrained classifier for more specific
    performance numbers, or use other methods for evaluating conditional
    generative models.
*   [Examples](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/):
    simple examples on how to use TF-GAN, and more complicated state-of-the-art examples

## Who uses TF-GAN?

Numerous projects inside Google. The following are some published papers that use TF-GAN:

*   [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
*   [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
*   [GANSynth: Adversarial Neural Audio Synthesis](https://arxiv.org/abs/1902.08710)
*   [Boundless: Generative Adversarial Networks for Image Extension](http://arxiv.org/abs/1908.07007)
*   [NetGAN: Generating Graphs via Random Walks](https://arxiv.org/abs/1803.00816)
*   [Discriminator rejection sampling](https://arxiv.org/abs/1810.06758)
*   [Generative Models for Effective ML on Private, Decentralized Datasets](https://arxiv.org/pdf/1911.06679.pdf)

The framework [Compare GAN](https://github.com/google/compare_gan) uses TF-GAN,
especially the evaluation metrics. [Their papers](https://github.com/google/compare_gan#compare-gan)
use TF-GAN to ensure consistent and comparable evaluation metrics.
Some of those papers are:

*   [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)
*   [The GAN Landscape: Losses, Architectures, Regularization, and Normalization](https://arxiv.org/abs/1807.04720)
*   [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035)
*   [High-Fidelity Image Generation With Fewer Labels](https://arxiv.org/abs/1903.02271)

## Training a GAN model

Training in TF-GAN typically consists of the following steps:

1. Specify the input to your networks.
1. Set up your generator and discriminator using a `GANModel`.
1. Specify your loss using a `GANLoss`.
1. Create your train ops using a `GANTrainOps`.
1. Run your train ops.

At each stage, you can either use TF-GAN's convenience functions, or you can
perform the step manually for fine-grained control.

There are various types of GAN setup. For instance, you can train a generator
to sample unconditionally from a learned distribution, or you can condition on
extra information such as a class label. TF-GAN is compatible with many setups,
and we demonstrate in the well-tested [examples directory](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/examples/)


## Maintainers

* (Documentation) David Westbrook, westbrook@google.com
* Joel Shor, joelshor@google.com, [github](https://github.com/joel-shor)
* Aaron Sarna, sarna@google.com, [github](https://github.com/aaronsarna)
* Yoel Drori, dyoel@google.com, [github](https://github.com/yoeldr)

## Authors
* Joel Shor, joelshor@google.com, [github](https://github.com/joel-shor)
