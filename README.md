# Human-Faces-with-WGAN-GP

In this repository I generate Human Face images with Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP).

My implementation uses Python 3.7, TensorFlow 2.0, Numpy and Matplotlib.

## Dataset

The dataset was downloaded from https://www.kaggle.com/jessicali9530/celeba-dataset which consists of 202,599 face images of various celebrities. Due to a limited resources, we train using 131,072 images which were preprocessed and resized to 64x64. Here are sample images from the dataset

![alt text](https://github.com/yernat-assylbekov/Human-Faces-with-WGAN-GP/blob/master/images/images_from_train_set.png?raw=true)

## Network Architecture

For the generator and the critic, I use the architectures suggested in the DCGAN (Deep Convolutional GAN) paper 
<a href="https://arxiv.org/pdf/1511.06434.pdf">[3]</a> with slight modifications following the guidlines listed below:<br>
• Replace all max pooling with convolutional strides.<br>
• Use transposed convolution for upsampling.<br>
• Use batchnorm in both the generator and the critic.<br>
• Use LeakyReLU activation in the generator for all layers except for the output layer.<br>
• Use LeakyReLU activation in the critic for all layers except for the flattening layer and the output layer.<br>
• The output layer of the generator uses the sigmoid activation.

The precise architectures for the generator and the discriminator are as shown below.<br>
### The generator:<br>
![alt text](https://github.com/yernat-assylbekov/Human-Faces-with-WGAN-GP/blob/master/images/generator_diagram.png?raw=true)<br>
### The critic:<br>
![alt text](https://github.com/yernat-assylbekov/Human-Faces-with-WGAN-GP/blob/master/images/critic_diagram.png?raw=true)

## Training Details

I used the same loss functions for the generator and the discriminator as in the WGAN-GP paper <a href="https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf">[2]</a> by Gulrajani et al. We use the Adam optimizer with `learning_rate = 0.0002`, `beta_1 = 0.5` and `beta_2=0.9`. I trained the model with an NVIDIA K80 GPU.

## Results

Below is the full training as a GIF for 100 epochs with images sampled after every epoch.

![alt text](https://github.com/yernat-assylbekov/Human-Faces-with-WGAN-GP/blob/master/images/faces_generated.gif?raw=true)

Images generated after 100 epochs look as follows:

![alt text](https://github.com/yernat-assylbekov/Human-Faces-with-WGAN-GP/blob/master/images/image_at_epoch_100.png?raw=true)

I expect that the results will be improved if one consider deeper generators.

## References

<a href="http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf">[1]</a> M. Arjovsky, S. Chintala and L. Bottou, <i>Wasserstein Generative Adversarial Networks</i>, PMLR (2017).

<a href="https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf">[2]</a> I. Gulrajani1, F. Ahmed, M. Arjovsky, V. Dumoulin and A. Courville, <i>Improved Training of Wasserstein GANs
</i>, NIPS Proceedings (2017).

<a href="https://arxiv.org/pdf/1511.06434.pdf">[3]</a> A. Radford, L. Metz, S. Chintala, <i>Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</i>, ICLR (2016).
