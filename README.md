# Autoencoder using Pytorch

## I implemented an Autoencoder for understanding the relationship of the different movie styles and what can we recommend to a person who liked a set of movies.


A autoencoder is a neural network that has three layers: an input layer, a hidden (encoding) layer, and a decoding layer. The network is trained to reconstruct its inputs, which forces the hidden layer to try to learn good representations of the inputs.

An autoencoder neural network is an unsupervised Machine learning algorithm that applies backpropagation, setting the target values to be equal to the inputs. An autoencoder is trained to attempt to copy its input to its output. Internally, it has a hidden layer that describes a code used to represent the input.


The autoencoder tries to learn a function hW,b(x)≈xhW,b(x)≈x. In other words, it is trying to learn an approximation to the identity function, so as to output x̂ x^ that is similar to xx.

Autoencoders belong to the neural network family, but they are also closely related to PCA (principal components analysis).

![down](https://cdn-images-1.medium.com/max/1600/1*wr9QeopG3BK4Lz6DGhlqbA.png)

## Some Key Facts about the autoencoder:

It is an unsupervised ML algorithm similar to PCA
It minimizes the same objective function as PCA
It is a neural network
The neural network’s target output is its input
Autoencoders although is quite similar to PCA but its Autoencoders are much more flexible than PCA. Autoencoders can represent both liners and non-linar transformation in encoding but PCA can only perform linear transformation. Autoencoders can be layered to form deep learning network due to it’s Network representation.

## Types of Autoencoders :

1. Denoising autoencoder

2. Sparse autoencoder

3. Variational autoencoder (VAE)

4. Contractive autoencoder (CAE)
