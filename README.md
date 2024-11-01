# Generating MNIST with a Recurrent Neural Network

A fun little project to learn how to train Recurrent Neural Networks (RNN) in PyTorch.

## The idea

Many of us have classified MNIST images before, but let's try something different to explore the magic of RNNs.

First, let's flatten the 28x28 images of handwritten digits into sequences of their pixel values.

In phase 1 (*training*), we'll train an RNN to predict 
a single next pixel given a sequence of pixels. 

In phase 2 (*testing*), we'll then ask: can this RNN generate realistic-looking
MNIST images? To test this we'll present the model with partially masked MNIST digits
and ask it to complete it.

## Results

After just a two or three epochs the model starts completing many digits sensibly:
it knows where the digits need to continue; it also frequently infers correctly
which digit it needs to draw based on the occluded digit - something that is not always obvious.
It then gets their overall gestalt right. 

Quite cool considering that:
1. the model has never seen 2D images, only pixel sequences
2. we have never told the model explicitly that the digits 0-9 exist (unlike in a classification setting)

Here are some examples. 

* *left*: the masked pixel sequence given to the model (500 pixels)
* *center*: the completion by the model (284 pixels)
* *right*: the ground truth image (digit in brackets)

![example](plots/3480.gif)
![example](plots/0952.gif)
![example](plots/4866.gif)

However, there are also still many failure cases:

![example 0](plots/0216.gif)
![example 0](plots/1809.gif)
![example 0](plots/2653.gif)

## Details

*(WIP)*

## Usage

You can re-create my `conda` environment via

```shell
conda env create -f env.yml
```

To run experiments


```shell
conda activate pytorch

python train.py --exp-name gru-128 --hidden 128
```
see `train.py` for all available command line options.

Each experiment produces two files: 
1. a results file (`.pkl`) containing training/test metrics
2. a model checkpoint (`.ckpt`) 

which are written to a subfolder `{exp_name}` in `experiments` (configurable via command line).

