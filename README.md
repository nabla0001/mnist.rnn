# Generating MNIST with a Recurrent Neural Network

A fun little project to learn how to train Recurrent Neural Networks (RNN) in PyTorch.

## The idea

Many of us have classified MNIST images before, but let's try something different to explore the magic of RNNs.

First, let's flatten the 28x28 images of handwritten digits into sequences of pixel values.

In phase 1 (*training*), we'll train an RNN to predict 
a single next pixel given a sequence of pixels. In phase 2 (*testing*), we'll then ask: can this RNN generate realistic-looking
MNIST images? To test this we'll present the model with partially masked, unseen MNIST digits
and ask it to complete them.

## Experiment

Here are the details of the first experiment:

### Architecture 

* Gated Recurrent Unit (GRU)
* 1 layer with 128 units
* sigmoid output

### Training hyperparameters

* Pixel values are binarised to be either {0,1}
* Trained via **teacher forcing**, i.e. at each time step *t* the ground truth input is presented to the model, not its output from step *t-1*
* Loss: `MSE`
* Optimiser: `Adam`

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

![example](plots/gru-128x2/ep20/8-1141.gif)
![example](plots/gru-128x2/ep20/0-1924.gif)
![example](plots/gru-128x2/ep20/6-4763.gif)
![example](plots/gru-128x2/ep20/9-0371.gif)

However, there are also still failure cases.

E.g. the model sometimes randomly adds bottoms of 8s like for this 4.

![example 0](plots/gru-128x2/ep20/4-2147.gif)

It's also not very good at the left-swinging bottoms of 3s and 5s and typically
just completes them downward.

![example 0](plots/gru-128x2/ep20/3-0687.gif)
![example 0](plots/gru-128x2/ep20/5-3577.gif)


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

