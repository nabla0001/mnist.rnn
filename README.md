# Generating MNIST with a Recurrent Neural Network

A fun little project to learn how to train Recurrent Neural Networks (RNN) in PyTorch.

## The idea

* **step 1**: train an RNN on images of handwritten digits (MNIST) to predict the next pixel
* **step 2**: use the model to complete images of partially masked, unseen digits

## Results

After just a two or three epochs the model starts completing many digits sensibly:
it knows where the digits need to continue; it also frequently infers correctly
which digit it needs to draw based on the occluded digit - something that is not always obvious.
It then gets their overall gestalt right. 

Quite cool considering that, unlike when we train classification models, we have never told the model explicitly that the digits 0-9 exist!

Here are some examples. 

* *left*: the occluded pixel sequence given to the model
* *center*: the completion by the model
* *right*: the ground truth image (digit in brackets)

![example](plots/3480.gif)
![example](plots/0952.gif)
![example](plots/4866.gif)

However, there are also still many failure cases:

![example 0](plots/0216.gif)
![example 0](plots/1809.gif)

# failure cases

## Details

*(WIP)*