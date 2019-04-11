# Lesson 11

## Next week we'll learn how to develop a complete new module fastai.audio
- Canceled
- Will be continued convering from last week

## Data loading and optimizers

## Today we're going to start to move forom a minimal training loop to something that is SoTA on Imagenet
- Last week we talked about batchNorm
- SImplified RunningBatchNorm
  - We can remove debiasing

- (off) Batchnorm: normalize the inputs to nonlinearities in every hidden layer.

## Notebook 7
- The most fastaish 
- All you need is a good init paper
- He came up with a technique called LSUV
- Little things can change the variance of the layer outputs
- If it is different than one gradients will explode
- His idea is let the computer figure it out
- Creating a bunch of layers
- Create convolution layer with relu
- bias defined es property
- Create a learner in the usual way
- Train it
- Get a single minibatch
- Find modules that are of the kind we want
- Of type convlayer
- When workigng with Pytorch we use srecursion a lot
- Create a Hook, that gets the mean of std
- we see that are not zero and one (mean and std)
- Rather than come witha  perfect init
- create a loop
  - check if the mean is close to zero, if it is not, we subtract the bias with the mean
  - same thing with with weights
  - with those loops, it will eventyally go to what we want
  - That's it
- Particular important for deeper NN
- Fast AI way to initialize weights with no math

## Notebook 8: https://colab.research.google.com/drive/1CsNbd7Fv89tYkOYruXTRniIQv-AVHSyC
### Data block API foundations
- not using imagenet because of time
- small images behave differently compared to big images
- Gap 
- If we use 128 by 128 discoveries apply to full 
  - but still takes time
- Tried creating new datasets
  - Subsets of imagenet
  - Contain 10 classes
  - number of versions with different sizes
  - one of them is easy
  - different classes from each other
- Harder dataset with similar with each other
- Imagenette
- Second is Imagewoof
- Learderboard
- With quick experiments results some things are wifferent and some are similar between both datasets
- fastai/imagenette on github: https://github.com/fastai/imagenette
- Your own version of toy problem
- Try to beat hiim
  - 90% on mini dog breed classifier
- So imagenet is too big
- We wanto some way to do that on scratch
- We'll build a data block API
- We'll be able to either create our own
- path.ls = ... 
  - a way to add arguments to a class (lazy way)
- impot load up an image
- 150 per 213 pixels with 3 channels
- setify function 
- Now we need a way to go trhou a single directory and get the files
- function get files, with a path and fs as list of files
  - Check if not in list of extensions
  - filter out non image files
- Put everyting togeher
- 72ms to get 13k filenames
- Imagenet is 100 bigger than imagenette, so we need to be fast

### Prepare for modeling
- Fast ai has a datablock api
- because he had to create one api for each type of strucure of files
- We need to do
  - Get files
  - split validation set
  - label
  - transform per image
  - transform to tensor
  - dataloader
  - transform per bach
  - data bunch
  - add test set

- In the end what we want is an imagelist
- Superclass with itemlist
- compose im python
  - go through a list of function and create a composed function
- Training with 12k and validation with 500k

### Labeling
- Has to be done after splitting because it uses training set information to apply to the validation set usinga processor
- Processor is a transformation applied to all the inputs once at initialization
- Create vocab from the training set and apply to validation se
- Function uniquefy
  - get unique values of something
- CategoryProcessor
  - Create vocab if in training set

### Modeling
- Batchsize
- Class DataBunch
- Channel in, chanel out with adjust of number of inputs outputs in c_in c_out

- Path
- get the transforms
- get the list
- label it
- turn into a databunch

### Model
- Define callack
- Create a convnet with 64 64 128 128 layers
  - Paper Back of Tricks for Image Classification with CNNs
- Validation loss is 72%

## Notebook 09 - Optimizer tweaks
- Massive improvement with natural language processing by using optimizer tweaks
- First library to have this implemented
- 8 lines of code implement the data
- Continue the basics 
- Model defined
- Callback defined


### Let's create an optimizer

- Colab notebook: https://colab.research.google.com/drive/1DeNgC-SlriiZ0NkKfTpWfFLcjIdRtUOj






