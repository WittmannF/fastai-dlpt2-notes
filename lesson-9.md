# How to train your model

## Revisit questions from the last week
- Research about the division from sqrt(5) in nn.comv2d
- Def get data
- def normalize
- torch.nn.modules
- x_train was resized to 28 by 28 (5000)
- a convd layer was created with 32 hidden layers 
- Validation set get 100 samples
- Shape is 100 x 1 28 x 28
- def stats return x(mean and standard deviation
- returns the mean and the standard deviation of a tensor
- layer 1 has has 32 x 1 x 5 x 5 (4D)
- Why is the size of the layer this way
    - Excel spreadsheet 
        - Filter for each input channel and for each output channel
        - for the next layer theres a four dimensional 3 by 3
- rconv2d.eset_parameters??
    - a = math.sqrt(5)
        - T receives the input tensor l1(x)
        - get stats
        - We'd like to have a mean of zero and variance of 1
            - variance is still far
        - using kaiming initialization, classic one
        - Recall that a lealky relu layer has a gradient of alfa in the negative axis. 
        - kaimin with a = 1 means linear layer. 
        
- def f1(x, a) F.leaky_relu(l
- init kaiming has variance of 1- however the variance in conv2d is very bad
- Try to write my own kaiming init function
    - work with regular fully connected matrix mltiplication
    - baxically the number of weighs 
    - we multiply all the filters to and then add them up
    - also the channel
    In order to calculate the total number of operations going on
        - we multiply the number of filter by the size of the filter 5 by 5
        - thats the recepts filter size
            - how many elements in that kernel (25)
            - shape of the weight matrix
            - the number of filters in and out
            - for the kaiming we calculate tha fan in and fan out
            - For the leaky relu we have to multiply by root two with a leaky a  
                - gain is sqrt(2/(2+a**2))
                - gain for a = 1 is 1, for 0 is 1.4142. 
                - In the case of root 5 is 0.5777, unexpected and a bit concerning
            - the inicialization that we use is kaiming uniform
                - normaliz distributed normally numers look a normal shape
                - uniform numbers look like squared distribution
                - -1 to 1
                - the std is not 1
                    - that's why sqrt(5)
                - persolalized kaiming2
                    - variance near to 1
                    - with a equal to sqrt of 5 it is the same value
- Train a convnet to check
- comparing the default init
  - variance of 006
- grab the predition and run the backward (what we did in the last week)
  - get stats and again variance is not 1
- checking kaiming uniform code
- go throu eachch layer and if it is a convolutional layer then change the initialization
  - it is not 1 but it is better than before
  - in the backward the variance is 0.5
  - this is concerning

## Why you need a good init
- x receives 512 random values
- a receives a matrix 215 by 512
- for in in range(100) x=a@x
- x.mean() and x.std were nan
- the problem is activation explosion. Very soon the activations go to nan
- no changing to 0.01
  - then it tends to zero after the multiplications
  - it cant learn anything
- that's why for decades pepople weren't able to train it
- activations vanished to zero

## The magic number for scaling

## Good bugh
- math.sqrt(5) is a bug
- BUt is it a "good bug"?
- In a couple of hours pytoch created an PR to update weight initialisations to current best practices 
  - Lesson: Do not trust in the standards.

# There are lots of interesting initializations approaches
- understand the difficulty of deep 
- delfing deep into rectifiers
- all you need is agood init
- exact solutions to the non linear dynamic (orthogonal i
- fixup initializaton
- nelf-normalizing neural networks
  - ho to set a combination of activation ad initialization to guarantee the initialization 
    - in both cases (with fixup) they xx 
    - selu if you put a dropout you need a ccorrection
    - difficult to train
    - relies on two different numbers
    - full of math, and different network
    - why nobody want to read me
- all you need is a good init is a better one
- From last week the shape of the wegiths and from pytroch were inverted
  - this come from 7 years 
    - couldnt handle the matrix multiplication and that's why did this way
    - always many thngs are done fowever and nobody asks why
- have NN well initialized is important
  - Hopefully this new approach will right things
  
## We are at Train loop
- last week Matmul, Relu/init, fully connect forward anf fully connected backward
- File 03_minibatch
- Get data
- Get some predictions from that model
- We need a better loss function
- We used MSE before for simplicity
- Cross entropy loss
  - reminder from excel
  - requires softmax
    - average of exponentials
  - softmax in code form
    - x.exp/x.exp().sum().log
    - apply a log in the end
  - is expressed in the form sum of x times log prob of value
    - actuals are 1 or zero
    - get predictions
    - whats the log of those probablities
    - compute equation
      - rather than multipling by zero is simply say the location of the one using a index
      - then lookup for the 
      - then the equation changes to -log(p_i) which i is the index of the prediction
        - how to write on pytorch
        - cool trick
        - lets look at three values of y train
          - 5, 0 and 4
          - we ant to find the probability associate with 5 zero and 4
          - sm_pred.shape
          - is 50000 by ten
          - sm_pred is -2.49
          - we can index into an array in sm_pred
          - this works becayse pytorch supports the advanced indexing from numpy
          - you pass a list for each dimension
            - two in this case
            - the first is the list of row indexes
            - second is the columns indexes
          - rang(target.shape of 0, target)
          - that returns the values that we need
          - take the mean
          - the loss is the negative ll of smm compared to y_train
- Note that the formula log(a/b) = log(a)-log(b)
- gives a simplification when we compute the log softmax which was defined a x.exp
- def log_softmax x - x.exp.sum.log
- test_near(lnn(log_softmax)
- We are taking the log of the sum of the exp
- there's a trix called logsumexp
  - very big numbers are highly innacurate
  - we want to avoid big numbers
  - we don't want exp to be beg
  - by doing a mat substitution we ccan subtract a value and sum and get the same number
  - we add outside
- in other workds logsumexp
  - get the max number of x
  - return m (x-m)exp.sum.log
- check that logsumexp is the same of pred.logsumexp)
  - actually a method in pytroch
  - we can use in pytorch
- test(near(F.nll)
- test_near(F.cross_entropy, loss)
  - now we can use it 

## Basic training loop
- accuracy is the agrmax of out == yb.floar.mean
- batch size of 64
- for x batch xtrain from zero to batch size
- calculate predictions
- let's compute them
- now we cam grab
- calculate the loss
- 2.3
- accuracy very bad
- define learning rate
- number of epochs
- training loop
  - from part 1 in lesson 2 sgd looks like update of calculating predictions, loss backward (dl1, lesson2)
- iterate over epochs
  - iterate over batches
    - setarti
    - end i
    - x_batch
    - ybatch
    - loss.backward()
      - for l in model.layers
        - l.weigth
        - ..
        ?? You didnt compute the derivative of the loss function
- define class dummy module
  - self.module receive an empty dict
  - written in pure python 
  - if not private, put values into modules dictionary
  - then du what the superclass._sett atrribute
- some refacory
- pytorch has the same thing
  - inhert from nn.Module
  - that's why you have to call super()__init
  - now create something 
- that's how print out model, it is a normal class with this behavior
- self add modulefor printing
- pytorch modulelist calls the addmodule
- sequentialmodel
- when calling 
- model = nn.Sequential(nn.Linear, nn.Relu
- ...
- assert acc>0.7

## Dataset and DataLoader

- Define calss dataset
- init sith x and y
- len has length of x
- get item will return index of x and y
- xb, train gets batch

## DataLoader
- DataLoader is initialized with dataset and batch size
  - in each iteration yield self.ds[i:i+self.bs]
- xb and yb gets next(iter(valid_dl))
  - will do this a lot
- assert xb.shape == bs, 28*28
- assert yb.shape
- now our fitness function
- def fit()
  - iterate over batches
    - iterate over batches
      - get predictions
      - get the loss
      - do the backward
      - do a step
      - zero_grad

## Random sampling
- class sampler
  - initialize
  - shuffling
  - get random permutation from torch
  - go through that sample and ield indexes
- now we replace a dataloader with a sampler
- __iter__ uses a sampler
  - actually looping through something
  - streaming computations
  - grab all the indexes and te dataset at the index and collate them
    - function for this purpose
    - stacks them up
    - pass as a different function
- we create a different function
- and train data loader 
- it's been shuffled
## PyTorch DataLoader
- Pytorch dataloader does the same 
- works in the same way
- you can just pass in shuffle
- nothe that in DataLoader we can pass the number of works to use multiple threads to call the dataset

## Validation
- You always should have a validation set to know if you have overfitting
- torch.nograd
- going throu the validation set
- just keep track of the loss but dont compute the gradients
- model . train
- model.eval
- Changing from model . training mode and eval mode is ipmportant
  - deactivatie dropout for example
- one minibatch of size 100 and one of 1
  - incorrect
  - does not work well when the batch size varies
  - data loader will be shuffled
- fit and do five epochs
- accuracy of 0.968
- Class Break

## An infinitelly customazible training loop
- current fit function
    for each epoch
        for each batch
            predict
            loss
            backward
            step
- but there are multiple tweaks that we want to add
    - keeping track of losses and metrics
    - now we have tensorboard integration
    - mixed precision training
    - or more complex trainings like GANs
- so either have to rewrite a few loop for each new kind of training
- or try to write something that incorporates eerything that you can think of
- fortunately there is a way around this: callbacks
    - let you not only look up but also sutomize every etep
    - those updates can be new vaulues or flags that skip steps or stop the training
- then eacch tweak of the trianing loop can be entirely written in its own callback 
    And then you can mi and match those blocks together
- case study:
    - GAN was defined as a GAN module
    - check if it is a generative or critic 
    - generator loss and critic loss
    - GAN trainir would switch each
    - and then would set requires_grad as appropriate
    - and would define the generator mode
    - very few codes
    
## DataBunch/Learner
- benefits of packaging things together
- Factor out the connected pieces of info out of the fit augmented list
- lets replace it with something that looks like this
    - fit(1, learn)
- This allow us to tweak whats happening inside the training loop in other places of the code because the larner object will be mutable so changing its attributes elsewhere will be seen in our training looop
- so that's our databunch class
- data model
- return model and optimizer
- everything stored in Learner class
    - no logic, just storing the parameters
- learn = Learner(*get_model(data), loss_func, data)*

## CallbackHandler
- def one_batch
    - fit loop has epochs and learner
- class Callback()
- Callbackhandler
    - keep track of all the callbacks
- Example of TestCallback
    - begin_fit
    - after_step(self)
- fit(1, learn)
    - it did 10 batches

## Runner
- future version of fast ai
- simpler fit
- factoring into Oriented Objects
- Callback
    - has a name property
    - if you have a name traineval
    and call calmel2snake(cnmame)
- we remove ccallback in the name and then this is its name
- it actually assigns attribute into a name. 
- Now we have a runner 
- Let's use to add metrics
- Class averageStatsCallback
    - defines epoch
    - after epoch
    - after loss
    - different batch sizes was fixed
- run = Runner(cbs=stats)
- run.fit.
- the main thing is def(fit, epochs,learn)
    - should recognize this
    - calls each at each time
 
 ## One cicle training
 
 ### Annealing
 - We'll create a callback to make a hiperparameter scheduler
    - Heple have been pointin out that we can schedule...
    - As you train a model, goes through differnet fase, lost function landscapes, look very different in the start in the begining, medion and end
        - schedule is super handy
- Class ParamScheduler(Callback):
    - def set_param
        - iterate over parameter groups

- We need a function that just takes position
    We do partial
- Decorator is a function that returns a function
- It is not possible to plot tensors
    - Solution: add property ndim in tensor
- Create another function called combine_schedulers
- Nice gentle warmup in the beginning, increases LR and 
    - last 4 monts you need to rain a high lr for a long time
    - but also you have to keep in a smal learning rate for a while

- All the pieces that we need for trying out
- Hopefully many things to try out
- Next week Conv Nets
- WE will start using GPU
- Callback 
!! Try some examples simple implementations using callbacks
- manual 
- see inside models
- find ways to train more nicely 
- and cover until notebook 10
