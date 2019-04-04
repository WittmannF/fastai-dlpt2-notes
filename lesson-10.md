# Lesson 10
- (off topic) Jeremy uses software called OBS for recording classes.
    - Streaming key from youtube is linked
- well be using pytorch-ninghtly
- Sharing unpublished research

## Wrapping up our CNN
- Remember, this is meatnt to take a while, don't worry
  - try to make us busy until next year
  - don't feel like you have to understand everything
  - covering more software engineering
  - DS need to be also software engineers

### What do we mean by from the foundations?
- Recreate fastai
- using python, non ds modules, pytorch, 
- but well make it better

### Steps to a basic modern CNN model
- matmul > ..
### Today were going to start move from a minimal training loop to something that is SoTA on Imagenet
- Cuda, convolutions, hooks, normalization, transforms, weight decay, label smoothin, Optimization, skip connection architectures
- or at least pretty close
- google skipped using LAMB
- Paper? 
- LAMB is a general optimizer that works for both small and large batch

### Next week well learn how to develop a complete new module fastai.audio
- module creation, jupyter docs, writing tests, complex numbers, fourier transforms, audio formats, spectrograms, nonimage transforms, gated CNNs

### Well also learn about sequence to sequence models with attention
- neural translation
seq2seq

### Well wrap up python adventures with a dive din to u nets, plos some more vision applications
- build your own dl box
- fasttec 2
- U-net deev dipe
- Pixel shuffle
- Self-attention
- Learnable blurring
- Generative video models
- Devise
- CycleGan
- Object detection

### Notebook 05a_foundations.ipynb
- we'll be looking at callbacks
- Important for fastai
- quickly adjust
- example
- function that prints hi
- create button with widgets
- w is a button 'click me'
- pass a function to the framework to run it when you click the button
- on click method
- w.on_click(f)
- now prints hi
- f is a callback
- it is not a particular class
- it is a concept
- it is a function that we treat as a object
- we are not calling a function
- that's our start point
- these widgets are really worth looking at
- example of plotlib documentation
- experiment with different types of funcitons
- very easy
- image labeling was built with widgets

#### Creating your own callback
- function slow calculation
- square of index with delay of 1 second
- funciton takes 5 seconds to run
- how to get the progress?
- pass a callback as parameter in the function
- function showprogress with parameter ephoch
- slow calculation with function as parameter to show progress
- this is callback
- we'll start looking at more callbacks

#### Lambdas and partials
- we can use lambda notation as well
- allows to define the funciton as parameter
- we cant pass slowcalculation if two arguments is not possible
- convert to only one argument
- lambdas allows for this
- indicate different a function that returns a persolanized exclamation
- a bit akwward, better with def _inner
- function inside a function
- make_show_progress (exlamation)
- return inner
- calls an inside function
- always create a new function with a new exlamation
- that's called a closure
- f2 receives function with exclamation
- use input as f2 also works
- partial function aplications
- call partial returns a new function that a parameter is given
- slow_calculation(partial(showprogress, 'ok I guess"))
- a variable can be stored with this information

#### Callbacks as callable classes
- class progress showing callback
- store into init and call
- cb = progresswhosingcallback('just super')
- treats objects as a function
- use the class as a callback

#### Multiple functs, `*args and **kwargs`

- reasons to use it
- to wrap other classes into it
- in earlier days they were over used in fastai
- f(3, 'a', thing='hello')
- args as tuples kwargs as dictionarries
- using in slow_calculation
- allow a callback cb.before_calc and after_calc
- after_calc takes two inputs parameters
- we cant just call cb

- class print step callback
- init pass
- functions before and after_calc with `*args and **args`
- we'd get error if not using
- use epoch and value to print the details
- `**kwargs` for optional argumetns

#### Modifying behavior
- def_slow calculation with stopping early
- in order to stop early check if has attribute after_calc
- stop if matches condition
- we can also the way the calculations are being done
- take calc into a class
- now the value to be calculated is an atribute
- we can reach the result and modify values calling calc.res
- taking advantage of this callback system
- pass the calculated object as a callback
- define a callback as a function
- return cb if cb
- last week we called inside __call__
- allowed doing self(...)
- remove extranoise but less informative


#### __dunder__ thingies
- is special somehow
- most languages allows to define special behavior
- special magic names
- in python all magic names look with dunder
- python docs has a data model reference
- tell about those special methods
- list of suggestions
    - getitem
    - getattr
    - repr
    - add
    - new

- for example we can creat two adders

#### Browsing source code
- jump to tag/symbol by with (with completions)
- jumpy to current tag
- jump to library tags
-go back
- search
- outlining/folding

- We'll see in vim how to do that
- most editor allows this

#### Using VIM
- using in terminal, which allows to work remotely
- jump to a symbol
- class, function or something like that
- jump straight to the definition of creatcnn
- `:tag create_` and tab coto autocomplete
- ... to info? right square brackets? 
- outlining or folding
- go back to where you were with `ctrl + t`
- easy to jump around
- also helpful to jump to source code of libraries
- additional tags of where you want to see
- more general searchers with `Ack lambda` and enter
    - get a list of places using lambda

#### Variance and stufs
- refesher
- how far away each data point is from the mean
- create tensor with 1,2,4,18
- m get mean
- tensor minus mean and its mean is zero
- fixing getting the squared of the difference or abs
- abs and pow show how far they are from the mean
- pow is std defined as sqrt of variance
- abs is mean absolute deviation
- 18 is an outlier
- std is more sensitive to outliers than mean absolute deviation
- very often well using mean absolute deviation
- math people tends to use std because math proofs are easier
- there's lot of places were we would want to use mean absolute
- and heres something useful
- $E[X^2] = E[X]^2$
- in the second we only have to keep track of two numbers
- it is easier to work with
- the definition of variance that we'll be using is usually the second
- difference between each t and its mean times each u and its mean
- comparing with a different dataset with random numbers
- the result was smaller
- lined nicelly values the values in x and y are higher due to linear correlation
- the number told how those numbers varies in some way
- it is called covariancee
- $E[XY] = E[X]*[Y]$
- from now on you are not allowed to look at an equation without typing it in python and calculating some values
- Pearson correlation
- normalized

### Excel entropy_example.xmlx
- Review softmax equation
- cross entropy as -logp
- clarify things that researchres get wrong
- when to use and not use softmax
- multiclass example
- various outputs
- get exponential of outputs
- get softmax
- different images with diffent activations
- but softmax is different
- it happens because in all cases had same ratio
- second image seems there's nothing because all activations have low values
- end up picking fish because has to pick something
- exponantial picks higher to add up one
- however, guess image 2 has nothing
- maybe the problem actually is image 1 had multiple classes
- it is not true that both
- softmax is a terrible idea
- unless only one, no more than one, and at least one example
- if does not have any of those, it still tell you that one of them exists
- what you do if there could be more things or no things?
- **Use binomial**
- image value divided by value plus 1
- allows for multiclass with multiple classes simuntaneously
- why we also use? becuse of imagenet with only one class
- an alternative if none of classe
- missing, background, if none classes
- many researchers tried that
- but it is a terribae idea
- predict missing
- features have to combine other classes negativelly
- there's no set of features to define them
- create a negative model of every single type
- very hard as well
- creating a binomial is easy and better results
- many academic papers make this mistakes
- if we cross with some academic with softmax
- try replicating with binomial
- for example language modeling
- next word? definitelly one words, not more than 1 word

### Notebook 
- Let's build learning rate finder
- Using exceptions for controw flow of statements
- Helpful for writing code
- get mnist 
- define callback class
- in `__call__` it was refactored inside callback class instead of runner
- users can create own callbacks
- print out every callback name
- or create own callback with something
- the keything is three types of exceptiong
- class that ineherst an exceptiong with pass as output
- has the same attributes of Exception to a new class name
- Let people to cancel anything on trin, epoch or batch 
- how it works?
- now fit has try with except `CalcelTrainException` and call after_cancel
- **No errors occours**
- Test Call Back will raixe cancelTrainException to stop when criteria occours
- `CancelEpochException` in zall_batches`
- `CancelBatchException` inside `one_batch`
- Allows anyone to stop any of those things to happen
- We'll use this in LR_Find
- Set an exponential curve to set
- and then after each steps checks if loss is much worse than the best we had so far or max iteration
    - Raise CancelTrainException
- Next version of FastAi will use it
- now we can create a runner
- put lrfind
- does less than 100 epochs because loss got worse
- **Now we have LRFinder**
- Let's create a CNN!

### Notebook 06_cuda
- get mnist
- normalize training set
- check mean
- creat a data bunch
- create a cnn model
- sequential model
- with many stride 2 convolutions
- relus
- padding 2  and 1
- Lambda fatten)
- x.view(-1,1,28,28) to resize mnist
- include function into nn.sequential
- pytorch does not has
- we create a layer called lambda with nn.module)
- we pass function and in foward calls that function
- and lambda fatten removes x.view for flattening
- we gat call back 
- frunner
- 5 seconds with one epoch results
- now gets slower
- lets use GPU!

#### CUDA
- this took a long time to run, so its time to use a GPU
- a simple callbacks can make sure the model, inputs and targests are on the same device
- define cuda callback
- defice
- on begin fitting model to device
- it moves osmethignwith parameters to a device
- create a device with roch.device('cuda' and number of GPUS)
- in begin batch check runner
- on begin batch we call self.xb - xb
- well changer using begin batch to move xb to device using sel.xb.to(device)
- it is flexible
- mabye it is easier with torch.cuda.set_device(device)
- class cuda callback
- on begin fit call mode.cua
- now 3 epochs on 5 secos
- definitelly faster

#### Refactor model
- we can regroub a fucntion conv2d with conv and relu
- has sequential 
- the model we cant use for anything except mnist_resize
- replace with a callback
- batchTransformCallback(callback)
- pass some transormationf ucntion
- on begin back stransmit self.tfm
- and append partial of batchTransformXCallback with mnist_view
- uses trick __iner
- creates a new view function with desired size
- great a picece of code to study
- using this approach we now have mnist view resizing as a callback
- now we create a generic getcnnmodel
- with arbiratry set of filtesr and numbers of layres
- get_cnn layresr
- lastfewlayresr has average opoling, lambda and nnlinear
- conv2d with kernel size
- whats the kernel size?
- 5 if first or 3 if latter.
- Why?
- our image had a single channel and we are using 3 by 3 filters
- as it scrows in the image only looing to a 3 by 3 windows
- in total there will be 9 input activations
- split those into dot product with 8 by 9 elements
- out of that will come a vector of length 8
- because we have 8 filters
- that's seems pointless
- we started with 9 numbers and end up with 8 nubmers
- there's no ont pmaking first laeyrs shuffing first layers into different values
- different models witll have 3 by 3 by 3 channels which is 27
- but still will be fitting and lowing information
- usually first layer 7 by 7
- for our default well make 5 by 5 for first layer
- and 3 for the following ones
- with 8 filters in the first
- function get_runner with model, data, lr, cbs
- everything was built by scratch
- get_cnn model
- check model summary
- strids
- let's check what's giong on inside

#### Hooks
- replace NN.seuantial with our own sequential class
- keep same two lines
- get the mean and stdand save in a bunch of lists for every layer
- now sequential model keeps track of mean and variance
- run and fit but with two extra things
- lets plot means
- they are increasing
- it looks alfuw
- in earlier training the means until colapses
- keep colapsing
- until eventually converges
- the concert is the cliff
- there's lot of parameters
- are we sure all of them are getting into reasonable places?
- maybe most are with zero gradients
- well check later
- let's make this not happen
- let's look at std
- also bad
- let's look at the 10 means
- closeish to zero
- the first layer has a variance of 0.35
- the next layer is lower and lower and so on exponentially going on
- really close to zero
- we can se whats goingon
- final layers getting no activations 
- by the time they got that
- the gradient was so fast tehey fall a cliff
- let's try to fix
- we need a better wya to do it
#### Question about losing information
- actually waisting information
- we are taking more space for the same information we started with
- the idea is to pull out some interestig features
- increasing the number of activations is awisting of time

#### Pytorch hooks
- we need to use feature inside pytorch
- pyroch doesnt use callbacks
- use hooks instead but it is the same
- pass register_foward_hok with partial apend stats and i))
- replace previous things with hooks
- store means
- def append stats to alculeate means and std
- and register_foward_hook will tell you call three thingswith model, input and output)
- output we want nome por value that's why using partial
- call fit and replicate same thing with pytorch

#### Hook class
- create hook class
- initializes with register foward hook on some function
- callback object with self and partial
- get access to the hook
- get empty lists
- means and stds calleds from stats
- now hooks with list(Hook(l append lstats)
- Replicated same thing

- Refactor hooks class 
- When we are done using hook module call hook.remove
- Create __dell__ to cleans up memory to automatically call remove
- symilar thing in hooks
- when it is done calls remove
- for each registered hooks and calls remove
- for h in self but no interating was created
- LIstContainer was created and used in the initialization as superclass
- works better than numpy for those things
- with list container gas get item
- allows calling self.items or iterate over self with super class listconainer
- also defines repr to print the contents of listconainer
- with this class we create useful class with few lines of code to use anywhere
- our own listy class with hooks
- call hooks with model and append stats function
- we can get a batch of data
- gets mean and std
- pass trhou the first layer of the model
- and the man is not zero and std is 0.38
. use initialization with kaiming
- the mean is now close to 0.5 because of relu
- now with hooks run.fit
- this time well diong after initializing
- does not have the exponantial crash
- variances are closer to one
- he used with hoos
- create object given a name
- and then it will do __exit__ method to remove
- nice way to ensure things are cleaned up
- that's why using with
- but the concert nas does initial are doing something bad?
- how many activation are really small
- how many are getting activated

#### Append other statistics
- def append stats
- includs histograms
- run this with kaiming initialization
- and what we find is that even with a lerarning rate high we ca
- we see the growhs
- but concern is yellow lines
- most of histogram is 
- get the sum of the two ins a
- how many activations are nearly zero
- in the last layer mostly are zero, over 90 percent
- training a model looks it is nice but 90 percent of activations are being wasted
- lets try to fix it

#### Generalized Rely
- The truck is to use the better Relu
- General Relu class
- add subtracks value
- and leaky relu
- and a maximum
- foward function with ..
- get_cnnlayers now has kwardgs
- now we are able to get caracteristics of relu
- now relu can go negative with leaky
- we update histograms to -7 to 7
- change definition of get_min
- now we can train the model like before
- visualize layers
- ocmpared wo beforewe now see with no death is going on
- and now in the final layer less than 20 percent
- we are using most activations with careful initialization and personalized relu

#### Questiona bout histograms
- x axis is the iteration and y axis is how many activations are are the highes that they can be or the lowes they can be
- they show the oscilation
- some are in the max, middel
- some shows that all activations are basically zero
- in the secon now most of them are zero
- but there are more activations distributesd
- vew less than zero
- we are doing minus zero point 4
so the line shows what percentage are learnly zero which is only 20%. 
- **Take a look in the histograms  and how was calculated**

#### Lets do one cycle train with improvements
- lr learner
- run 8 epochs
- 0.98 in test set
- one option was added with uniform as kaiming uniform
- train same model
- People think uniform is better
- maybe uniform initialization mght cause a better richness of activations
- During the week check how accurate can we amek the model
- how can we tell if it is good
- **try to beat 0.98 percent with playing around**

### Notebook 07_batch_norm.ipynb
- Let's look at batchnormalization- paper batch normalization
- first show why it is a good idea
- page 3 provides algorithm
- looks scary
- but only has mean and variance and normalization
- we've done with code
- after that they multiply by gama and beta
- they are parameters to be learned
- this is the most important line here
- there a r two types of numers
- aprameters we calculated
- things that we learned


#### Convnet
- fit

#### Batchnorm
- get the mean and variance
- and update stats
- and variance
- and sobtrack mean and divide my squared of variance
- gamma as mults and adds 
- are defined as nn.Parameters which is initially ones
- adds as a bunch of zeros
- and they can be learned
- it is just like weight and biases
- two things
- what happens in inference times
- we ight remove things interesting on an image
- let's keep an exponentially weighed moving average
- running average of last batches means and variances
- in foward we use fmeans and variances
- how to caluclate running average
- we call register_buffer vars
- if we move the model to gpu the register will also be moved
- variances and means are now part of the model
- when doing the inference we need those values
- we need to save them
- register_buffer allows to save them
- averages all the batches in x y coordinates
- only a mean in each filter
- keepdim is true so still broadcast
- let's take a moving average
- if we have a bunch of datapoints
- grab 5 a time and the average of 5
- however every single one hast it 
- it is giant
- we don't want to save all the history
- there's a handy trick
- use an exponentially weighted moving average
- first average is the first point
- then second point is 5
- starts with momentum
- and second value we multiply by momentum and add second value and multiply by one minos momentum
- and mean 3 is mean 3 times .9 plus a new value times .01
- the thing before plus a ittle of the other one
- bu the time we get later the amont from the other datapoints is an exponential decays
- we only have to track one value
- function where is osme previous value tiems a beta 
- this is called linear interpolation
- in pytorch is called lerp_
- with new mean using momentum
- however lerp use the oposit of momentum
- so momumentum of 0.1 is 0.9
- momentum is the opposit of what you expect
- in foward use that
- now create a new conv layer
- append batch norm layer
- remove the bias layer beauase of batch
- more convinient initialization of init_cnn
- and then we train withoug hooks
- means start with zero and std with 1
- and training hasnt entired got ridefo and then crash in the very end of training in the end of 1 epoch
- **something to check**
- we are now able to use putorch batchnorm

#### With scheduler
- now let's ty with learning finder
- batch norm allows ofr that
- however has a problem
- you can apply to online learning tasks
- the variance of that batch is infinite
- for segmentation task task can be a problem
- anytime we have variance
- also hard for RNNS

### Slides RNN
- we have a hidden stats
- and we unroll it 
- stack together as a recursive
- propagates same weights variances
- how to do batchnorm?
- cant put different ones
- not really clear how to put them
- the paper layer norm suggests a solution https://arxiv.org/abs/1607.06450
- looks terifying
- with nobel price
- when converting to code has 10 lines of codes
- full of explanations
- withouth running average
- not taking the mean across batch
- each image has its own mean
- that's what layer norm is
- the problem sis even with a lower learning rate it is not good
- throwing together but does not helps
- for RNNs for now is what we have to use
- Though experiments: Distinguish foggy days and sunny days
- less contrast 
- will make the means to be the same
- answer: we couldnt with layer norm
- everywhere in the middle layer is the same
- throws away
- layers norms is partial
- Paper called instance norm
- easier to read
- turning to code reflcts into tyni differences
- now removing difference in means and so throwing away 
- it was designed for style transfer
- finally a paper called group norm with pecutres of differences
- group norm get a few channesl N and H together
- in putorch we can use group norm
- How to solve it?
#### Fix small batch sizes
- whats the problem
- when we compute the statistics on a small batch it is possible that we get a standard deviation close to 0 because there aren't many samples
- use epslon
- epslon to sefide in the variance to avoid floating points
- common to see in the botton of division
- but it is a fantastic hyperparameter
- avoids in the worst case to not blow out
- so one option is to use a higher value of eps and use as hyperparameter to take advantage

#### BEtter idea
#### Running Batch Norm
- Very simple
- in the foward function don't divide byt the batch std
- but instead use the statistics of the moving average
- because if batchsize is 2 then variances will be almost zero
- Test code
- learn and run
- now has 91 percent accuracy on one epoch
- some details to get it right to work
- but are details we will see in other places during the coruse
- dont get overwhelmed
- first thing is in normal batch we take the the running averagte of the variances
- instead we calculate variance using the second calculation using e(x) squared
- keep track of the squared and the sums
- register buffer with sums and squares
- dimensions with 2, 3, 
- and then well tak the mom
- for the variance we take the squares using equation
- other details is to register buffer with count
- required debiasing
    - make sure at every point no observation is weighet to highly
    - initialize both sums and squares to zeros
- divide result by beta value
- uses the division os squared sums
- this is called debiasing
- straith foward math
- **Try during week**

#### What can we do in a single epoch
- Got 0.97 percent in one epoch
- Try to beat him
