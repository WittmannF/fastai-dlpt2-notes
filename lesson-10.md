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
