# Deep Learning Part II: Deep Learning from the Foundations

## Intro
- Very different part II of previous year
- We will implement fastai library from foundations
    - Basic matrix calculus
    - Training loops
    - Optimizers customized
    - Customized annealing
- Read papers
- Solve applications that are not fully backed
- In the end, implement on Swift
- Cutting edge is really about engineer, not about papers
    - Who can bake that things in code
- Part II will be more about bottom up (with code)

## Embracing Swift for Deep Learning
- Chris
    - Built compilers, C for Mac
    - Built the most recent language
    - Currently dedicating his life to deep learning
- Julia has pottential as well!
- S4TF Pros
    - Write everything in swift
    - See whats happening
    - opportunities
- Cons
    - Minimal Ecosystem
    - Very little works
    - Lots to learn
- PyTorch Pros
    - Get work done now
    - Great ecosystem
    - Docs and tutorials
- Cons
    - Performance
    - Pythons types
    - Mismatch with backend libs
- Swift will possibibly take place in this field

## What do we mean by from the foudations?
Recreate fastai and much of pytorch? matrix multiply, torch.nn, using:
- Python
- Python stdlib
- Non ds modules
- Pytorch array creation, RGN, indexer
- fastai.datasets
- matplotlib

## But why?
- Really experiment
- Understand it by creating it
- Correlate papers with code
- Tweak everythin
- Contribute

## There are many opportunities in this class
- Your homework will be at the cutting edge
- Few DL practictioners know what you know now
- Experiment lots, especially in your area of expertise
- Much of what you find will have not be written about before
- Don't wait to be perfect before you start communicating
    - > Write stuff down for the You of 6 months ago, that's your audience
- If you don't have a blog, try medium.com

## Recap of part I
- He assumes that you don't remember everything. 
- As we go on, if necessary, you go back and watch that video
- Especially the second half
- He assumes that you know know about SGD from the scratch
- Topics
    - Convolutions
    - Weight decay
    - Dropout
    Batch

## Overfit > Reduce overfitting > There's no step 3
- Try to make sure we can train good models
- There are ~~three~~two steps for trainig a good model
1. First we try to create something with way more capacity than we need
    - No regularization
    - Overfit
2. Overfitting does not mean training loss lower than validation loss 
    - A wealthy model almost always will have such behavior
    - **Overfitting is when you actually see your validation loss getting worse**
- Possible three would be visualize output
- One is easy, the two is more difficult. 

## Five steps to avoid overfitting
1. More data
2. Data augmentation
3. Generalizable architectures
4. Regularization
5. Reduce archtecture complexity

- Most begginers start with 5 but that should be the last
- Unless the model is too slow

## It's time to start reading papers
- Even familiar stuff look complex in a paper!
- Papers are important for deep learning beyond the basics, but hard to read
- Google for a blog post describing the paper
    - They are not selected for their outstanding clarity of comunication
    - Usually a blog post will do the job way better than the paper does
- Learn to produce greek letters


## List of mathematical symbols on Wikipedia
- https://en.wikipedia.org/wiki/List_of_mathematical_symbols
- or use [detexify](http://detexify.kirelabs.org/classify.html)

# In the next couple of lessons

## Steps to a basic modern CNN model
> We are going to create a pretty confident modern CNN model

- Matmul
- Relu/init
- Fully Connected foward
- Fully Connected backward
- Train loop
- Conv
- Optim
- Batch-norm
- Resnet
    - We already have this last one from Part 1
    
**Goal of today's class**
- Go from matrix multiplication to backward pass

## [Lesson 00.ipynb](
https://github.com/fastai/fastai_docs/blob/master/dev_course/dl2/00_exports.ipynb)
- How to buld an app on jupyter notebooks
- More productive on Jupyter notebooks

### How to pull out bits of code from jupyter into a package
- Use the special comment `#export` to tell the system a cell that you want to keep and reuse.  
- Then use the file [`notebook2script.py`](https://github.com/fastai/fastai_docs/blob/master/dev_course/dl2/notebook2script.py) which goes through the program and find cells with the special comment `#export` and put them into a python module.
    - Path.stem.split("-") is used for the output filename, hence, the output name is the first portion before an undesrcore. If there's no underscore, then the full name. 
    - The exported module goes to a folder called `exp`
- We can then import the exported module using `from exp.nb_00 import *`
- Creating a test framework
    - `test` and `test_eq` using `assert`
- Use [`run_notebook.py`](https://github.com/fastai/fastai_docs/blob/master/dev_course/dl2/run_notebook.py) to run the tests outside of the jupyter notebook
    - `python run_notebook.py 01_matmul.ipynb` run the tests outside of the jupyter notebook
    - We can see the assertion error when running in the terminal
- Now we have an automatable unit test framework on jupyter notebook 
- Fire to execute a function
    - **It takes any function and automatically converts into a command-line interface**
    - Inputs of a function are converted into arguments in the command-line
- Notebooks are json files. 
    - We can import cells and play around jupyter notebook files converting them to json files
    - Example: `

## Matrix multiplication
- import mnist
- extract mnist into train and y valid with numpy arrays
- convet numpy arrays to tensor (np is not allowed)
- tensor was previoulsy imported from pytorch
- get number of columns and rows from training data
- Some visualizations and stats 
- Doing some obvious tests from above
- img = xtrain
- img.view28
- plot(img)

### Initial python model
- wights receives random values 784 in and 10 out
- bias initialized with zeros

#### Matrix multiplication pseudocode
- function of matrix multiplication
- review of matrix multiplication from matrixmultiplication.xyz
    - A few loops going on: three
- def matmul
- ar and ac receive shape of matrix a
- br and bc receives shape of matrix be
-  assert the shapes of ac and br are the same
- c receives zeros with shape ar and br
- fori in range ar
    - for j in range 
        - for k in range(ac):
            - c(i,j)...
    
- m1 receives x_validation
- m2 receives weights
- time the usage of matrix multiplication
    - result 800ms
- the multiplication is quite slow. Let's try to speed it up 50000


#### Elementwise operations
a receives a tensor
b receives another nesor
sum both tensors
a less tham b conveted to a float and get their mean
m receives a tensor matrix

**calculate frobenius norm**
- trying to translate equations into code
- sum is two for loops, one in i and other in j
- square of the sum of all the terms
- Howard dont write latex :)
    - He copy and paste from google wikipedia
- m times m sum and sware root

#### Elementwise matmul
- replace third loop with frobenius norm
- c = a_i + b_j . sum
- time it 700 faster
- backend in c
- let's check if it is right
- define function near(a,b) using torch. allclose()
- test_near receives test(near)

#### Broadcasting
- run at cuda
- remove loops
- describes how arrays with differents shapes are treated during arithmetic operations
- first used in numpy
- is like a new programming language
- a > 0 comparing tensor with value, it works because of 0 is being broadcast to have same dimension as a
- a+1 also broadcast 1 to the tensor a
- 2*m broadcasts 2 to tm

#### Broadcasting a vector to a matrix
- c receives a tensor
- m and c have different shapes (3x3) and (3x1)
- array is broadcasted to 
- theres no loop but it looks like as there was a loop
- c.expand_as(m)
- version of c as a broadcast tensor rank 2 instead of array
- c speed with no looop
- t.storashe show sthat thers onl 3 values bein stored
- b.stride() 
- tensor behave like higher rank things

what if we want to take a column instead fo a row?

rank 2 tensor of shape . 

- c.unsqueeze(0j=) is a shape one comma 3
- c.unsqueeze 1 is a shape three coma one
- this is interesting because 
- c none columns is is same shape 1 and c : none is same of squeeze 1
- c[:, None].expand as m broadcast as columns

#### Broadcasting in excel
- semicol and none is column and note afer is row

### Eliminating loops with broadcasting
- The entire row of c[i] (which is c i : (you can eliminate the semicolomns)
- ci receives row i of a. unsqueeze (-1) which is the last dimension
- you can also write a, i None instad of -1
- rank 2 tensor and b is also a rank 2 tensor. unsqueeze to broadcast into b and then sum them over the rows (dim 0)
- **Homework:** Why this works??
- 3200 faster now with broadcasting
- getting rid of looops also reduces errors

#### Broadcasting rules
c[None, :] is row based
shape 1 by 3
eleentwise multiplication
c times c[:, none] value by value matrix
broadcast into squared
- they dont have to be in the same rank
- you can normalize by channel with no limnes of code

### Break
- Goal make code faster 
- how to do our own stuff
- how to write codes fast
- broadcast trick is one of the best for making loops fater. 

## Einstein summation
- Popularized by einsteing for higher rank matrix
- Compact representation for combining products and sums in a general way
- if pytorch didnt have batchwise multiplication, noe new index oadded would transform it
- c i j + a i k times b , j
- a i k k j -> i j
- using index inside string for notation and matrix
- def matmul(a,b) return torch.einsum('ik,kj->ij', a, b)
- now it is 16 times faster using einstein sum
- trajedy that it exist
- a programming language using string
- amazing but so few thinkgs it does
- I want to generalize to a language
- hope is that swift giv ability to write stuffs that really fast 
- swift is even faster than einsum

### Pythorch op
- use pytorchs function or operator directly for matrix multiplication
- 50 thousand faster
- m1.matmul
- divide into batches, written into assemb, blal, library of linear algebra, for example cuBlAS
- awfaw, bc program is limited to a subset of thinks that BLAS can write read
- limited to python methods
- people working on this on swifit
- facebook research and tensor compresions
- in python we are restricted to m1.matmul
- still pure ehanced way is 10 thousand slower
- need of libraries
- t2 = m1@m2

## Notebook 02 Fully Connected .ipynb

### The foward and backward passes
- x train, y train, x y get data
- get standard deviation
- normalize using standard deviation
- note use training, not validation mean for normalizing validation set
- afer doing that mean is close to zero and std close to 1
- test function if it is really normalized
-n,m get xtrain shape
- c output size

defining the model

Model has one hidden layer

Foundations version

basic architecture

number of hidden layers  nhis 50

two layers is two wegiths and biases matrices

w1 is random values divided by sqare root of m
b are zeros

w2 is random values (nh,1) divided by math sqarue of nh

t is linear of three vectors

divide by sqare root m then tensor has lower values

simplified kaiming initialization, wrote a paper about it

test mean and standard of weight 1

thing that really matters when training

fixup initialization
paper with 10000 layers ....
how initialization is made really matters
spend a lot of time on this in depth
first layer is defined by relu
relu is grag data and clamp min to z (replace negative to zero)
try to find the function internal on pytorch

unfortunatelly does not have mean zero and std of 1

demonstration 

distribution of data
then took evertyhing smaller and took out
obviously mean and std are gong to differ
one of the best papers of the last years
suprassing human level performance on imagenet calssification
full of great ideas
read papers from competition winners if a great idea
where competition ideas has 20 good 
kine initialization
seciont 2.2 initialization of filter weights for rectifiers
are easier to train cbut a bat initializatiom still hamper the learning of a non linear system
initializet with random gaussian distributions
glorot and benchi proposded a new initialization
paper undesrtand the difficulty of training deep neural netowrs

well be reimplementing stuffs from the paper

! read this paper

one suggestion is another approach called normalized initialization

however does not account to relu

super simple is to replace the one in the top to a two in the top in relu
divide to swqre of 2/m

closer to std 1 and mean zero

!! homework , read 2.2

#### Foward propagation layer

take it throuh step by step.
6 paragraphs to read
!! read section foward
> "this leads to a zero-mean gaussian distribution whose standar deviation"

something new and obvious is to replace relu to x.clamp_min(0)-0.5 which will return to the correct mean

he had to add a mode callsed fan out 
- fan ipreserves the magnitudes in the output pass
    - Dividing by the fifrst or second
- we need it because the weights shape is 784 by 50 while a linear torch is 50 by 784
- look into source code to undertand using double question mark
- it calls F.linear (F.nn.functional)
- letds look
- a linear layer with their transposed
- thats why we gave the oposite when compared to torch code

what about conv layers??

check documentation
mostly documentation
mostly code is under $_ConvND$_ 
at the very bottong theres the file conv
it has a special multiplier math.sqrt(5)
seem to work pretty badly
always a good idea to add comment
- feeling that this is not great
- we desined our own activation function
- of relu minus 0.5
- using it the mean is almost zero and variance is almost one
- make sense why this makes better results

Doing a foward pass

def model
linear layer
relu 
linear layer

time it
test

#### Loss function: MSE
- simplify thinkgs using mean square error
- expect a single vector
- use squeze to get rid in output.squeeze()
- very common broke code because squeeze into a scaler
better to put dimension in squeeze (-1) for example
y train, y validation get floats

get the mean squared error

### Gradients and backward pass
- paper the matrix calculus you need for deep learning html by jeremy howard and terence parr: https://explained.ai/matrix-calculus/index.html

> all you need to know is the chain rule

start with an input, then a linear layer then a relu then second linear layer, then mse then y pred

other way is
y_pred = mse(lin2(relu(lin1(x))),y)

we want the gradient of the ouput y with the respect of the input x

y equals to f(u)
u equals to f(x)
derivatie dy/dx is dy/dy times du/dx
thats all you need to know

usually iti is not treated as division, but yo actually you cant
defivatinve is taking some fuction
dividing small change in y by small change in x

start with mean squared error
gradient of the loss with respect to outputprevious layer

it is two times error 

def mse_grad(input, target)

the input of mse is the output of previous layer

def gradient of relu

either zero or 1 which is inp > 0 dot float times out.g

linear gradient defined


def foward and backward

in backward pass
mse_grad
lin_grad
relu_grad

value inp.g is updated in each function

loss is the mse we never use it

the loss never appears in the gradients

so w1.g w2.g ans so on contain the gradients

let's clone weights and biases and test them

we cheat a little bit and use pytorch autograd required grad to check our results

using test near to check if results are correct

### Layers as classes
Refactory
- recreating pytorch api

class Relu()
    def __call__ # treat relu as a function and call whats inside
    safe input and outpu
    def backpro self.inp.g = self.float.self.out
    
    
for linear compute self.w.g in backward

** backward always compute .g**

Class model


init has w1, b1, w2, b2

def call with x and target

def backwar with self.loss.backward() to save loss.g

w1.g, b1.g w2.g 
model = 

However, that was slow!!

### Module foward

create a new def in module callde forward which initially raise not implemented

class Relu(module)
it used foward

the thing to calculate the gradient with the respect to the weights, we can reexpress that with einsum
now it is faster with einsum

time it now is 143ms intesad of 3s

> This is why we have to use those modules

#### Without einsum

repplace with matrix multiplication
140ms

implemented nn.linear 

#### nn linear and nn.module
even faster

?? So, does pytorch uses or not autograd?

### Next lesson
train loop
