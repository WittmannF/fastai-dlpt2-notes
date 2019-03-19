# Deep Learning Part II: Deep learning from the foundations

## Some logistics
- Room 153 @ USF (101 Howard ST)
- In person 
- International visotiors prior

## Intro
- Very different part II
- Implement fastai library from foundations
    - Basic matrix calculus
    - Training loops
    - Optimizers customized
    Customized annealing
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
    opportunities
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

## There are many opportunities

## Review of part I

## Overfit > Reduce overfitting > There's no step 3

## Five steps to avoid overfitting
1. More data
2. Data aug
3. Generalizable architectures
4. Regularization
5. Reduce archtecture complexity

## It's time to start reading papers
- Even familiar stuff look complex in a paper!
- Papers are important for deep learning beyond the basics, but hard to read
- Google for a blog post describing the paper
- Learn to produce greek letters

## List of mathematical symbols on Wikipedia
- https://en.wikipedia.org/wiki/List_of_mathematical_symbols
- or use detexify

In the next couple of lessons:
## Steps to a basic modern CNN model
- Matmul
- Relu/init
- Fully Connected foward
- Fully Connected backward
- Train loop
- Conv
- Optim
- Batch-norm
- Resnet

## Lesson 00.ipynb
- How to export a code from jupyter 
- Fire to execute a function
- Play around on jupyter, automate stuffs, and use fire
? So fire is used to execute methods and test them?
- You truly program on jupyter

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
how to do our own stuff
how to write codes fast
broadcast trick is one of the best for making loops fater. 


