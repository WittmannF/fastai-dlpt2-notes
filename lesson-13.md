# Swift for Tensorflow
- Colab Notebook with Swift: https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb
- Instalation of Swift to tensorflow
    - Jeremy's Harebrained install guide
    - https://forums.fast.ai/t/jeremys-harebrained-install-guide/43814
- Possible on cloud with GCP

# fastai + Swift for TensorFlow
- "Lesson 13 aka lesson 6 aka lesson 1"
- Cool things during th week
    - RobG: Came in 1st and 2nd in a contest
    - Alena Harley: Classifying Clinicall Actionable Genetic Mutations
        - Using fastai and wiki103
        - 15th place
## Goals
- Recreate fast ai
- and much of PyTOrch
- Python, Python stdlib, nonds moduls, fastai.datasets, matplotlib
    - We will revisit this but now on Swift
- In the end we'll get to a swift version of RsBlock and XResNet
    - Very familiar to python version
- There are datablocks apis in swift
- Three weeks ago they didn't expect to be possible

## So many people to than!
- Chris Lattner, Sylvain Gugger for writing fastai notebooks in 3 weeks. Swift for Tensoflow Team. Alexis Gallagher for value types

## What's the future of fastai?
- We don't have a choice
- Pace of progress
    - We couldn't build together on tensoflow somethings
    - Then pytorch allowed because of gap in tensorflow
    - fastai was created because of gap in tooling for pytorch
    - but now were hitting the limits of python
    - so we need to jump this gap too - we're working towards Swift 4 TF
    - I am confident that this will also make fastai for pytorch better too!

- Why Swift over Julia?
    - Swift has google

## Why not Python?
- Python is nice... but:
    - Slow: Forces things into external C libraries
    - Concurrency: GIL forces more into external C libraries
    - Accelerators: Forces more into CUDA, etc

## What is Swift? The claims
- Swift defines the way large classes of common programming erros
- Is complide and optimize to get the most out of modern hardware
- Ambitious goals
- Swift tries to be expressive, flexible, concise, safe, easy to use and fast. Most languages compromise significantly in at least one of these aresas. Swift doesn't. 

## What is swift?
- Young and fast: first released 5 years ago
- Tuned for IDE and user experience:
    - intellisense, inline error detection, refactoring, etc
- Centerist: not trying to be novel
    - aim to feel familiar
    - borrow good ideas wherever they are
- built by the team who made LLVM and Clang

## Swift for TensorFlow
- First principles rethink of deep learning systems from the ground up
    - langage integrated autodiff
    - python integration
    - efficient base language
    - no barriers: everything, all layers, is exposed and hackable by you
- Early phase 0.3 project:
    - many things changing, envolving, iterating, and improving
    - bugs and missing features too
    
## Build on what you already know!
- Swift vs python code on pytorch

## Example Layer: piece by piece

```
struct MyModel Layer {

    var conv = ...
    
    @differentiable # tell the compalier that this should be differentialbe
}
```

## Why Language Integrated Autodiff?
- MS word autocorrector style for showing errors while typing

## S4TF and TensorFlow?
- Keras estimator
- Python4TF
- fastai
- Swift4TF
- Tensorflow infrastructure
    - Graphs, kernels, TF Lite, Distribution, Deplyiment, TFX, Eager runtime, ...

## let's Dive in! --> 00a_intro_and_float.ipynb

```
let batchSize = 6 //constant
var b = 1+(4*batchSize)

batchSize = 8

var someFloat: Float = 1 + 4*9

//Jeremy might not like greek letters but he surely loves emoji
var emojy...
```

func distance(x: Float, y: Float) -> Float {
    return sqrt(x*x+y*y)
}

// Functions default to having argument labels:
distance(x: 1.0, y: 2.0)
```

- We can use it to import fast ai python library using swift
- But we should graudally replace

## Impractical Programming Languages
- We now have a bit of graps on how swift fasics works
- Try matmul
- But first what is a compiler

## What is a compiler

## Programming Languages ias an intermediate

- Human ()> swift (compiler)> intel PC


## How do compilers work 
- Swift -> swift frontend -> optimizer > X86 CodeGen > X86 Code
- LLVM

## Compiler optimizations
- Constant folding
- Dead code elimination
- inlining
- arithmetic ismplification
- loop hoisting

## Languages atoms and composition
- Python:
    - Atoms: C code that implements python object methods
    - Composition slow interpeter that combines c calls in interesting ways
- C++ atoms is int, float C arrays, ointers
    - Composition is stricts and classes, std complex std string std vector
- why is array hard coded into the complier when string is library featrure?

## How does swift work?
- Swift is syntatic sugar for LLVM
- primitive operations are LLVM instruction
- Compositions structs and classes
    - String, dictionary, but also array
    
## Float in Swift
```
public struct Float {
    var _value
}
```

- Check implementation on github of float
- you can implement low level stuffs

## Back to notebook 

## Building floats
- Assemble code very similar to native code
- Can use emojis
- He created libraries using basemath with floats
- and started using in the language
- You can change
- Common to add extension to basic code
- and you can make it in the way you want
- How does an work...

## Tensor internals and Raw TensorFlow operations
- Closure {} are like lambdas in python
- Support extension of array, see `doubleElements`
- Swift can run in backend with GPU using PTX (from nvidia)

## Notebook 01_matmul.ipynb
- Swift version of matmul
- 900 ms in python vs 0.13ms in swift
- Now we can write in obvious ways and have them fast
- time was wirtten in notebook 00_load_data.ipynb
    - Buit on sracth
    - using dispatch
    - both time it and%% time (using repeat)
- Implemented downloadFile from url
- There's Path
- foundation is not that great
- equivalent notebookToSript(fname...)

## Getting the performance of C
- We can do better
- Using unsafemutablepointerbuffer
- Run twice fast

## Swift loves C APIs too: yo get the full utility of the C ecossystem

## Working with Tensor
- Lets get back to matmul and explore the Tensor type as provided in the TensorFlow module
- Some highlights cow to get random or zeros
```
...
```

- **matMul operator** compose key or matmul 
- We can reshape them
- convolutions, sum, sqrt, comparisons iwll turn into booleans
- point wise operator .< instead of <
- 6 seconds to do a matmul with tensors
- why is that slow??doing one float at a time
- it turns out that tensors are very goold at buld tata processing, byt they are not ogod 
- make sureto use...
- Presentation

## Pytorch is like an airplane
- you have to give it plenty of work to do to justify the time to drive to the airport, go security, takeoff , land, 

## Tensorflow was desined around the idea of creating a call graph then feeding it values
- there will be a x and a y
- now run this computation graph and use these things
- Pass in some data, different feel to pytorch
- because of this, TF behave like a shipping shipping ships

## Tf eager is largely syntax-sugar. it stills need a lot to do to make it truly useful

```
import tensorflow as tf
tf.enable_eager_execution()

x=[[2]]
m = tf.matmul(x,x)
print("hello, {}".format(m))

```

AS of april 2019 a small matrix multipy on gpu using tf eager takes 0.28 ms 10x longer thatn tensorflow

## Tensloflow needs an ecosystem to leverage such ans approach

## Under the hood everything is chanigng...
- Now it is using TF eager
- In future XLA and MLIR and XLA in noe year
- Swift 4 tensoflow is on top of them
- fast ai wil be on top of that

- So we wont be able to see the full potential yet
- In one year it is going to be very fast
- Next year we'll use XLA

## Vectorize the whole thing with one Tensoflow op
- time(repeating 100) {m1 * m2}
    - average 0.02 ms
- tensorflow has a database of the operators it defines

## Jump to notebook 11_imagenette.ipynb
- ipmort fastai notebook 10 mixups
- images augmentations
- load data
- We don't need partial anymore
- Databunch
- Transformations
- Training closure
- Batch might not return anything and exlamation assume it is sometihng
- Got show batch
- ResBlock conv layers very similar to python code
- We have a stateful optimizer
- and also a Learner
- And one Cycle Train
- 3 4 times slower than Pytorch and twice memory
    - but it was made in two weeks

## For next week
