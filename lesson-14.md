# Lesson 14: fastai + Swift for TensorFlow

## Today The ML part of S4TF
- Building up the stack
- Swift features
- https://colab.research.google.com/drive/1uYAfmzsj4IRvAYqvuTdErwO0BZuU6Nf8

## Under the hood, everything is changing
- fast ai
  - swift 4 TF
    - TF eager (now) > XLA (2-3 months) > XLA + MLIR (1 year)
## Efficient Matrix Codegen is Hard: Tensors are Harder
- Fusion, tiling, cache blocking, unrolling, multithreading

## What is XLA?
- Graph of Tensor Ops > Tensor optimizer > Tensor code gen > Google TPU code, GUPU Code, Intel X86 Code

## XLA Fused RunningBatchNorm Computation
- Graph with transformations

## but where do graphs come from?
- Tensorflow 1.x Sessions
- TensorFlow 2.x:
  - tfe.defun - tracing
  - Autograph - sort of like torch
- Swift for TF
  - Improved tracing
  - Graph program extraction

## What is MLIR
- XLA is greag for
  - super high performance with common operators
  - common operators that can be combined in lots of ways
  - high performance accelerators (even weird ones)

## Custom Ops?
- Lots of different approaches
  - Halide, PlaidML, TVM, Tensor Comprehensions
- Stragety? provide a common framework for these all to plug into
  - you can express things in highest abstraction that can do what you want
  - S4TF always let you reach lower if you have to

## Example from Tensor Comprehensions
- Einsum to a crazy xxx version
- Not a syntax proposal! Taken from tensor comprehension paper
- We'll be able to have a "descritive syntax" (like SQL) to ML and DL

## Back to Workbook: C integration Examples, 
- https://github.com/fastai/fastai_docs/blob/master/dev_swift/c_interop_examples.ipynb
```
%install-extra-include-command pkg-config --cflags vips
....
```
- Let's say I want to do some video processing 

### Sox
- init sox and read sox audio
- Jumped into VIM
- created a cdicrectory called swiftsox and created a few things
- file for package.swift
- define swift packages
- next step is to create a file called sources/sox/module.modulemap
- The final 3rd file is in sourcex/sox/sox.h
- with those three files you can import sox and the c function is available to swift
- opening doors
- C libraries now are available 
- functions initsox and readsoxaudio
- Initialize sox and read sox audio
- usually return a c pointer, calling signal which will have rate, precision, channels and length
- in python we can also import c libraries however it is not so helpful
- We can bring both C and Python together in swift
- use ipython .display to display the audio

# Swift+C interop

## C is important but also very gross
- important! Lots of useful code available in C
  - lowest commmon denominator between language and library interop
  - ofter use as the stable api for c++ libraries

- gross
  - pervarsively unsafe by default - security and correctness
  - weird features: macros, bitfields, unions, varargs, volatile, inline
  - very difficult to parse? preprocessor, context sensitive grammar
  
- C++ is simultaneously more important and more gross

## Swift loves C APIs
- provide direct low-level access to almost arbitrary C APIs and types
  - remap them into swift concepts and types
  - you directly use it fro your code with no wrappers build steps or overheads
- best practice
  - write a nice swift api that wharps up
## Import remaps c into swift

## Clang importer remaps C into swift
- C Header files

- Generated swift interface

- Swift code vs C code
- Fullof macros, conditions, inline functions, structures
- cleaner in swift 
- look the original math.h and math.h in swift

## Parsing C is hard, codegen is hard too
- function is inline only it cant be called without generating code for it
- cant parse code without parsing 

## Swift loves clang and LLVM
- Let clang and LLVM do the heavy lifting
  - parse the header files, store in binary modules, build asts
  - generate code form all the weird C family features
- integration between C and swift swiftC and clang then to LLVM and finally to machine codeee
- more details, skip the FFI, embedding clang for oxxteroperability

## Back to workbook c_interop_examples.ipynb

### Vips
- import tensorflow
- libvips fast image processing
- find aswiftvips folder
- same package.swift with some extra lines
- sources/vips/module maps
- sources/vips/shim.h with one line of code
- can now import vips
  - their docuemntation optional argumets in C
  - use bars
- file core.h
  - perform resize 
- now can add to swift loading recise
- and now can load image and use optional arguments
- vips get functionand then free function calling mem
  - defer free
  
### OpenCV integration example
- Fast and reliable
  - but hates python multiprocessing
- Maybe can work on swift
- but now opencv is onlyc++ not compatible on swift

- BUT, there's a workaround
- Using C in between C++ and swift
- created COPenCV
- creatin headerin plain C 
- In the header look like C
- And now can import opencv

- read image, check size, underlining C pinter, timing

## Table with unsafe[mutable][raw][buffer]pointer[<T>]
- Pointers are just memory addresses. Direct memory access is unsafe. Mutable means you can write it ...

## For DS going low level
- Huge world called sparse convolutions
- two people in the worold doing that because requires extra cuda code
- used to know only excel and then visualB

## Overwhelness of too many new languages
- 30s in python to load images, then 11s using cuda and then on opencv 7.2 seconds 
- Have no choice 

## Data block foundations, in Swifity/functional style
- comparing to the python version is significantly faster

- problems with datablocks in python
  - difficult to deal
- One little package of imformation and swift will tell if something is missing
- understand whats going
- using protocols

## Protocols in Swift
- interfaces, typeclasses or abstract classes

## Split interface and implementation
- with this approach swift willl tell how the function should been called

## Define behaviros of a group of types
- describes a behavior of a group of types
- has a name, expected behavior, infariants

## protocols compose and refine

## Protools and generics work together
- protococls give ability to define functionality over type classes with <> syntax
- One fucntion that will 

## Protocols provide Mixins
- give new methods to all implementations of a protocol

## Mixins show up where they make sense

## Protocols: Lots of more to learn
- just skimming the surface
- blog posts

## OFF
- https://en.wikipedia.org/wiki/Category:Numerical_programming_languages

## Fully connected model
### Foward and backward passes

## Mutation game
- x.append in swift vs python
- python end up mutable in y if x=y
## Values vs References
## Tensor references causes
- full of clones in fast ai

## What if high level api design was able to influece the creation of a differentiable programming language?
- Stateful Optmizer
- Callbacks v2
- datablock api newly functional
- But still havent answered the question. What if keypaths were a query language. What does an embedded DSL for writing kernels look like?
- What could DS do with an infinitely hackable differentiable language?
