# Lesson 12 - Transfer Learning and RNN
- State of art in imagenet
- Refactoring on 09b_learnier.ipynb
- Pass optimizer as opimization function

## Mixup/Label smoothing
- Data augmentation might not be necessary for great results
- Paper Bag of Tricks for Image Classification with CNNs
- Grab imagenette dataset
- get rgb, resize to 128
- create a databunch

### Mixup
- We'll take two images and combine them
- We'll create a data augmentation to predict a mix of things
- Linear combination of two images
- And do that for the labels
- Output 0.7 and 0.3 from linear combination
- Paper mixup: beyond Empirical Risk Minimization
- Implementation we'll have to decida what well use
- Values of linear combination will be randomized using U shape on edges near 0 and 1
- Smoth histogram
- Sampling from a probability distribution
- The paper mention as beta distribution
- The U shape requires math (gamma function)
- start with factorial function
- plot them
- divide both sides by n
- and now you got factorial(n)/n
- breaking the no greek letters rule
- does not have a domain meaning
- easier to write it on code
- Easy to write using compose keys
- quick as typing
- We can pick a parameter high, the probability of near 0.5 is higher
- Instead of using hot encoding it is easier to use in the loss function p*loss + (1-p)*loss
- If validation, don't use mixup
- Powerful augmentation system
- We're actually replacing loss function
- Create new loss function that it is going to reduce 
- You can use this for layers adam
- How dows softmax interact?
- Instead of one hot encoding use 0.9 hot encoding

- Mixup does not require domain specific 
- does not create lossiness
- Almost infinite data aug options
- De
### Label smoothing
- Handle noising labels
- Diagnostics are not perfect
- Noisy data improve
- Don't trust if they say that you can't
- Powerful technique

# Training in mixed precision - Notebook 10c_fp16.ipynb
- Half precision floating point
- You can't use half precision everywhere
- Do the foward in fp16 weights and everywhese else in fp32
- apex do that
- function model_to_half
- run.model = fp15.convert_network(self.model, dtype=torch.float16)
- Speedup in vision models and transformers

## Dynamic loss scaling
- Sometimes training with half precision gives better results

## XResNet
- Its the bag of tricks pager resnet 
- With their suggestions of tweaks
- Figure 2
- 2.b three confs in a row
- stem is the very start
- Use ReLU in the activation function
- Initiazlize weights sometimes 0 and sometimes 1
- Why?
- Figure 2.c with two paralel layers
- What if we set the weights near output to zero?
- Output is zero
- Great way to initialize a model
- This way initially all gradients are the same
- Allows training at higher learning rates
- When adding stride 2
- 0.83 training in 2 and a half minutes

# 11a_transfer_learning.ipynb

## CUstom head
- create a cnn with 10 activations out
- Select cells, hit c, then v and merge
- Worse results compared to frozen layers
- Batchnorm problem
- Easily fixed
- Not freeze all body parameters

## Discriminative learning rates

# RNNs

- Are not in the past anymore
- Transfer learning path to RNN
- Genomic aplications
- Drug discovery
- Still it is the tip of the iceberg

## ULMFiT is transfer learning applied to AWD-LSTM
- Create a langague model on some text
- Fine tune wt103 using IMDb
- In both cases we have to preprocess 
- Preprocess IMDb for classification
- Fine tune IMDb LM for classification

### Preprocess text
- training, unsupervised and testing folders
- Define a sublcass to read the texts in the corresponding filenames
- Just in case there are some log files, we restric to thake the folders train, test and unsupervised
- Get a text
- use random splitter
- Let's convert to numbers

### Tokenizing
- Use spacy
- We have pre-rules, codes that are run before tokenization
  - Remove useles spaces for example
- Especial tokens
- If we see a character, replace it with a function
- Repeating token with the number of times that repeated
- why?
  - avoid useless vocab elements
- Add tokens before and after each phrase
- List of special tokens: https://forums.fast.ai/uploads/default/original/3X/b/1/b12bc835a3bda16de1d44ccf1a040b6b1bf364f1.png
- Stop words are a terrible idea
  - Never do this

# Notebook 12a
- RNN Dropout
- Dropout are applied to the weigths

# Why learn Swift for TensorFlow
- The Swift community 
## Questions to Chris
- Swift playground
- a swift tour google about it
- Goals are to substract on complexity
  - Avoid GPU for example
  - Works fast from top to botton
  - 

