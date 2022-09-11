# week 1
## lecture slides
- 1a 🗸
- 1b 🗸
- 1c 🗸
## lecture attendance, videos and notes
- Tuesday 🗸
- Thursday 🗸

## exercises
- perceptrons
- MLPs
- perceptrons and backprop
- probability
- probability 2
## video recordings & notes:
  - basic neuroanatomy (4:40) 🗸
  - McCulloch & Pitts model of a single Neuron (5:40) 🗸
  - Perceptrons by hand (14:23) 🗸
  - linear separability (8:27) 🗸
  - limitations of perceptrons and multi-layer perceptrons (?) 🗸
  - gradient descent (21:11) 🗸
  - training (12:20) 🗸
  - Hinton networks (7:12) 🗸
  - Probability (16.57) 🗸
  - entropy and KL divergence (15:15) 🗸
  - types of learning (6:53) 🗸
  - curve-fitting, over-fitting, Ockham's razor and generalisation (12:11) 🗸
  - how do we avoid over fitting? (6:17) 🗸
  - drop out (9:18) 🗸
## readings
  - Deep Learning (ch 1, 3, 3.10, 4.3, 5.2, 6.5.2, 7.11, 7.12) 
  - Deep Learning in Neural Networks: An Overview 
  - Nature Deep Learning Review 
  - learning representations by backpropagation 

---------------------------------------------------
# week 2
## lecture slides
- 2a: cross entropy, softmax, weight decay and momentum
- 2b: pytorch
- 2c: Hidden Unit Dynamics
## lecture attendance, videos and notes
- tuesday 🗸
- thursday 🗸
## exercises
- softmax and back backpropagation
- backprop variations
- gradient descent with numpy
- running pytorch
- XOR with PyTorch
- basic pytorch operations
- hidden unit dynamics
## video recordings & notes:
  - cross entropy (7:38) 🗸
  - softmax (3:55) 🗸
  - weight decay (6:36) 🗸
  - momentum (8:47) 🗸
  - adam () 🗸
  - pytorch (52:49) 
  - limitations of two layer sigmoidal neural networks (10:37) 🗸
## readings
- adam - a method for stochastic optimisation
- an overview of gradient descent
- Barkinok's Rational Functions
- learning to tell two spirals apart
- ruder.io
- deep learning (6.3, 5.2.2, 5.6.1, 8.3, 8.5)
------------------------------------------------------
# week 3: computer vision
## lecture slides
- 3a: Convolution 
- 3b: Image Processing

## lecture attendance, videos and notes
- tuesday 
- thursday
## exercises
- 
## video recordings & notes:
  - ()
  - ()
  - ()
  - ()
  - ()

## readings
- adam - a method for stochastic optimisation
- optimising gradient descent
- deep learning (chapters 3.13, 5.2.2, 5.5, 5.6.1, 6.3.2, 8.3, 8.5)
-------------------------------------------------------
# week 4: NLP
## lecture slides
- 4a: Recurrent Neural Networks
- 4b: Long Short Term Memory
- 4c: Word Vectors
- 4d: Language processing
## lecture attendance, videos and notes
- tuesday 
- thursday
## exercises
- 
## video recordings & notes:
  - ()
  - ()
  - ()
  - ()
  - ()
## readings
-------------------------------------------------------
# week 5: Reinforcement Learning
## lecture slides
- 5a: Reinforcement Learning
- 5b: Policy Learning & DeepRL
## lecture attendance, videos and notes
- tuesday 
- thursday

## exercises
- 

## video recordings & notes:
  - ()
  - ()
  - ()
  - ()
  - ()

## readings

-------------------------------------------------------
# week 6: Unsupervised Learning
## lecture slides
- 6a: autoencoders and adversarial training

## lecture attendance, videos and notes
- tuesday 
- thursday
## exercises
- 
## video recordings & notes:
  - ()
  - ()
  - ()
  - ()
  - ()
## readings

--------------------------------------------------------
# Thoughts
why do we not need to know standard deviation in derivations least squares?

what is hook's law?

what's the difference between shannon's entropy, boltzmann plank entropy, and 
Gibbs-Von Neumann Entropy?

what library do you use for augmentation in CV? I've mainly used albumentations.

it seems like transfer learning has become the norm. Is that your experience?

how do you decide what model to use in a particular application? I don't get to 
use DL at work yet, so my main deep learning is focused on kaggle, which i 
spend way too much time on. But i enjoy it and a person should have a hobby, so 
I think it's ok. Usually the community converges on a number of models, and it 
becomes an exercise in data engineering, fine-tuning and ensembling. 

I don't think i've seen a high scoring solution that uses a novel model built 
from scratch. 

Recently for CV i use models in the TIMM library. For nlp i use huggingface 
transformers. For tabular i usually use LGBM, XGBoost or catboost. 

A good result almost always requires an ensemble. How does that compare to your 
experience in academia and industry?

do you ever build a new model from scratch these days?

how do you optimise the weights in your ensemble? I see most of the top solutions
have tuned weights for their ensemble. I have tried a number of ways to do this but 
the result is always worse for me than naive ensembles. Obviously I'm missing 
something but i don't know what.

how do you fine tune your hyperparameters? I've been using optuna, ray and wandb
(weights and biases). I honestly think i have a lot of room for improvement here.
How do you tune HPs?

i wonder how i can use dropout to improve?

in terms of hardware, what do you use? I'm thinking mainly about GPU. 

I used to have an rtx2060 with 8GB vram. I still have that but i really only 
use it for debugging now. I upgraded to an rtx3090 which has generally been 
quite good, the exception has been NLP, where i have found that in some cases i 
can only run batch size = 1 before i run out of memory. I have tried renting GPU 
clusters on AWS and the training is MUCH faster, but of course there is an immediate 
cost and you don't have direct control of the hardware. I've found running in 
sagemaker can be a little limiting as it often doesn't have the very latest models.
I've only used AWS a couple of times, but when i do i dockerize my project rather 
than run on sagemaker. If i had a ton of money I'd buy a DGX or something.

I think its probably still the best option. Buying a 4 pack of a100s would be 
very expensive. I figure i can rent about 1000 hours of AWS time before it gets 
close to the cost of buying the GPUs. 

In Chapter 1, the textbook talks about emphasis having swung back to large labelled data sets.
Obviously the book was published in 2016 and things have changed.
My perception is that much of the recent progress has been made on large unlabelled datasets.
As an example, i would cite most of the large language models developed since 2017. 
GPT-3 for example was trained on 175 billion data points, most of which were from the common crawl.
Is my perception correct?

Chapter 1 also talks a lot about depth, and for the time that was certainly true.
However, it seems to me that more recently, deep learning has been dominated by transformer base architectures, which are actually quite shallow in comparison to CNNs.
Am i wrong?

Initially transformers were viewed as a language model, however more recently transformers have been achieving SOTA performance on CV and many other domains.
There are even some newer transformers like TFT (Temporal Fusion Transformers) from Google, which can deliver SOTA performance on time series. 
My understanding was that tabular data was one of the few domains where deep learning generally wasn't as reliable and robust as more classical techniques.
If indeed TFT achieves SOTA performance on time series this will be a major breakthrough for deep learning (I think). what do you think?

I know LSTMs are useful for time series, but the really successful application of LSTMs seems to usually be ensemble with some tree based model.

What do you make of the preponderance of transformer focused research? I feel like we're going to exhaust the marginal gains from this approach soon.
Do you think that we're approaching 'saturation'? ie the point at which we've exhausted a lot of the gains that can be achieved with the transformers approach?

At what point do you think we will reach diminishing marginal gains in terms of improving model performance by simply providing more training data?

I understand that transformers are quite 'shallow' compared with say CNNs. The key to their effectiveness is the 'self attention mechanism'.
Do you think that there is an opportunity to combine the self attention mechanism in transformers with the depth of CNNs to achieve a new wave of model architectures?

------
the book indicates that modern computer vision models don't require resizing, however from what i observe resizing is still a really common pre-processing step

