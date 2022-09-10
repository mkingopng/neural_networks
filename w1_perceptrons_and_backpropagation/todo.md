# week 1
## lecture slides
- 1a 🗸
- 1b 🗸
- 1c 🗸
## lecture videos and notes
- Tuesday
- Thursday
## exercises
## video recordings & notes:
  - basic neuroanatomy (4:40) 🗸
  - McCulloch & Pitts model of a single Neuron (5:40) 🗸
  - Perceptrons by hand (14:23) 🗸
  - linear separability (8:27) 🗸
  - limitations of perceptrons and multi-layer perceptrons () 🗸
  - gradient descent (21:11) 🗸
  - training (12:20) 🗸
  - Hinton networks (7:12) 🗸
  - Probability (16.57) 🗸
  - ()
  - ()
  - ()
  - ()
  - ()
## readings
  - Deep Learning (ch 1, 3, 3.10, 4.3, 5.2, 6.5.2, 7.11, 7.12) 
  - Deep Learning in Neural Networks: An Overview 
  - Nature Deep Learning Review 
  - learning representations by backpropagation 

---------------------------------------------------
# week 2
## lecture slides
- 2a
- 2b
- 2c
## exercises
- 
## video recordings:
  - ()
  - ()
  - ()
  - ()
  - ()
## readings

------------------------------------------------------
# week 3
- lecture slides


# Thoughts

In Chapter 1, the textbook talks about emphasis having swung back to large labelled data sets.
Obviously the book was published in 2016 and things have changed.
My perception is that much of the recent progress has been made on large unlabelled datasets.
As an example, i would cite most of the large language models developed since 2017. 
GPT-3 for example was trained on 175 billion data points, most of which were from the common crawl.
Is my perception correct?

Chapter 1 also talks a lot about depth, and for the time that was certainly true.
However it seems to me that more recently, deep learning has been dominated by transformer base architectures, which are actually quite shallow in comparison to CNNs.
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

