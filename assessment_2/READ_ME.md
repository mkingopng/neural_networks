# Assessment 2
In this assignment, you will be implementing and training various neural 
network models for three different tasks, and analysing the results.You are to 
complete and submit three Python files `kuzu.py`, `spiral.py` and `encoder.py`, 
as well as a written report `a2.pdf` (in pdf format).

This assessment comprises 3 parts with multiple steps in each part. The steps 
require you to complete actions using the files provided and/or record your 
findings into the report. Ensure you read and complete each step as directed. 
Please post any questions in the corresponding Ed forum.You will be penalised 
5% per day of the marks available for this assessment task if you submit it 
after the due date, unless you have an approved extension through Special 
Consideration.

# Part 1 - Japanese character recognition (10 marks)
For Part 1 of the assignment, you will be implementing networks to recognize 
handwritten Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or 
KMNIST for short. The paper describing the dataset is available here. It is 
worth reading, but in short: significant changes occurred to the language when 
Japan reformed their education system in 1868, and the majority of Japanese 
today cannot read texts published over 150 years ago. This paper presents a 
dataset of handwritten, labeled examples of this old-style script (Kuzushiji).
You can find the paper [here](https://arxiv.org/pdf/1812.01718.pdf)

Along with this dataset, however, they also provide a much simpler one, 
containing 10 Hiragana characters with 7000 samples per class. This is the 
dataset we will be using. This part of the assessment comprises four steps:

**Step 1** (1 mark)

Implement a model `NetLin` which computes a linear function of the pixels in 
the image, followed by log softmax. In Python on your computer, run the code 
by typing: `python3 kuzu_main.py --net linCopy` the final accuracy and 
confusion matrix into your report. The final accuracy should be around 70%. 
Note that the rows of the confusionmatrix indicate the target character, while 
the columns indicate the one chosen by the network.

(0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha",6="ma", 7="ya", 8="re", 9="wo"). 

More examples of each character can be found 
[here](http://codh.rois.ac.jp/kmnist/index.html.en).

**Step 2** (1 mark)

Implement a fully connected 2-layer network NetFull (i.e. one hidden layer, 
plus the output layer), using tanh at the hiddenlayer and log softmax at the 
output layer. Run the code by typing:

`python3 kuzu_main.py --net full`

Try different values (multiples of 10) for the number of hidden nodes and try 
to determine a value that achieves high accuracy (at least 84%) on the test 
set. Copy the final accuracy and confusion matrix into your report.

**Step 3** (2 marks)

Implement a convolutional network called `NetConv`, with two 
convolutional layers plus one fully connected layer, all using `relu` 
activation function, followed by the output layer, using log softmax. **You are 
free to choose for yourself the number and size of the filters, meta parameter 
values (learning rate and momentum)**, and whether to use max pooling or a 
fully convolutional architecture. Run the code by typing:

`python3 kuzu_main.py --net conv`

Your network should consistently achieve at least 93% accuracy on the test set 
after 10 training epochs. Copy the final accuracy and confusion matrix into 
your report.

**Step 4** (6 marks)

Discuss what you have learned from this exercise, including the following 
points:
- the relative accuracy of the three models,
- the confusion matrix for each model: which characters are most likely to be 
mistaken for which other characters, and why?
- experiment with other architectures and/or meta-parameters for this dataset, 
and report on your results; 

The aim of this exercise is not only to achieve high accuracy but also to 
understand the effect of different choices on the final accuracy.

## Part 2 - Twin spirals (12 marks total)
For Part 2, you will be training on the famous Two Spirals Problem (Lang and 
Witbrock, 1988). The supplied code `spiral_main.py` loads the training data 
from `spirals.csv`, applies the specified model and produces a graph of the 
resulting function, along with the data. For this task there is no test set as 
such, but we instead judge the generalization by plotting the function computed 
by the network and making a visual assessment.This part of the assessment 
comprises six steps:

**Step 1** (1 mark)

Provide code for a Pytorch Module called `PolarNet` which operates as follows: 
- First, the input (x, y) is converted to polar co-ordinates `(r, a)` with 
`r = sqrt(x * x + y * y)`, `a = atan2(y, x)`.
- Next,`(r, a)` is fed into a fully connected neural network with one hidden 
layer using tanh activation, followed by a single output using sigmoid 
activation. The conversion to polar coordinates should be included in your 
`forward()` method, so that the Module performs the entire task of conversion 
followed by network layers.

**Step 2** (1 mark)

<img src="assignment 2 images/spiral.png"/>

In Python on your computer, run the code by typing:
`python3 spiral_main.py --net polar --hid 10`.

Try to find the minimum number of hidden nodes required so that this PolarNet 
learns to correctly classify all the training data within 20,000 epochs, on 
almost all runs. The `graph_output()` method will generate a picture of the 
function computed by your `PolarNet` called `polar_out.png`, which you should 
include in your report.

**Step 3** (1 mark)

Provide code for a Pytorch Module called `RawNet` which operates on the raw 
input `(x, y)` without converting to polar.

**Step 4** (1 mark)

In Python on your computer, run the code by typing:
`python3 spiral_main.py --net raw`. Try to choose a value for the number of 
hidden nodes (`--hid`) and the size of the initial weights (`--init`) such that 
this `RawNet` learns to correctly classify all the training data within 20,000 
epochs, on almost all runs. Include in your report the number of hidden nodes, 
and the values of any other meta-parameters. The `graph_output()` method will 
generate a picture of the function computed by your RawNet called 
`raw_out.png`, which you should include in your report.

**Step 5** (2 marks)

Using `graph_output()` as a guide, write a method called 
`graph_hidden (net, layer, node)` which plots the activation (after applying 
the tanh function) of the hidden node with the specified number (node) in the 
specified layer (1 or 2). (Note: if net is of type `PolarNet`, `graph_output()` 
only needs to behave correctly when layer is 1). 

**Hint**: you might need to modify `forward()` so that the hidden unit 
activations are retained, i.e. replace `hid1 = torch.tanh(...)` with 
`self.hid1 = torch.tanh(...)`. Use this code to generate plots of all the 
hidden nodes in `PolarNet`, and all the hidden nodes in both layers of `RawNet`, 
and include them in your report.

**Step 6** (6 marks)

Discuss what you have learned from this exercise, including the following 
points:

- The qualitative difference between the functions computed by the nodes in 
the hidden layer(s) in `PolarNet` and `RawNet`
- A brief description of how the network uses these functions to achieve the 
classification
- The effect of different values for initial weight size on the speed and 
success of learning for RawNet• Experiment with other changes and comment on 
the result - for example, changing batch size from 97 to 194, using SGD instead 
of Adam, changing `tanh` to `relu`, adding a third hidden layer, etc. 

The aim is to understand how different choices impact the final result.

## Part 3 - Hidden unit dynamics (8 marks total)

<img src="assignment 2 images/hidden dynamics.png"/>

In Part 3, you will be investigating hidden unit dynamics, as described in 
lesson 2, using the supplied code `encoder_main.py` and `encoder_model.py` as 
well as `encoder.py` (which you should modify and submit). This part comprises 
four steps:

**Step 1** (1 mark)

Run the code by typing: `python3 encoder_main.py --target=star16` 
Save the final image and include it in your report. Note that target is 
determined by the tensor star16 in `encoder.py`, which has 16 rows and 8 
columns, indicating that there are 16 inputs and 8 outputs. The inputs use a 
one-hot encoding and are generated in the form of an identity matrix using 
`torch.eye()`

**Step 2** (2 marks)
Run the code by typing `python3 encoder_main.py --target=input --dim=9 --plot`. 
In this case, the task is a 9-2-9 encoder

**Step 3** (2 marks)

Create by hand a dataset in the form of a tensor called `heart18` in the file 
`encoder.py` which, when run with the following command, will produce an image 
essentially the same as the heart-shaped figure shown below (but possibly 
rotated). Your tensor should have 18 rows and 14 columns. Include the final 
image in your report, and include the tensor `heart18` in your file 
`encoder.py` `python3 encoder_main.py --target=heart18`

<img src="assignment 2 images/heart.png"/>

**Step 4** (3 marks)

Create training data in tensors target1 and target2, which will generate two 
images of your own design, when run with the command 
`python3 encoder_main.py --target=target1` (and similarly for `target2`). You 
are free to choose the size of the tensors, and to adjust parameters such as 
`--epochs` and `--lr` in order to achieve successful learning. Marks will be 
awarded based on creativity and artistic merit. Include the final images in 
your report, and include the tensors target1 and target2 in your file 
`encoder.py`

## Presentation style/format
You will submit four files:
- A written report in pdf format (a2.pdf)
- `kuzu.py`
- `spiral.py`
- `encoder.py` 
- The written report should be comprised of your results and actions from the 
steps of each of the three parts of this assessment task. 

Before you submit please ensure you have clearly identified each action or 
response with appropriate sections, headings and subheadings.How to submit: 
Submit your pdf report and three Python files via Turnitin on this page. You 
may only submit a single file to each Turnitin part - use the tabs at the top 
of this page to navigate to each submission area for each file.

You may re-submit multiple times up to the due date. If you submit for the 
first time after the due date, you may only submit once (if you have submitted 
before the due date, you will be unable to re-submit after the due date). 
Normal late penalties apply for submission after the due date. 

Additional information may be found in the Ed discussion board and will be 
considered as part of the specification for the project. You should check that 
page regularly.

## Marking and feedback
You will receive feedback on your submission within five business days.

## marks summary
- KMNIST: 10 marks
- twin spirals: 12 marks
- hidden unit dynamics: 8 marks
**total**: 30 marks


## Useful Links
The following links have examples of other people's assignment submissions for 
inspiration. Some claim to have received good marks. The goal is to be better 
than all of them.

- [spirals](https://github.com/anantkm/IntertwinedSpirals)

- [KMNIST](https://github.com/anantkm/JapaneseCharacterRecognition)

- [ProductReviewPrediction](https://github.com/anantkm/productReviewPrediction)

- [COMP9444](https://github.com/raymondluk1995/COMP9444-Deep-Learning-Assignment-1)

- [COMP9444](https://github.com/gakkistyle/comp9444)

- [COMP9444](https://github.com/gavinpppeng/COMP9444)

- [COMP9444](https://github.com/z5208980/comp9444-our-model-is-training/blob/master/hw2main.py)

- [COMP9444](https://github.com/echushe/COMP9444)

- [COMP9444](https://github.com/kovidd/Neural-Networks)

- [COMP9444](https://github.com/chenquzhao/19T3_COMP9444)

# initial thoughts
- I need to look closely at how i can use data augmentation. That's been a strong 
focus in week 3. The focus has mainly been on `torchvision.transforms()`. There 
is no specific reason i couldn't use albumentations, but i think it might be 
wise to limit myself to torch functions. Its probably not wise to create the 
added complexity of having to install dependencies for the marker.

For **part 1** (KMNIST):
- I think i'll run through all the examples above, and test each one out. 
- identify what works and what doesn't, and aggregate the best features of each of the examples.
- then look at MNIST examples and see what I can learn/borrow from those.
- then finally look at what i think might be possibilities to further improve.

For **part 2** (the twin spirals), I think it will be a similar approach to part 1
- run examples for GitHub repos
- identify what works and what doesn't, and aggregate the best features of each of the examples.
- look at other similar examples and see what I can learn
- then finally look at what i think might be possibilities to further improve.

I suspect that **part 3** is going to be the more difficult... especially the last 
part, which calls for something novel. There are few novel things, as evidenced 
by the number of repos already publicly available about this specific subject 
and topics