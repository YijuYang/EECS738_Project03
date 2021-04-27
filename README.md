# EECS738_Project03

## Project 3 - Says One Neuron To Another
###  Neural network architectures
##### 1. Set up a new git repository in your GitHub account
##### 2. Pick two datasets from
##### https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
##### 3. Choose a programming language (Python, C/C++, Java)
##### 4. Formulate ideas on how neural networks can be used to accomplish the task for the specific dataset
##### 5. Build a neural network to model the prediction process programmatically
##### 6. Document your process and results
##### 7. Commit your source code, documentation and other supporting files to the git repository in GitHub


## Dataset

#### Iris Dataset
The dimension of the input layer for the Iris Dataset was 4, since each iris item consists of four features: SepalLength, SepalWidth, PetalLength and PetalWidth. The output layer would be three dimensional since we are classifying flowers into 3 categories. To make it computational feasible,  we replace "Iris-setosa" with 0, 'Iris-versicolor' with 1, and 'Iris-virginica' with 2.

#### Seeds Dataset
The dimension of the input layer for the Seeds Dataset was 7, since each seed item consists of four features: Area, Perimeter, Compactness, Length of Kernel, Width of Kernel, Asymmetry Coefficient, Length of Kernel Groove. The output layer would be three dimensional since we are classifying seeds into three types of wheat (represented in the dataset as integers from 1 to 3). For consistency with our implementation, we subtracted 1 from each output integer such that 3 was mapped to 2, 2 mapped to 1, and 1 mapped to 0.


## Using Neural Network for Classification tasks
### Architecture & Method
We make a Neural Network with 2 layers. We add activation function after each layer. The we use softmax for the final classification result.

### Description
In this project, we utilize a simple neural network (more precisely, a multilayer perceptron, or MLP) to perform two classification tasks. This neural network consists of one input layer, one hidden layer and one output layer. Our datasets were the "Iris Dataset" and the "Seeds Dataset".
The input layer's dimension is equal to the dimension of the input data (amount of features we consider when classifying an item). The output layer's dimension is equal to the dimension fo the output data (amount of classes we classify our input into)
There is no best solution to determine the dimension of the hidden layer, which is usually determined empirically. According to ***Kolmogorov theorem***, we set it to the 2n + 1, where n is the dimension of input layer. The activation function we choose for hidden layer is ***tanh***.
To find best parameters for training model, difference between prediction and expected class needs to be minimized. We use ***cross-entropy loss*** to define this difference. To minimize the total loss, we employ ***batch gradient descent***. The back propagation algorithm will be used to calculate gradient efficiently.
We tried different values for learning rate and regularization strength. It turns out that when learning rate == 0.001 and regularization strength == 0.001, comparable decent results can be achieved.
Finally, to make sure the final probabilities is in range [0, 1], we apply activation function *softmax* on output layer.

### Training Process
**predict species of iris
iteration 0: 1.161197
iteration 1000: 0.083090
iteration 2000: 0.064597
iteration 3000: 0.056058
iteration 4000: 0.052731
iteration 5000: 0.049409
iteration 6000: 0.046738
iteration 7000: 0.044277
iteration 8000: 0.041716
iteration 9000: 0.038994
iteration 10000: 0.036507
iteration 11000: 0.034541
iteration 12000: 0.032955
iteration 13000: 0.031629
iteration 14000: 0.030536
iteration 15000: 0.029616
iteration 16000: 0.028807
iteration 17000: 0.028066
iteration 18000: 0.027358
iteration 19000: 0.026653
iteration 20000: 0.025924
iteration 21000: 0.025152
iteration 22000: 0.024324
iteration 23000: 0.023436
iteration 24000: 0.022490
iteration 25000: 0.021500
iteration 26000: 0.020483
iteration 27000: 0.019460
iteration 28000: 0.018451
iteration 29000: 0.017470

**predict species of seeds
iteration 0: 0.964791
iteration 1000: 0.214192
iteration 2000: 0.187125
iteration 3000: 0.150711
iteration 4000: 0.154773
iteration 5000: 0.147893
iteration 6000: 0.138293
iteration 7000: 0.131307
iteration 8000: 0.125016
iteration 9000: 0.118828
iteration 10000: 0.112600
iteration 11000: 0.106276
iteration 12000: 0.099677
iteration 13000: 0.092938
iteration 14000: 0.087784
iteration 15000: 0.082442
iteration 16000: 0.077271
iteration 17000: 0.072352
iteration 18000: 0.067694
iteration 19000: 0.063287
iteration 20000: 0.059114
iteration 21000: 0.055172
iteration 22000: 0.051460
iteration 23000: 0.047975
iteration 24000: 0.044709
iteration 25000: 0.041652
iteration 26000: 0.038797
iteration 27000: 0.036139
iteration 28000: 0.033672
iteration 29000: 0.030034
iteration 30000: 0.066752
iteration 31000: 0.061834
iteration 32000: 0.043230
iteration 33000: 0.041756
iteration 34000: 0.038683
iteration 35000: 0.056955
iteration 36000: 0.043557
iteration 37000: 0.039376
iteration 38000: 0.024839
iteration 39000: 0.022031
iteration 40000: 0.020822
iteration 41000: 0.020021
iteration 42000: 0.018765
iteration 43000: 0.268845
iteration 44000: 0.017013
