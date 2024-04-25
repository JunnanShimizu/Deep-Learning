# AIT Budapest - Deep Learning course
## Assignment 1.
## Created by: Bálint Gyires-Tóth

# Introduction

In this assignment, we focus on some concepts of neural networks and backpropagation. Neural networks are a fundamental component of deep learning, and understanding how they work is crucial for building and training effective models. Backpropagation is an algorithm used to train neural networks by adjusting the weights and biases of the network based on the error between the predicted and actual outputs. Through this assignment, we will delve into some details of the theory behind neural networks and backpropagation, and implement them in code to gain hands-on experience.

Please always write your anwser between the "------" lines.

Let's get started!

## Rules

Your final score will be penalized if you engage in the following behaviors:

1. 20% penalty for not using or following correctly the structure of this file.
2. 20% penalty for late submission within the first 24 hours. Late submissions after the first 24 hours will not be accepted.
3. 20% penalty for lengthy answers.
3. 40% penalty for for joint work or copying, including making the same, non-tipical mistakes, to all students concerned.

# Theory (50 points)
We have the following neural network architecture:

Input: 10 variables
Layer 1: Fully-connected layer with 16 neurons and sigmoid activation, with bias
Layer 2: Fully-connected layer with 1 neuron and sigmoid activation, no bias

The fully-connected layer is defined as s^(i+1) = a^(i)*W^(i), where

s^(i): output of the i-th fully-connected layer without activation.
a^(i): output of the activation function of the i-th layer.
a^(1): X (the input data).
W^(i): weight matrix of the i-th layer.

Use the notation as above. For partial deriavative use the @ sign. The cost function is MSE denoted by C, and the ground truth is y.

Question 1: Define the number of parameters in the neural network. (10 points)
------
Answer 1: 192
------

Question 2: Define the output of Layer 1 after the activation w.r.t. the input data. (10 points)
------
Answer 2: a^2 = sigmoid(np.dot(X, W^1) + b^1) # b^1 is bias for layer 1
------

Question 3: Define the output of Layer 2 after the activation w.r.t. the input data. (10 points)
------
Answer 3: a^3 = sigmoid(np.dot(a^2, W^2))
------

Question 4: Define the gradient of W^(2) w.r.t. the loss. (10 points)
------
Answer 4: @C / @W^(2) -> -(y - yhat) * (@yhat / s^(3)) * ((@a^(2)*w^(2))/(@w^(2)))
------

Question 5: Define the gradient of W^(1) w.r.t. the loss. (10 points)
------
Answer 5: @C / @W^(1) -> -(y - yhat) * (@yhat / s^(2)) * ((@X*w^(1))/(@w^(1)))
------

# Practice (50 points)

Please submit your work based on the shared notebook. Always test your solution in the shared notebook before submission. Only modify the specified code snippet.

Task 1: Complete the training loop with early stopping method. You can use any existing variable in the code if needed. (25 points)
------
Answer 6:

# Training loop for epochs times
valid_loss = []
prev_weights = model.weights
prev_valid_err = -1

for i in range(epochs):
# Training phase - sample by sample
train_err = 0
for k in range(X_train.shape[0]):
model.propagate_forward( X_train[k] )
train_err += model.propagate_backward( Y_train[k], lrate )
train_err /= X_train.shape[0]

# Validation phase
valid_err = 0
o_valid = np.zeros(X_valid.shape[0])
for k in range(X_valid.shape[0]):
o_valid[k] = model.propagate_forward(X_valid[k])
valid_err += (o_valid[k]-Y_valid[k])**2
valid_err /= X_valid.shape[0]
if prev_valid_err != -1:
valid_loss.append(round((valid_err - prev_valid_err), 5))
prev_valid_err = valid_err

if i % patience:
prev_weights = model.weights

if len(valid_loss) >= (patience - 1):
patience_vals = valid_loss[(len(valid_loss) - patience):]
if all(val >= 0.0 for val in patience_vals):
model.weights = prev_weights
print("STOPPING EARLY")
break
print("%d epoch, train_err: %.4f, valid_err: %.4f" % (i, train_err, valid_err))

# Notes: I rounded to 5 decimal places because it never seemed that the validation error got worse, so I set it so that if that the validation error stayed the same for epoch# to 5 decimal places, it would end.
------

Task 2: Complete the backpropagation algorithm with momentum method. You can use any existing variable in the code if needed. (25 points)
------
Answer 7:

momentum = 0.9
velocity = [np.zeros_like(w) for w in self.weights]

for i in range(len(self.weights)):
layer = np.atleast_2d(self.layers[i])
delta = np.atleast_2d(deltas[i])
dw = (-lrate*np.dot(layer.T,delta)) - (momentum * velocity[i])
self.weights[i] += dw
------