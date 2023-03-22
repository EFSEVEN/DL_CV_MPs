"""Neural network model."""

from typing import Sequence

import numpy as np

'''
https://haydar-ai.medium.com/learning-how-to-git-creating-a-longer-commit-message-16ca32746c3a
'''
class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            output_size: int,
            num_layers: int,
            optimizer: str
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            optimizer: Specify which param updating method will be used in training
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers


        self.m = {}  # = beta1*m + (1-beta1)g
        self.v = {}  # = bata2*v + (1-beta2)g^2
        self.t = 0
        
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            self.m["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.m["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
            
            self.v["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.v["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
        
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input in entire batch. Shape (N, D_i)
            b: the bias
        Returns:
            the output
        """
               
        return X @ W + b

    def linear_grad(self, W: np.ndarray) -> np.ndarray:
        return W.T

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return np.array(X > 0).astype(int)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # this is numerically stable
        return 1/(1 + np.exp(-x))
       

    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def mse(self, y: np.ndarray, p: np.ndarray) -> float:
        """ Parameters:
            y: ground-truth label    shape: (N, C)
            p: predicted label
        Returns:
            a SCALAR, the TOTAL mse loss of a batch of samples.
        """
        return np.mean(np.square(y - p)) / 2.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
            require_grad: If using forward in testing stage, no back prop,
                            so don't need store intermediate outputs for
                            computing gradients
        Returns:
            Matrix of shape (N, C)
        """
        self.outputs = {}
        # store the output of each layer in self.outputs
        # as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        """
        self.num_layers:        num of W&b sets
        len(self.hidden_sizes): num of levels of hidden neurons
           
        keys for outputs:
            linear(1-k), sigmoid(1-k), relu(1-(k-1))
            k = self.num_layers
           
        Example: size of self.params[W1] = (sizes[0], sizes[1]) = D, H_1
        self.outputs["<layer name>k"] stores the output value of k-th <layer-name>
        """
        out = X
        self.outputs["relu0"] = X
        
        for i in range(1,self.num_layers):
            
            out = self.linear(self.params["W" + str(i)], out, self.params["b" + str(i)])
            self.outputs["linear" + str(i)] = out
            
            out = self.relu(out)
            self.outputs["relu" + str(i)] = out
            
        
        out = self.linear(self.params["W" + str(self.num_layers)], out, self.params["b" + str(self.num_layers)])
        self.outputs["linear" + str(self.num_layers)] = out
            
        out = self.sigmoid(out)
        self.outputs["sigmoid"] = out
        
        return out

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets. shape: (N, C)
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # store the gradient of each parameter in self.gradients
        # as it will be used when updating each parameter and
        # during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        assert self.outputs is not None
        """
        keys for self.gradients:
            W1 ... Wk
            b1 ... bk
            k = self.num_layers
        """
        N = y.shape[0]
        n_layers, outputs, params = self.num_layers, self.outputs, self.params
        g = outputs["sigmoid"] - y  # g shape: (N, C)
        g = g*self.sigmoid_grad(outputs["sigmoid"])
        
        self.gradients["W"+str(n_layers)] = np.dot(outputs["relu"+str(n_layers-1)].T, g) / y.size
        self.gradients["b"+str(n_layers)] = np.sum(g, axis=0) / y.size ##############
        
        # pass back the gradient through each layer
        for i in range(n_layers-1, 0, -1):
            # till here, g shape = (N, H_i)  C is H_k.
            g = np.dot(g , params["W"+str(i+1)].T)#g @ self.linear_grad(params["W" + str(i)]) 
            
            g = g*self.relu_grad(outputs["linear"+str(i)]) #dw/d
            
            self.gradients["W" + str(i)] = np.dot(outputs["relu" + str(i - 1)].T , g)/y.size  # average g over the entire batch
            self.gradients["b" + str(i)] = np.sum(g, axis=0)/y.size  # from (N, H_i) to shape (H_i,)

        return self.mse(y, outputs["sigmoid"])

    def update(
            self,
            lr: float = 0.001,
            b1: float = 0.9,
            b2: float = 0.999,
            eps: float = 1e-8,
            opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for key, value in self.gradients.items():
                self.params[key] -= lr * value

        if opt == "Adam":

            self.t += 1
            
            t_m = 1 / (1 - b1 ** self.t)  # coefficients for bias correction
            t_v = 1 / (1 - b2 ** self.t)

            for key, gradient in self.gradients.items():
                self.m[key] = b1 * self.m[key] + (1 - b1) * self.gradients[key]
                self.v[key] = b2 * self.v[key] + (1 - b2) * (self.gradients[key] ** 2)

                self.params[key] -= lr * (self.m[key] * t_m) / (np.sqrt(self.v[key] * t_v) + eps)

        return
