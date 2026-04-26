import numpy as np 
import os

# demonstrating softmax
# def softmax(n): 
#     e = np.exp(n)
#     s = np.sum(e)
#     return e / s


def softmax(n): 
    e = np.exp(n - np.max(n))
    s = np.sum(e)
    return e / s

def cross_entropy(p,q): 
    return -np.dot(p, np.log(q))


def relu(n): 
    return (n >= 0 )*n

def reluderiv(n): 
    return (n >= 0).astype(np.float64)

def xavier_uniform(input_count, output_count): 
    x = np.sqrt(6 / (input_count + output_count))
    return np.random.uniform(-x, x, (output_count, input_count))

def xavier_normal(input_count, output_count): 
    x = np.sqrt(2 / (input_count + output_count))
    return np.random.normal(-x, x, (output_count, input_count))
    
def one_hot_encode(y):
    r = []
    for i in range(10):
        if i == y: 
            r.append(1.)
        else: 
            r.append(0.)
            
    return np.array(r)

class NN:

    # ERRORED - IN ACTIVATED CALCULATION 
    
    def __init__(self, network_architecture): 
        
        self.layer_count = len(network_architecture)
        self.hidden_count = self.layer_count - 2
        self.losses = []
        self.arch = network_architecture
        
        self.weights = [] 
        self.biases = []
        self.activated = []
        
        
    def _init(self, arch): 
    
        # if we have totally N layers, then we have N - 1 weight matrices and bias vectors
        for i in range(self.layer_count-1): 
            self.weights.append(xavier_uniform(arch[i], arch[i+1]))
            self.biases.append(xavier_uniform(arch[i+1],1).reshape(arch[i+1],))

            
    def forward(self, X): 
        
        self.activated = [] # clear cache
        result = X / 255. # normalize data
        
        # multiply 
        for i in range(self.layer_count - 2): 
            result = relu(self.weights[i] @ result + self.biases[i])
            self.activated.append(result)
        
        # softmax at the last layer
        result = softmax(self.weights[self.layer_count - 2] @ result + self.biases[self.layer_count - 2])
        return result
    
    def backpropagation(self, X, y, prediction, learning_rate): 
        
        X = X / 255. # normalize data 

        # cross-entropy loss function 
        # L = - sum(Yi * ln(Pi)) where P -prediction and Y - real values
        loss = cross_entropy(y, prediction)
        
        self.losses.append(loss)

        delta = prediction - y
        last_activated = self.layer_count - 3
        last_weights = last_bias = self.layer_count - 2
        
        # update last layer first
        self.weights[last_weights] -= learning_rate * np.outer(delta, self.activated[last_activated])
        self.biases[last_bias] -= learning_rate * delta 
    
        
        # we have to update every weight and every bias 
        # so the iteration count is layer_count - 1 
        # note that self.activated does not contain last layers values(i.e. prediction)
        # and len(self.activated) = self.layer_count - 1
        # so, for weigths[i+1], we use activated[i]
        # it means for weights[j], we use activated[j-1]
        for i in range(last_weights - 1, 0, -1): # exclude first and last because they are updated differently
            delta = np.multiply(self.weights[i+1].T @ delta, reluderiv(self.activated[i]))
            self.weights[i] -= learning_rate*np.outer(delta, self.activated[i-1])
            self.biases[i] -= learning_rate * delta
              
        delta = np.multiply(self.weights[1].T @ delta, reluderiv(self.activated[0]))
        self.weights[0] -= learning_rate * np.outer(delta, X)
        self.biases[0] -= learning_rate * delta 
        
        
    def train(self, X, y, learning_rate, epochs): 

        if epochs == -1: # use all
            epochs = len(X)
        else:           
            assert epochs <= len(X), "not enough data for training"   

        assert len(X) == len(y), "different size of X and y!!"
        self._init(self.arch)
        
        for i in range(epochs):
#             ### 
#             #
#             print(f"Entering epoch {i}...")
#             #
#             ###
            try: 
                pred = self.forward(X[i])
                self.backpropagation(X[i], y[i], pred, learning_rate)
            except Exception as e: 
                print(f"Errored index: {i}")
                print(X[i])
                print(self.weights)
                break
            
    def predict(self , X): 
        
        if len(self.weights) == 0:
            raise AttributeError("The neural network is not trained or is not loaded!")
            
        return np.argmax(self.forward(X))
    
    def save_model(self, paramdir): 

        with open(os.path.join(paramdir, "info.txt"), "w") as archinfo: 
            for i in self.arch: 
                archinfo.write(str(i) + ", ")

        for i in range(self.layer_count - 1): 
            np.save(os.path.join(paramdir, f"W{i+1}"), self.weights[i])
            np.save(os.path.join(paramdir, f"b{i+1}"), self.biases[i])
            
    def load_model(self, paramdir): 
        
        self.weigths = []
        self.biases = []
        
        try: 
            for i in range(self.layer_count-1): 
                self.weights.append(np.load(os.path.join(paramdir, f"W{i+1}.npy")))
                self.biases.append(np.load(os.path.join(paramdir, f"b{i+1}.npy")))
                
        except Exception as e:
            print("Error occured! Invalid file count or names!")
        


def heinit(inp, out): 
    return np.random.normal(0,np.sqrt(2 / inp), (out, inp))

def sigmoid(x): 
    return 1 / (1 + np.exp(x - max(x)))

def sigmoidderiv(x): 
    return x*(1-x)

def leakyRelu(x): 
    a = np.multiply(x >= 0, x) + np.multiply(x < 0, x)*0.001
    return np.array(a) 
    
def leakyReluDeriv(x): 
    return (x >= 0).astype(np.float64) + (x < 0)*0.001


class EnhancedNN:
    
    def __init__(self, network_architecture, activation, activationderivative, init_strategy): 
        
        self.layer_count = len(network_architecture)
        self.hidden_count = self.layer_count - 2
        self.losses = []
        self.arch = network_architecture
        
        self.weights = [] 
        self.biases = []
        self.activated = []
        self.not_activated = []
        self.initializer = init_strategy
        self.activation = activation
        self.activationderiv = activationderivative
        
        
    def _init(self, arch): 
    
        # if we have totally N layers, then we have N - 1 weight matrices and bias vectors
        for i in range(self.layer_count-1): 
            self.weights.append(self.initializer(arch[i], arch[i+1]))
            self.biases.append(self.initializer(arch[i+1],1).reshape(arch[i+1],))

            
    def forward(self, X): 
        
        self.activated = [] 
        self.not_activated = [] # clear cache
        result = X / 255. # normalize data
        
        # multiply 
        for i in range(self.layer_count - 2): 
            result = self.weights[i] @ result + self.biases[i]
            self.not_activated.append(result)
            result = self.activation(result)
            self.activated.append(result)
        
        # softmax at the last layer
        result = softmax(self.weights[self.layer_count - 2] @ result + self.biases[self.layer_count - 2])
        return result
    
    def backpropagation(self, X, y, prediction, learning_rate): 
        
        X = X / 255. # normalize data 

        # cross-entropy loss function 
        # L = - sum(Yi * ln(Pi)) where P -prediction and Y - real values
        loss = cross_entropy(y, prediction)
        
        self.losses.append(loss)

        delta = prediction - y
        last_activated = self.layer_count - 3
        last_weights = last_bias = self.layer_count - 2
        
        # update last layer first
        self.weights[last_weights] -= learning_rate * np.outer(delta, self.activated[last_activated])
        self.biases[last_bias] -= learning_rate * delta 
    
        

        for i in range(last_weights - 1, 0, -1): 
            delta = np.multiply(self.weights[i+1].T @ delta, self.activationderiv(self.not_activated[i]))

            # ###########
            # # DELTA NORMALIZATION (A.K.A. gradient clipping) 
            # ########### 

            # for j in range(delta.shape[0]): 
            #     if delta[j] > 100: 
            #         delta[j] = 100
            #     elif delta[j] < -100: 
            #         delta[j] = -100 [ NOTE: DOES NOT IMPROVED PERFORMANCE AND EVEN IT DECREASED PERFORMANCE]

            self.weights[i] -= learning_rate*np.outer(delta, self.activated[i-1])
            self.biases[i] -= learning_rate * delta
              
        delta = np.multiply(self.weights[1].T @ delta, self.activationderiv(self.not_activated[0]))
        self.weights[0] -= learning_rate * np.outer(delta, X)
        self.biases[0] -= learning_rate * delta 
        
        
    def train(self, X, y, learning_rate, epochs, keepweights = False): 

        if epochs == -1: # use all
            epochs = len(X)
        else:           
            assert epochs <= len(X), "not enough data for training"   

        assert len(X) == len(y), "different size of X and y!!"
        
        if not keepweights: 
            self._init(self.arch)
        
        for i in range(epochs):

            try: 
                pred = self.forward(X[i])
                self.backpropagation(X[i], y[i], pred, learning_rate)
            except Exception as e: 
                print(f"Errored index: {i}")
                print(X[i])
                print(self.weights)
                break
            
    def predict(self , X): 
        
        if len(self.weights) == 0:
            raise AttributeError("The neural network is not trained or is not loaded!")
            
        return np.argmax(self.forward(X))
    
    def save_model(self, paramdir): 

        os.makedirs(paramdir, exist_ok=True)

        with open(os.path.join(paramdir, "info.txt"), "a") as archinfo: 

            s = []
            for i in self.arch: 
                s.append(str(i))

            s = ",".join(s)
            archinfo.write(s) 

        for i in range(self.layer_count - 1): 
            np.save(os.path.join(paramdir, f"W{i+1}"), self.weights[i])
            np.save(os.path.join(paramdir, f"b{i+1}"), self.biases[i])
            
    def load_model(self, paramdir): 
        
        self.weigths = []
        self.biases = []

        infofile = os.path.join(paramdir, "info.txt")

        with open(infofile, "rt") as info:
            self.arch = list(map(int,info.read().split(',')))
            self.layer_count = len(self.arch)
        
        try: 
            for i in range(self.layer_count-1): 
                self.weights.append(np.load(os.path.join(paramdir, f"W{i+1}.npy")))
                self.biases.append(np.load(os.path.join(paramdir, f"b{i+1}.npy")))
                
        except Exception as e:
            print("Error occured! Invalid file count or names!")

    @classmethod
    def load_from(cls, paramdir): 

        result = cls([-1], leakyRelu, leakyReluDeriv, heinit) # random architecture
        result.load_model(paramdir)
        return result
