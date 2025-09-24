import numpy as np


class GraphSpikeNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Basic sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.body_size = hidden_size + output_size
        # Indices for slicing
        self.input_idx = slice(0, input_size)
        self.hidden_idx = slice(input_size, input_size + hidden_size)
        self.output_idx = slice(input_size + hidden_size, input_size + hidden_size + output_size)
        # Parameters
        self.adj = np.zeros((hidden_size + output_size, input_size + hidden_size + output_size))
        self.w = np.random.uniform(-0.01, 0.01, (hidden_size + output_size, input_size + hidden_size + output_size))
        self.b = np.zeros((hidden_size + output_size, 1))

    def set_adjacency(self, adj):
        self.adj = adj
        self.w  *= adj # mask weights with adjacency

    def activation(self, x):
        return (x > 0).astype(float)

    def activation_prime(self, x):
        return (x > 0).astype(float) # ReLU derivative
    
    def _forward_iterate(self, state):
        z = self.w @ state + self.b # w: (3x5) * state (5x4) + (3, 1) = (3x4)
        a = self.activation(z) # f((3x4)) = (3x4)

        # integrate new h and o state into new state
        new_state = state.copy() # (5x4)
        new_state[self.input_size:, :] = a # (5x4)[3:4] = (5x4)

        return new_state, z

    def forward(self, X, iterations=2000):
        # initialize state of every neuron in the network
        batchsize = X.shape[1]
        state = np.zeros((self.body_size+self.input_size, batchsize)) # ([i+h+o]xt) (5x4)
        state[:self.input_size, :] = X # insert input

        states = []
        z_states = []

        for _ in range(iterations):
            noise = np.random.rand(*X.shape)
            x = (noise < X).astype(float)

            state[:self.input_size, :] = x # insert input
            state, z = self._forward_iterate(state)
            states.append(state)
            z_states.append(z)

        average_state = np.mean(states, axis=0)
        average_z = np.mean(z_states, axis=0)

        output = average_state[self.input_size + self.hidden_size:, :].T # extract output (txo) (4x1)
        return output, average_state, average_z

    def error(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2) # MSE
    

    def _backward_init(self, yHat, Y, state):
        # loss gradient
        dJ = yHat-Y # yHat(txo) - y (txo) = (4x1)
        # setup initial delta state
        delta_state = np.zeros_like(state) # (5x4)
        delta_state[self.input_size + self.hidden_size:, :] = dJ.T # (5x4)[1:4] <= (1x4)

        return delta_state

    def backward_step(self, delta_state, nr_inputs, states, z_states):
        delta = delta_state[nr_inputs:] * self.activation_prime(z_states)
        dw = (delta @ states.T) * self.adj # ((3x4) @ (4x5)) * (3x5) = (3x5)
        db = np.sum(delta, axis=1, keepdims=True) # (3x1)
        return self.w.T @ delta, dw, db
    
    def backward(self, yHat, Y, states, z_states, backiterations=2):
        delta_state = self._backward_init(yHat, Y, states)

        total_dw = np.zeros_like(self.w)
        total_db = np.zeros_like(self.b)
        for i in range(backiterations):
            delta_state, dw, db = self.backward_step(delta_state, self.input_size, states, z_states)
            total_dw += dw
            total_db += db

        return total_dw, total_db
    
    def update_params(self, dw, db, lr=0.01):
        self.w -= lr * dw
        self.b -= lr * db

    def train(self, X, Y, epochs, iterations=2, backiterations=2, lr=0.01):
        for epoch in range(epochs):
            yHat, states, z_states = self.forward(X, iterations)
            print("yHat:\n", yHat)
            print("Y:\n", Y)
            error = self.error(yHat, Y)
            dw, db = self.backward(yHat, Y, states, z_states, backiterations)
            self.update_params(dw, db, lr)
            print(f"Epoch {epoch+1}/{epochs}, Error: {error}")
        return error

if __name__ == "__main__":
    X = np.array([  
        [0.1, 0.1, 1, 1], # x1
        [0.1, 1, 0.1, 1], # x2
    ]) * 0.8# 4x2 (ixt)
    Y = np.array([[0.1],[1],[1],[0.1]])*0.8 # 4x1 (txo)

    adj = np.array([
        # x1, x2, h1, h2, o1 (3x5) ([h+o]x[i+h+o])
        [1, 1, 0, 0, 0], # h1
        [1, 1, 0, 0, 0], # h2
        [0, 0, 1, 1, 0], # o1
    ])

    gn = GraphSpikeNet(input_size=2, hidden_size=2, output_size=1)
    gn.set_adjacency(adj)
    
    gn.train(X, Y, epochs=10, iterations=2, backiterations=2, lr=0.01)

