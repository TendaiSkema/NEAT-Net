import numpy as np
from keras.datasets import boston_housing

# -------------------
# Aktivierungen
# -------------------
class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return (x > 0).astype(float)

class Linear:
    def __call__(self, x):
        return x
    def derivative(self, x):
        return np.ones_like(x)

# -------------------
# Verlustfunktionen
# -------------------
class MSE:
    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(diff ** 2)
    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

# -------------------
# Optimizers
# -------------------
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, w, b, grad_w, grad_b, state):
        """
        w:        Gewichte (Vektor)
        b:        Bias (Skalar)
        grad_w:   Gradienten bzgl. Gewichte (Vektor, gleiche Form wie w)
        grad_b:   Gradienten bzgl. Bias (Skalar)
        state:    Dict mit mw, vw, mb, vb, t
        """
        # unpack
        mw, vw = state["mw"], state["vw"]
        mb, vb = state["mb"], state["vb"]
        t      = state["t"] + 1

        # 1st moment
        mw = self.beta1 * mw + (1 - self.beta1) * grad_w
        mb = self.beta1 * mb + (1 - self.beta1) * grad_b
        # 2nd moment
        vw = self.beta2 * vw + (1 - self.beta2) * (grad_w ** 2)
        vb = self.beta2 * vb + (1 - self.beta2) * (grad_b ** 2)

        # bias correction
        mw_hat = mw / (1 - self.beta1 ** t)
        mb_hat = mb / (1 - self.beta1 ** t)
        vw_hat = vw / (1 - self.beta2 ** t)
        vb_hat = vb / (1 - self.beta2 ** t)

        # parameter update
        w = w - self.lr * (mw_hat / (np.sqrt(vw_hat) + self.eps))
        b = b - self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))

        # repack
        state["mw"], state["vw"] = mw, vw
        state["mb"], state["vb"] = mb, vb
        state["t"] = t
        return w, b, state

# -------------------
# Neuron
# -------------------
class Neuron:
    def __init__(self, n_inputs, activation=ReLU()):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.activation = activation
        self.z = None
        self.a = None
        self.input = None

        # Adam-States (pro Parameter)
        self.opt_state = {
            "mw": np.zeros_like(self.weights),
            "vw": np.zeros_like(self.weights),
            "mb": 0.0,
            "vb": 0.0,
            "t": 0
        }

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a

    def backward(self, da, optimizer: Adam):
        # lokale Ableitung
        dz = da * self.activation.derivative(self.z)  # Skalar pro Sample
        # Gradienten
        dw = self.input * dz            # Vektor
        db = dz                         # Skalar
        dx = self.weights * dz          # Vektor (für vorherige Schicht)

        # Adam-Update
        self.weights, self.bias, self.opt_state = optimizer.update(
            self.weights, self.bias, dw, db, self.opt_state
        )
        return dx

# -------------------
# Layer
# -------------------
class Layer:
    def __init__(self, n_inputs, n_neurons, activation=ReLU()):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def forward(self, x):
        self.input = x
        return np.array([neuron.forward(x) for neuron in self.neurons])

    def backward(self, da_vec, optimizer: Adam):
        dx_total = np.zeros_like(self.input)
        for neuron, da in zip(self.neurons, da_vec):
            dx_total += neuron.backward(da, optimizer)
        return dx_total

# -------------------
# Network
# -------------------
class Network:
    def __init__(self, layer_sizes, loss=MSE(), optimizer=Adam(lr=0.01)):
        self.layers = []
        self.loss_fn = loss
        self.optimizer = optimizer
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], ReLU()))
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], Linear()))  # output linear

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_pred, y_true):
        dloss = self.loss_fn.derivative(y_pred, y_true)  # Skalar beim 1D-Output
        da = dloss
        for layer in reversed(self.layers):
            da = layer.backward(da, self.optimizer)

    def train(self, X, Y, epochs=100, lr=None):
        # optional: Lernrate zur Laufzeit ändern
        if lr is not None:
            self.optimizer.lr = lr

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            # einfacher SGD über einzelne Samples (Adam macht den Rest)
            for x, y in zip(X, Y):
                y_pred = self.forward(x)
                loss = self.loss_fn(y_pred, y)
                epoch_loss += loss
                self.backward(y_pred, y)
            losses.append(epoch_loss / len(X))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
        return losses

# -------------------
# Daten vorbereiten
# -------------------
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Features standardisieren
x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
x_train = (x_train - x_mean) / x_std
x_test  = (x_test  - x_mean) / x_std

# Targets normalisieren (hier: [0,1])
y_max = np.max(y_train)
y_train = y_train.reshape(-1, 1) / y_max
y_test  = y_test.reshape(-1, 1) / y_max

# -------------------
# Training mit Adam
# -------------------
net = Network([x_train.shape[1], 3, 1], loss=MSE(), optimizer=Adam(lr=0.01))
losses = net.train(x_train, y_train, epochs=200, lr=0.01)

# -------------------
# Test
# -------------------
preds = np.array([net.forward(x) for x in x_test])
mse_test = np.mean((preds - y_test)**2)
print("Test MSE:", mse_test)
