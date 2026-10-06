from PIL import Image
from neuron import Neuron, Layer, Network, loss, xavier_init, save_weights
import numpy as np


hidden = []
for _ in range(32):
    weights = [xavier_init(2, 32), xavier_init(2, 32)]
    hidden.append(Neuron(weights, 0))

output_weights = [xavier_init(32, 1) for _ in range(32)]

output_neuron = Neuron(output_weights, 0)
hidden_layer = Layer(hidden)
ouput_layer = Layer([output_neuron])
network = Network([hidden_layer, ouput_layer])
img = Image.open("cat.jpg")
grey = img.convert("L")

grey = grey.resize((100, 100))
pixels = grey.load()

training_data = []

for y in range(100):
    for x in range(100):
        input_data = [x / 99, y / 99]

        target = pixels[x, y] / 255

        training_data.append((input_data, target))

for epoch in range(100):
    total_error = 0
    for inputs, target in training_data:
        pred = network.forward(inputs)[0]
        error = loss(pred, target)
        network.backward(target, 0.01)
        total_error += error

    print(epoch, total_error / len(training_data))

save_weights(network, "cat_weights.json")

chars = " .:-=+*#%@"

for y in range(0, 100, 4):
    for x in range(0, 100, 4):
        pred = network.forward([x / 99, y / 99])[0]

        index = int(pred * (len(chars) - 1))
        print(chars[index], end="")

    print()
