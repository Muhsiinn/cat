from PIL import Image
from neuron import Network, loss
import sys

architectures = [
    [2, 16, 1],
    [2, 32, 1],
    [2, 64, 1],
    [2, 128, 1],
    [2, 32, 16, 1],
    [2, 64, 32, 1],
    [2, 128, 64, 1],
    [2, 64, 32, 16, 1],
    [2, 128, 64, 32, 1],
    [2, 256, 128, 1],
    [2, 512, 256, 1],
    [2, 1024, 512, 1],
]

index = int(sys.argv[1])
learning_rate = float(sys.argv[2])

arc = architectures[index]

epochs = [100, 200, 300, 500, 1000, 2000, 3000]

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


network = Network.create(arc)

for epoch in range(1, max(epochs) + 1):
    total_error = 0

    for inputs, target in training_data:
        pred = network.forward(inputs)[0]

        error = loss(pred, target)

        network.backward(target, learning_rate)

        total_error += error

        average_error = total_error / len(training_data)

        print(f"Architecture: {arc} | Epoch: {epoch} | Loss: {average_error}")
    if epoch in epochs:
        network.save(f"model/cat_{'-'.join(map(str, arc))}_{epoch}.json")
