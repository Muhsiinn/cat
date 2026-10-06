from neuron import Network

network = Network.load("./model/cat_2-64-32-16-1_1000.json")

chars = " .,:;irsXA253hMHGS#9B&@"

for y in range(0, 100, 2):
    for x in range(0, 100, 2):
        pred = network.forward([(x / 99), (y / 99)])[0]
        index = int(pred * (len(chars) - 1))
        print(chars[index], end="")
    print("")
