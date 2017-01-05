from sdcnnet.lenet import Lenet
from sdcnnet.mnist import shuffle, mnist_pad_to_32, mnist_load
from sdcnnet.net import NNet

data = mnist_load()
mnist_pad_to_32(data)
shuffle(data['train'])

n = NNet(Lenet, data, output_depth=10)
n.train(save_as='lenet')
# n.test(load_from='lenet')
