from sdcnnet.lenet import Lenet
from sdcnnet.mnist import mnist_pad_to_32, mnist_load

data = mnist_load()
mnist_pad_to_32(data)

n = Lenet(data, output_depth=10)
n.train(save_as='lenet')
# n.test(load_from='lenet')
