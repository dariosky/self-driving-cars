from sdcnnet.lenet import Lenet
from sdcnnet.mnist import mnist_pad_to_32, mnist_load

data = mnist_load()
mnist_pad_to_32(data)

n = Lenet(data, output_depth=10)
# n.train(save_as='lenet', EPOCHS=15)
n.load('lenet')
# n.test(load_from='')

# now let's try some prediction
xset, yset = data['train']
print("Correct output would be: ", yset[0])
print("Network predict:", n.predict([xset[0]])[0])


