import numpy as np
import pandas as pd
from convolution import ConvolutionFirstLayer, ConvolutionInerLayer
from maxpool import MaxPool
from softmax import Softmax


data = pd.read_csv("train.csv").to_numpy()
train_data_num = 10000
test_data_num = 2000

train_data = data[:train_data_num, 1:].reshape(train_data_num, 28, 28)
train_labels = data[:train_data_num, :1].flatten()

test_data = data[train_data_num:(train_data_num+test_data_num),
                 1:].reshape(test_data_num, 28, 28)
test_labels = data[train_data_num:(
    train_data_num+test_data_num), : 1].flatten()


class Model:
    def __init__(self):
        self.conv1 = ConvolutionFirstLayer(4)    # 28x28x1 -> 24x24x4
        self.pool1 = MaxPool()                   # 24x24x4 -> 12x12x4
        self.conv2 = ConvolutionInerLayer(4, 4)  # 12x12x4 -> 8x8x4
        self.pool2 = MaxPool()                   # 8x8x4 -> 4x4x4
        self.softmax = Softmax(4 * 4 * 4, 10)    # 4x4x4 -> 10

    def forward(self, img, label):
        out = self.conv1.forward(2*(img / 255) - 1)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.pool2.forward(out)
        out = self.softmax.forward(out)

        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def backprop(self, img, label, lr=.005):
        out, loss, acc = self.forward(img, label)

        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool2.backprop(gradient)
        gradient = self.conv2.backprop(gradient, lr)
        gradient = self.pool1.backprop(gradient)
        gradient = self.conv1.backprop(gradient, lr)

        return loss, acc

    def train(self, train_data, train_labels):
        for epoch in range(10):
            print('--- Epoch %d ---' % (epoch + 1))

            permutation = np.random.permutation(len(train_data))
            train_data = train_data[permutation]
            train_labels = train_labels[permutation]

            loss = 0
            num_correct = 0
            for i, (img, label) in enumerate(zip(train_data, train_labels)):
                if i > 0 and i % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, loss / 100, num_correct)
                    )
                    loss = 0
                    num_correct = 0

                l, acc = self.backprop(img, label)
                loss += l
                num_correct += acc

    def test(self, test_data, test_labels):
        print('\n--- Testing the CNN ---')
        loss = 0
        num_correct = 0
        for im, label in zip(test_data, test_labels):
            _, l, acc = self.forward(im, label)
            loss += l
            num_correct += acc

        num_tests = len(test_data)
        print('Test Loss:', loss / num_tests)
        print('Test Accuracy:', num_correct / num_tests)


model = Model()
model.train(train_data, train_labels)
model.test(test_data, test_labels)
