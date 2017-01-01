# js-NeuralNetwork
Simple Neural Network for ES6

## example
```javascript
let x = Tensor.from(MNIST_100.X).div(255, true).reshape([100, 28*28])
let y = Tensor.from(Util.toOnehot(MNIST_100.Y, 10)).reshape([100, 10])

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(28*28, 200))
net.addLayer(new Tanh())
net.addLayer(new FullyConnected(200, 10))
net.setLoss(new SoftmaxAndCrossEntropy())

net.fit(x, y, 0.2, 5, 5)    // (input, target, learning rate, num of epoch, batch size)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```

## example output
```
[Epoch 1] loss: 207.6040564700961
[Epoch 2] loss: 72.34003535239026
[Epoch 3] loss: 30.0361167822266
[Epoch 4] loss: 11.531669068266638
[Epoch 5] loss: 5.8979841276595835
Training Accuracy = 100%
```
