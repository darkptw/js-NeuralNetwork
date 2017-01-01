# js-NeuralNetwork
Simple Neural Network for ES6

## Logical Or example
```javascript
let x = Tensor.from([0,0, 0,1, 1,0, 1,1]).reshape([4, 2])
let y = Tensor.from([0, 1, 1, 1]).reshape([4, 1])

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(2, 1))
net.setLoss(new SigmoidAndCrossEntropy())

net.fit(x, y, 1, 10, 4)    // (input, target, learning rate, num of epoch, batch size)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```

## Logical Or example output
```
[Epoch 1] loss: 2.936002790927887
[Epoch 2] loss: 1.4656985439360142
[Epoch 3] loss: 1.2658850327134132
[Epoch 4] loss: 1.1337947361171246
[Epoch 5] loss: 1.0291592609137297
[Epoch 6] loss: 0.9418610222637653
[Epoch 7] loss: 0.8677062541246414
[Epoch 8] loss: 0.8039499772712588
[Epoch 9] loss: 0.7485710242763162
[Epoch 10] loss: 0.7000356866046786
Training Accuracy = 100%
```

## MNIST example
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

## MNIST example output
```
[Epoch 1] loss: 207.6040564700961
[Epoch 2] loss: 72.34003535239026
[Epoch 3] loss: 30.0361167822266
[Epoch 4] loss: 11.531669068266638
[Epoch 5] loss: 5.8979841276595835
Training Accuracy = 100%
```
