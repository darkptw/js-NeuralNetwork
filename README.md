# js-NeuralNetwork
Simple Neural Network for ES6

## example
```javascript
let x = Tensor.from(MNIST_100.X).reshape([100, 28*28])
let y = Tensor.from(Util.oneHot(MNIST_100.Y, 10)).reshape([100, 10])

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(28*28, 200))
net.addLayer(new Tanh())
net.addLayer(new FullyConnected(200, 10))
net.setLoss(new SoftmaxAndCrossEntropy())

net.fit(x, y, 0.05, 20, 5)    // (input, target, learning rate, num of epoch, batch size)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + accuracy)
```

## example output
```
[Epoch 1] loss: 223.96560394763947
[Epoch 2] loss: 195.08037796616554
[Epoch 3] loss: 185.85122828185558
[Epoch 4] loss: 176.40520545840263
[Epoch 5] loss: 171.53241415321827
[Epoch 6] loss: 164.3385956734419
[Epoch 7] loss: 158.15451497212052
[Epoch 8] loss: 154.40567430853844
[Epoch 9] loss: 149.41050691530108
[Epoch 10] loss: 144.006102707237
[Epoch 11] loss: 139.65008791536093
[Epoch 12] loss: 134.50506859272718
[Epoch 13] loss: 130.59612009860575
[Epoch 14] loss: 123.85574395768344
[Epoch 15] loss: 118.73275896534324
[Epoch 16] loss: 114.23433944769204
[Epoch 17] loss: 110.98669997043908
[Epoch 18] loss: 107.57127949222922
[Epoch 19] loss: 102.80002194084227
[Epoch 20] loss: 100.04465127177536
Training Accuracy = 0.8
```
