# js-NeuralNetwork
Simple Neural Network for ES6

## Logical Or example
```javascript
let x = Tensor.from([0,0, 0,1, 1,0, 1,1]).reshape([4, 2])
let y = Tensor.from([0, 1, 1, 1]).reshape([4, 1])

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(2, 1))
net.setLoss(new SigmoidAndCrossEntropy())

net.fit(x, y, 3, 5, 2)
	
console.log('Prediction = ' + net.predict(x).toString())

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```
```
[Epoch 1] loss: 2.495827106758952
[Epoch 2] loss: 2.1905609320238
[Epoch 3] loss: 0.4799218690750422
[Epoch 4] loss: 0.24883715630858205
[Epoch 5] loss: 0.19734249857719988
Prediction = [
  [ 0.10700967907905579 ],
  [ 0.9676132798194885 ],
  [ 0.9830451607704163 ],
  [ 0.9999307990074158 ]
]
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
```
[Epoch 1] loss: 207.6040564700961
[Epoch 2] loss: 72.34003535239026
[Epoch 3] loss: 30.0361167822266
[Epoch 4] loss: 11.531669068266638
[Epoch 5] loss: 5.8979841276595835
Training Accuracy = 100%
```
