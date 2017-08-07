# js-NeuralNetwork
Simple Neural Network for javascript (ES6)

## Logical Or example
```javascript
let x = Tensor.from([0,0, 0,1, 1,0, 1,1]).reshape([4, 2]) // nSample X nFeature
let y = Tensor.from([0, 1, 1, 1]).reshape([4, 1]) // nSample X 1

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(2, 1))
net.setLoss(new SigmoidAndCrossEntropy())
net.setOptimizer(new Adam(1))

net.fit(x, y, 5, 2) // (input, target, num of epoch, batch size)
	
console.log('Prediction = ' + net.predict(x).toString())

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```

## MNIST example (MLP)
```javascript
let x = Tensor.from(MNIST_100.X).div(255, true).reshape([100, 28*28]) // nSample X nFeature
let y = Tensor.from(Util.toOnehot(MNIST_100.Y, 10)).reshape([100, 10]) // nSample X nCategory

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(28*28, 200))
net.addLayer(new Relu())
net.addLayer(new FullyConnected(200, 10))

net.setLoss(new SoftmaxAndCrossEntropy())
net.setOptimizer(new Adam(0.01))

net.fit(x, y, 5, 5) // (input, target, num of epoch, batch size)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```

## MNIST example (CNN)
```javascript
let x = Tensor.from(MNIST_100.X).div(255, true).reshape([100, 28, 28, 1]) // nSample X height X width X nChannel
let y = Tensor.from(Util.toOnehot(MNIST_100.Y, 10)).reshape([100, 10]) // nSample X nCategory

let net = new SequentialNetwork()
	
net.addLayer(new Convolution([28, 28, 1], [5, 5, 2])) // image shape, kernel shape
net.addLayer(new Relu())
net.addLayer(new Convolution([24, 24, 2], [5, 5, 3]))
net.addLayer(new Relu())
net.addLayer(new Reshape([20, 20, 3], [20*20*3]))
net.addLayer(new FullyConnected(20*20*3, 10))

net.setLoss(new SoftmaxAndCrossEntropy())
net.setOptimizer(new Adam(0.01))

net.fit(x, y, 5, 5)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```
