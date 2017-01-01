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
```
[Epoch 1] loss: 2.88749960064888
[Epoch 2] loss: 1.247955984203145
[Epoch 3] loss: 1.6632246683329868
[Epoch 4] loss: 1.4022898312612142
[Epoch 5] loss: 0.8140126581715776
Prediction = [
  [ 0.28458359837532043 ],
  [ 0.999606192111969 ],
  [ 0.9964053630828857 ],
  [ 0.999999463558197 ]
]
Training Accuracy = 100%
```

## MNIST example (MLP)
```javascript
let x = Tensor.from(MNIST_100.X).div(255, true).reshape([100, 28*28]) // nSample X nFeature
let y = Tensor.from(Util.toOnehot(MNIST_100.Y, 10)).reshape([100, 10]) // nSample X nCategory

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(28*28, 200))
net.addLayer(new Tanh())
net.addLayer(new FullyConnected(200, 10))

net.setLoss(new SoftmaxAndCrossEntropy())
net.setOptimizer(new Adam(0.01))

net.fit(x, y, 5, 5) // (input, target, num of epoch, batch size)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```
```
[Epoch 1] loss: 213.83740594188566
[Epoch 2] loss: 80.41895973801638
[Epoch 3] loss: 18.098858451086926
[Epoch 4] loss: 20.713332781404915
[Epoch 5] loss: 18.956331428320937
Training Accuracy = 100%
```

## MNIST example (CNN)
```javascript
let x = Tensor.from(MNIST_100.X).div(255, true).reshape([100, 28, 28, 1]) // nSample X height X width X nChannel
let y = Tensor.from(Util.toOnehot(MNIST_100.Y, 10)).reshape([100, 10]) // nSample X nCategory

let net = new SequentialNetwork()
	
net.addLayer(new Convolution([28, 28, 1], [5, 5, 2])) // image shape, kernel shape
net.addLayer(new Tanh())
net.addLayer(new Convolution([24, 24, 2], [5, 5, 3]))
net.addLayer(new Tanh())
net.addLayer(new Reshape([20, 20, 3], [20*20*3]))
net.addLayer(new FullyConnected(20*20*3, 10))

net.setLoss(new SoftmaxAndCrossEntropy())
net.setOptimizer(new Adam(0.01))

net.fit(x, y, 5, 5)

let accuracy = net.evaluate(x, y)
console.log('Training Accuracy = ' + (accuracy*100) + '%')
```
```
[Epoch 1] loss: 478.9218714556846
[Epoch 2] loss: 224.22441467690373
[Epoch 3] loss: 54.15575810527514
[Epoch 4] loss: 23.00490460639834
[Epoch 5] loss: 20.197968472316845
Training Accuracy = 99%
```
