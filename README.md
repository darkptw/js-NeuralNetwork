# js-NeuralNetwork
Simple Neural Network for ES6

## example
```javascript
//  logcial OR pattern
let x = MatrixF32.from([[0,0],[0,1],[1,0],[1,1]])  // 4 X 2 (nSample X featureDim)
let y = MatrixF32.from([[1,0],[0,1],[0,1],[0,1]])  // 4 X 2 (nSample X targetDim)

let net = new SequentialNetwork()

net.addLayer(new FullyConnected(2, 5))
net.addLayer(new Tanh())
net.addLayer(new FullyConnected(5, 2))
net.addLayer(new Tanh())

net.setLoss(new SoftmaxAndCrossEntropy())

net.fitWithSgd(x, y, 1, 50)

let p = net.predict(x)
console.log(p.toString())

let accuracy = net.evaluate(x, y)
console.log(accuracy)
```
