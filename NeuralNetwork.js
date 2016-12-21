class MatrixF32 {
    constructor(rows, cols, data) {
        this.rows = rows
        this.cols = cols
        this.data = data
    }

    static from(array2d) {
        let rows = array2d.length, cols = array2d[0].length
        let data = new Array(array2d.length)
        for(let i=0; i<array2d.length; ++i)
            data[i] = Float32Array.from(array2d[i])
        return new MatrixF32(rows, cols, data)
    }

    static zeros(rows, cols) {
        let data = new Array(rows)
        for(let i=0; i<rows; ++i)
            data[i] = new Float32Array(cols)
        return new MatrixF32(rows, cols, data)
    }

    get(row, col) {
        return this.data[row][col]
    }

    set(row, col, val) {
        this.data[row][col] = val
    }

    toString() {
        return '[\n  ' + this.data.map(row => '[ ' + row.join(', ') + ' ]').join('\n  ') + '\n]'
    }

    map(func, rightMat) {
        let mapped = new Array(this.rows)
        if(arguments.length == 1)
            for(let i=0; i<this.rows; ++i)
                mapped[i] = this.data[i].map(func)
        else
            for(let r=0; r<this.rows; ++r) {
                mapped[r] = new Float32Array(this.cols)
                for(let c=0; c<this.cols; ++c)
                    mapped[r][c] = func(this.data[r][c], rightMat.data[r][c])
            }
        return new MatrixF32(this.rows, this.cols, mapped)
    }

    add(right) {
        if(right instanceof MatrixF32)
            return this.map((a,b) => a+b, right)
        return this.map(e => e+right)
    }

    subtract(right) {
        if(right instanceof MatrixF32)
            return this.map((a,b) => a-b, right)
        return this.map(e => e-right)
    }

    multiply(right) {
        if(right instanceof MatrixF32)
            return this.map((a,b) => a*b, right)
        return this.map(e => e*right)
    }

    divide(right) {
        if(right instanceof MatrixF32)
            return this.map((a,b) => a/b, right)
        return this.map(e => e/right)
    }

    negate() {
        return this.map(e => -e)
    }

    power(exp) {
        return this.map(e => e**exp)
    }

    dot(rightMat) {
        let dotted = MatrixF32.zeros(this.rows, rightMat.cols)
        for(let r=0; r<this.rows; ++r)
            for(let c=0; c<rightMat.cols; ++c)
                for(let i=0; i<this.cols; ++i)
                    dotted.data[r][c] += this.data[r][i] * rightMat.data[i][c]
        return dotted
    }

    transpose() {
        let transposed = MatrixF32.zeros(this.cols, this.rows)
        for(let r=0; r<this.rows; ++r)
            for(let c=0; c<this.cols; ++c)
                transposed.data[c][r] = this.data[r][c]
        return transposed
    }

    repeatRow(count) {
        let repeated = new Array(this.rows * count)
        for(let i=0; i<count; ++i)
            for(let r=0; r<this.rows; ++r)
                repeated[i*this.rows + r] = this.data[r].slice()
        return new MatrixF32(repeated.length, this.cols, repeated)
    }

    repeatCol(count) {
        let repeated = new Array(this.rows)
        for(let r=0; r<this.rows; ++r) {
            repeated[r] = new Float32Array(this.cols * count)
            repeated[r].set(this.data[r])
            for(let i=1; i<count; ++i)
                repeated[r].copyWithin(i*this.cols, 0, this.cols)
        }
        return new MatrixF32(this.rows, repeated[0].length, repeated)
    }

    sum(axis = -1) {
        if(axis == -1) {
            let acc = 0
            this.data.forEach(row => acc = row.reduce((a,b) => a+b, acc))
            return acc
        }
        else if(axis == 0) {
            let acc = MatrixF32.zeros(1, this.cols)
            this.data.forEach(row => {
                for(let i=0; i<this.cols; ++i)
                    acc.data[0][i] += row[i]
            })
            return acc
        }
        else {
            let acc = MatrixF32.zeros(this.rows, 1)
            for(let i=0; i<this.rows; ++i)
                acc.data[i][0] = this.data[i].reduce((a,b) => a+b, 0)
            return acc
        }
    }

    argmax() {
        let maxIndices = new Int32Array(this.rows)
        for(let r=0; r<this.rows; ++r) {
            let maxValue = Number.MIN_VALUE
            for(let c=0; c<this.rows; ++c)
                if(this.data[r][c] > maxValue) {
                    maxValue = this.data[r][c]
                    maxIndices[r] = c
                }
        }
        return maxIndices
    }
}

class Layer {
    forward(input) {}
    backward(gradient) {}
    update(learningRate) {}
}

class FullyConnected extends Layer {
    constructor(inDim, outDim) {
        super()
        this.inDim = inDim
        this.outDim = outDim
        this.weight = MatrixF32.zeros(inDim, outDim).map(e => (Math.random()*2 - 1) / inDim)
        this.bias = MatrixF32.zeros(1, outDim).map(e => Math.random()*2 - 1)
    }

    forward(x) {
        this.x = x
        return x.dot(this.weight).add(this.bias.repeatRow(x.rows))
    }

    backward(g) {
        this.g = g
        return g.dot(this.weight.transpose())
    }

    update(lr) {
        this.bias = this.bias.subtract( this.g.multiply(lr).sum(0) )
        this.weight = this.weight.subtract( this.x.transpose().dot(this.g).multiply(lr) )
    }
}

class Sigmoid extends Layer {
    forward(x) {
        this.z = x.map(e => 1 / (1 + Math.exp(-e)))
        return this.z
    }

    backward(g) {
        return g.multiply( this.z.subtract(this.z.power(2)) )
    }
}

class Tanh extends Layer {
    forward(x) {
        this.z = x.map(e => Math.tanh(e))
        return this.z
    }

    backward(g) {
        return g.multiply( this.z.power(2).negate().add(1) )
    }
}

class Loss {
    forward(input, target) {}
    backward() {}
}

class SquaredError extends Loss {
    forward(x, y) {
        this.diff = x.subtract(y)
        return this.diff.power(2)
    }

    backward() {
        return this.diff
    }
}

class SoftmaxAndCrossEntropy extends Loss {
    forward(x, y) {
        let expX = x.map(e => Math.exp(e))
        let z = expX.divide( expX.sum(1).add(Number.EPSILON).repeatCol(expX.cols) )
        this.diff = z.subtract(y)
        return y.negate().multiply( z.map(e => Math.log(e)) )
    }

    backward() {
        return this.diff
    }
}

class SequentialNetwork {
    constructor() {
        this.layers = []
    }

    addLayer(layer) {
        this.layers.push(layer)
    }

    setLoss(loss) {
        this.loss = loss
    }

    fitWithSgd(input, target, learningRate, nEpoch, verbose=true) {
        for(let epoch=0; epoch<nEpoch; ++epoch) {
            let x = input, y = target

            this.layers.forEach(layer => x = layer.forward(x))
            let lossVal = this.loss.forward(x, y).sum()
            let g = this.loss.backward()
            this.layers.reverse().forEach(layer => g = layer.backward(g))
            this.layers.reverse().forEach(layer => layer.update(learningRate))

            if(verbose)
                console.log('[Epoch ' + epoch + '] loss: ' + lossVal)
        }
    }

    predict(input) {
        let x = input
        this.layers.forEach(layer => x = layer.forward(x))
        return x
    }

    evaluate(input, target) {
        let pi = this.predict(input).argmax()
        let ti = target.argmax()

        let correctCount = 0.0
        for(let i=0; i<pi.length; ++i)
            if(pi[i] == ti[i])
                ++correctCount

        return correctCount/input.rows
    }
}
