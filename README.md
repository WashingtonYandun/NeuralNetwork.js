# NeuralNetwork.js

Simple neural network implemented from scratch in JavaScript

## Implementation

The neural network is implemented in the `NeuralNetwork` class. The class is initialized with the number of input nodes, hidden nodes, and output nodes. The class has a `train` method that takes in an input array and a target array and adjusts the weights and biases of the network accordingly. The class also has a `predict` method that takes in an input array and returns the output of the network.

The network uses the sigmoid activation function and the mean squared error loss function.

## Example

```javascript
const nn = new NeuralNetwork(2, 2, 1);

nn.train([1, 0], [1]);
nn.train([0, 1], [1]);
nn.train([1, 1], [0]);
nn.train([0, 0], [0]);

console.log(nn.predict([1, 0])); // 0.999
console.log(nn.predict([0, 1])); // 0.999
console.log(nn.predict([1, 1])); // 0.001
console.log(nn.predict([0, 0])); // 0.001
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
