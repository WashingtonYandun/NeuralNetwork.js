import { Matrix } from "./matrix.ai.js";
import { sigmoid, sigmoidDerivative } from "./activationFunctions.ai.js";

/**
 * Represents a neural network.
 */
export class NeuralNetwork {
    /**
     * Creates a new instance of the NeuralNetwork class.
     * @param {number} inputSize - The size of the input layer.
     * @param {number} hiddenSize - The size of the hidden layer.
     * @param {number} outputSize - The size of the output layer.
     */
    constructor(inputSize, hiddenSize, outputSize) {
        // Initialize weights and biases for input to hidden layer
        this.weightsInputHidden = new Matrix(hiddenSize, inputSize);
        this.weightsInputHidden.randomize();

        this.biasHidden = new Matrix(hiddenSize, 1);
        this.biasHidden.randomize();

        // Initialize weights and biases for hidden to output layer
        this.weightsHiddenOutput = new Matrix(outputSize, hiddenSize);
        this.weightsHiddenOutput.randomize();

        this.biasOutput = new Matrix(outputSize, 1);
        this.biasOutput.randomize();
    }

    /**
     * Performs a feedforward operation on the neural network.
     * @param {number[]} inputArray - The input array.
     * @returns {number[]} The output array.
     */
    feedforward(inputArray) {
        // Convert input array to a Matrix object
        const inputs = Matrix.fromArray(inputArray);

        // Calculate hidden layer output
        const hidden = Matrix.dot(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(sigmoid);

        // Calculate output layer output
        const output = Matrix.dot(this.weightsHiddenOutput, hidden);
        output.add(this.biasOutput);
        output.map(sigmoid);

        return output.toArray();
    }

    /**
     * Trains the neural network using backpropagation.
     * @param {number[]} inputArray - The input array.
     * @param {number[]} targetArray - The target array.
     */
    train(inputArray, targetArray) {
        // Convert input and target arrays to Matrix objects
        const inputs = Matrix.fromArray(inputArray);
        const targets = Matrix.fromArray(targetArray);

        // === Feedforward ===
        // Calculate hidden layer output
        const hidden = Matrix.dot(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(sigmoid);

        // Calculate output layer output
        const output = Matrix.dot(this.weightsHiddenOutput, hidden);
        output.add(this.biasOutput);
        output.map(sigmoid);

        // === Backpropagation ===
        // Calculate output layer errors
        const outputErrors = Matrix.subtract(targets, output);

        // Calculate output layer gradients
        const outputGradients = Matrix.map(output, sigmoidDerivative);
        outputGradients.multiply(outputErrors);
        outputGradients.multiply(0.1); // Learning rate

        // Calculate hidden layer errors
        const hiddenErrors = Matrix.dot(
            Matrix.transpose(this.weightsHiddenOutput),
            outputErrors
        );

        // Calculate hidden layer gradients
        const hiddenGradients = Matrix.map(hidden, sigmoidDerivative);
        hiddenGradients.multiply(hiddenErrors);
        hiddenGradients.multiply(0.1); // Learning rate

        // Update weights and biases
        this.weightsHiddenOutput.add(
            Matrix.dot(outputGradients, Matrix.transpose(hidden))
        );
        this.biasOutput.add(outputGradients);
        this.weightsInputHidden.add(
            Matrix.dot(hiddenGradients, Matrix.transpose(inputs))
        );
        this.biasHidden.add(hiddenGradients);
    }
}
