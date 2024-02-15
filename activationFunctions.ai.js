/**
 * Calculates the sigmoid activation function for a given input.
 * @param {number} x - The input value.
 * @returns {number} The result of the sigmoid function.
 */
export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

/**
 * Calculates the derivative of the sigmoid function.
 * @param {number} x - The input value.
 * @returns {number} The derivative of the sigmoid function.
 */
export function sigmoidDerivative(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}
