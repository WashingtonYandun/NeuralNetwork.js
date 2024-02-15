/**
 * Represents a matrix and provides various matrix operations.
 */
export class Matrix {
    /**
     * Creates a new Matrix instance.
     * @param {number} rows - The number of rows in the matrix.
     * @param {number} cols - The number of columns in the matrix.
     */
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
    }

    /**
     * Creates a new Matrix instance from an array.
     * @param {number[]} arr - The array to create the matrix from.
     * @returns {Matrix} The new Matrix instance.
     */
    static fromArray(arr) {
        return new Matrix(arr.length, 1).map((elm, i, j) => arr[i]);
    }

    /**
     * Converts the matrix to a flat array.
     * @returns {number[]} The flat array representation of the matrix.
     */
    toArray() {
        return this.data.flat();
    }

    /**
     * Randomizes the values of the matrix.
     */
    randomize() {
        this.map(() => Math.random() * 2 - 1);
    }

    /**
     * Subtracts one matrix from another.
     * @param {Matrix} a - The matrix to subtract from.
     * @param {Matrix} b - The matrix to subtract.
     * @returns {Matrix} The resulting matrix after subtraction.
     */
    static subtract(a, b) {
        return new Matrix(a.rows, a.cols).map(
            (_, i, j) => a.data[i][j] - b.data[i][j]
        );
    }

    /**
     * Transposes a matrix.
     * @param {Matrix} matrix - The matrix to transpose.
     * @returns {Matrix} The transposed matrix.
     */
    static transpose(matrix) {
        return new Matrix(matrix.cols, matrix.rows).map(
            (_, i, j) => matrix.data[j][i]
        );
    }

    /**
     * Performs matrix multiplication between two matrices.
     * @param {Matrix} a - The first matrix.
     * @param {Matrix} b - The second matrix.
     * @returns {Matrix} The resulting matrix after multiplication.
     */
    static dot(a, b) {
        return new Matrix(a.rows, b.cols).map((_, i, j) => {
            let sum = 0;
            for (let k = 0; k < a.cols; k++) {
                sum += a.data[i][k] * b.data[k][j];
            }
            return sum;
        });
    }

    /**
     * Applies a function to each element of the matrix.
     * @param {Function} func - The function to apply.
     * @returns {Matrix} The matrix after applying the function.
     */
    map(func) {
        this.data = this.data.map((row, i) =>
            row.map((val, j) => func(val, i, j))
        );
        return this;
    }

    /**
     * Applies a function to each element of a matrix.
     * @param {Matrix} matrix - The matrix to apply the function to.
     * @param {Function} func - The function to apply.
     * @returns {Matrix} The resulting matrix after applying the function.
     */
    static map(matrix, func) {
        return new Matrix(matrix.rows, matrix.cols).map((val, i, j) =>
            func(matrix.data[i][j], i, j)
        );
    }

    /**
     * Multiplies the matrix by a scalar or another matrix.
     * @param {number|Matrix} n - The scalar or matrix to multiply by.
     * @returns {Matrix} The matrix after multiplication.
     */
    multiply(n) {
        if (typeof n === "number") {
            this.map((val) => val * n);
        } else {
            this.map((val, i, j) => val * n.data[i][j]);
        }
        return this;
    }

    /**
     * Adds another matrix to the current matrix.
     * @param {Matrix} matrix - The matrix to add.
     * @returns {Matrix} The matrix after addition.
     */
    add(matrix) {
        this.map((val, i, j) => val + matrix.data[i][j]);
        return this;
    }
}
