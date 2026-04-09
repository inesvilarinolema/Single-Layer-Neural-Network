# Single-Layer Neural Network for Digit Recognition

## About The Project
This repository contains a **Single-Layer Neural Network** implemented entirely from scratch in **C**. 

The objective of this algorithm is to recognize and classify four distinct digit patterns ("1", "4", "7", and "2") represented as 4x3 pixel grids.
Instead of using modern high-level libraries, this project explores the mathematical foundations of machine learning, focusing on linear separability, cumulative error calculation, and weight optimization through custom learning rules.

## Key Features
* **Visual Pattern Classification:** Processes 12-node inputs (a flattened 4x3 pixel grid) to identify specific numeric shapes.
* **Binary Encoding:** Maps the identified digits into a unique 2-bit binary code using 2 output neurons.
* **Generalization Testing:** Includes custom noisy/distorted inputs to test the network's ability to classify unseen data based on structural similarities.

## Architecture & Mathematics
* **Input Layer:** 12 nodes + 1 Bias term.
* **Output Layer:** 2 neurons.
* **Activation Function:** Unipolar Sigmoid (mapping outputs to the `[0, 1]` range).
* **Learning Rule:** Weights are adjusted proportionally to the error and the derivative of the activation function.
* **Learning Rate ($\alpha$):** 0.8.
* **Stopping Condition:** Cumulative Mean Squared Error threshold of `0.0005`.

## How to Compile and Run
The project uses standard C libraries (`<stdio.h>` and `<math.h>`). Link the math library using `-lm` when compiling.

```bash
# Compile the code
gcc exercise1.c -o digit_nn -lm

# Run the executable
./digit_nn
```

## Results & Verification
The network successfully learns to classify the digits, proving that the 12-pixel representations of 1, 4, 7, and 2 are linearly separable in the weight space. 

The cumulative error decreases consistently, reaching convergence automatically.

**Verification Output:**
```text
---CHECKING RESULTS---
Digit 1 | Target: 0 0 | Output: 0.0064 0.0124
Digit 4 | Target: 1 0 | Output: 0.9863 0.0091
Digit 7 | Target: 0 1 | Output: 0.0135 0.9856
Digit 2 | Target: 1 1 | Output: 0.9880 0.9990
```

