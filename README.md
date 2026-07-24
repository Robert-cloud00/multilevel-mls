## Prerequisites

To run this code, you will need:

* **C++ Compiler:** A modern C++ compiler (e.g., `g++`) with OpenMP support. No external third-party C++ libraries are required; the implementation uses only standard C++ library headers.
* **Python 3:** An environment with `numpy`, `matplotlib`, and `jupyter` installed.

## Compilation

The core algorithm calculating the multilevel MLS errors is written in C++ and must be compiled before running the notebook. Open a Linux/WSL terminal in this repository's directory and run:

```bash
g++ mls.cpp -o mls.bin -O3 -march=native -fopenmp
```

> **Note on optimization flags:** The flags `-O3`, `-march=native`, and `-fopenmp` are not strictly required for the code to compile or output correct results, but the algorithm relies heavily on parallel loops. Omitting these flags (especially `-fopenmp`) will severely increase runtime. If your compiler does not support `-march=native`, you can safely omit it.

## Reproducing the Results

Once the binary (`mls.bin`) is compiled, execution and visualization are entirely handled by Python:

1. Launch Jupyter Notebook and open `mls_plots.ipynb`.
2. Run all cells. The notebook automatically calls the C++ binary with user-specified parameters and reads the generated `errors.csv` output file.

Instructions on how to recreate the figures and tables as they appear in the manuscript are provided directly within the notebook. The target function and key parameters can all be customized.
