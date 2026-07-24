```
# Code for: Improved Convergence of Multilevel Moving Least-Squares
Approximation
```

```
This repository contains the code used to generate the numerical examples and
figures in Section 4 of the paper "Improved Convergence of Multilevel Moving
Least-Squares Approximation".
```

```
**Disclaimer:** This is unpolished, proof-of-concept research code intended
strictly for reproducing the specific examples in the paper. It is not
maintained as a general-purpose library. A generalized and fully documented
version is planned for future work.
```

# `## Prerequisites` 

```
To run this code, you will need:
```

```
* A C++ compiler (e.g., `g++`), ideally with OpenMP support. No external third-
party C++ libraries are required; the implementation uses only standard library
headers.
```

```
* A Python 3 environment with `numpy`, `matplotlib`, and `jupyter` installed.
```

# `## Compilation` 

```
The core algorithm calculating the multilevel MLS errors is written in C++ and
must be compiled before running the notebook. Open a Linux/WSL terminal in this
repository's directory and run:
```

```
```bash
```

```
g++ mls.cpp -o mls.bin -O3 -march=native -fopenmp
```
```

```
**Note on optimization flags:** The flags `-O3`, `-march=native`, and `-fopenmp`
are not strictly required for the code to compile or output correct results, but
the algorithm relies heavily on a parallel for-loop. Omitting these flags
(especially `-fopenmp`) will severely increase the runtime. If your compiler
does not support `-march=native`, you can omit it.
```

# `## Reproducing the Results` 

```
Once the binary (`mls.bin`) is compiled, the execution and visualization are
entirely handled by Python:
```

```
1. Launch Jupyter Notebook and open `mls_plots.ipynb`.
```

```
2. Run all cells. The notebook will automatically call the C++ binary with the
parameters set by the user and read the generated output file "errors.csv".
Instructions on how to recreate the figures and tables as they appear in the
manuscript are given in the notebook file. The target function and most
parameters can be customized.
```

