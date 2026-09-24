# LinearMath

A lightweight, header-only C++17 linear algebra and numerical mathematics library designed for physics simulation, robotics, and geometric computing.

## Features

- **Dense Linear Algebra:** Fixed-size and dynamic vectors, matrices, and transformations (`Core.h`, `Dense.h`).
- **Decompositions:** Eigen-decomposition, SVD with inversion handling, polar decomposition (including stable high-strain formulations), and Jacobi rotations (`Decomposition.h`).
- **Linear Solvers:** Cholesky and LU decompositions with forward/back-substitution (`Cholesky.h`, `LU.h`).
- **Geometry & Kinematics:** Quaternion operations, skew-symmetric cross-product matrices, cotangent weights, and Delassus/kinematics operators ($K$, $G$, $Q$, $\hat{Q}$) (`Geometry.h`, `Matrix.h`).
- **Zero Dependencies:** Pure C++17 standard library implementation.

## Quick Start

### CMake FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
    LinearMath
    GIT_REPOSITORY https://github.com/Giannossk/LinearMath.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(LinearMath)

target_link_libraries(your_target PRIVATE LinearMath)
```

### Usage Example

```cpp
#include "LinearMath.h"
#include <iostream>

using namespace LinearMath;

int main() {
    Vector3r r(0.1, 0.2, 0.0);
    Matrix3r K;
    matrix::computeMatrixK(r, 1.0, Matrix3r::Identity(), K);

    Matrix3r R, S;
    decomposition::polarDecomposition(K, R, S);

    std::cout << "Rotation determinant: " << R.determinant() << std::endl;
    return 0;
}
```

## Building Tests

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).  
Copyright 2026 IOANNIS SIOKOS.
