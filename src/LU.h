#pragma once
#include "Core.h"

namespace LinearMath {

    // Matrix inverse helper for fixed-size 2x2
    template<typename Scalar, int Options>
    inline Matrix<Scalar, 2, 2, Options> inverse2x2(const Matrix<Scalar, 2, 2, Options> &m) {
        return m.inverse();
    }

    // Matrix inverse helper for fixed-size 3x3
    template<typename Scalar, int Options>
    inline Matrix<Scalar, 3, 3, Options> inverse3x3(const Matrix<Scalar, 3, 3, Options> &m) {
        return m.inverse();
    }

} // namespace LinearMath