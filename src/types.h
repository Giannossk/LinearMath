#pragma once

#include <cmath>
#include <limits>

#include "Dense.h"

namespace LinearMath {

#ifdef USE_DOUBLE
    using Real = double;
#else
    using Real = float;
#endif

    inline constexpr Real RealMax = std::numeric_limits<Real>::max();
    inline constexpr Real RealMin = std::numeric_limits<Real>::lowest();

    using Vector2r = Matrix<Real, 2, 1, DontAlign>;
    using Vector3r = Matrix<Real, 3, 1, DontAlign>;
    using Vector4r = Matrix<Real, 4, 1, DontAlign>;
    using Vector5r = Matrix<Real, 5, 1, DontAlign>;
    using Vector6r = Matrix<Real, 6, 1, DontAlign>;
    using Matrix2r = Matrix<Real, 2, 2, DontAlign>;
    using Matrix3r = Matrix<Real, 3, 3, DontAlign>;
    using Matrix4r = Matrix<Real, 4, 4, DontAlign>;
    using Matrix6r = Matrix<Real, 6, 6, DontAlign>;
    using AlignedBox2r = AlignedBox<Real, 2>;
    using AlignedBox3r = AlignedBox<Real, 3>;
    using AngleAxisr = AngleAxis<Real>;
    using Quaternionr = Quaternion<Real, DontAlign>;

} // namespace LinearMath

// Compatibility macros for projects expecting REAL_MAX/REAL_MIN
#ifndef REAL_MAX
#define REAL_MAX (::LinearMath::RealMax)
#endif
#ifndef REAL_MIN
#define REAL_MIN (::LinearMath::RealMin)
#endif