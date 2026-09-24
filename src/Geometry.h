#pragma once
#include <limits>
#include "Core.h"

namespace LinearMath {

    // -----------------------------------------------------------------------------
    // AngleAxis
    // -----------------------------------------------------------------------------
    template<typename T>
    class AngleAxis {
    public:
    using Scalar = T;
    using Vector3 = Matrix<Scalar, 3, 1>;
    using Matrix3 = Matrix<Scalar, 3, 3>;

    AngleAxis() : m_angle(0), m_axis(1, 0, 0) {}

    AngleAxis(Scalar angle, const Vector3 &axis)
        : m_angle(angle), m_axis(axis.normalized()) {}

    Scalar angle() const { return m_angle; }
    Scalar& angle() { return m_angle; }

    const Vector3& axis() const { return m_axis; }
    Vector3& axis() { return m_axis; }

    Matrix3 toRotationMatrix() const {
        Matrix3 res;
        Scalar c = std::cos(m_angle);
        Scalar s = std::sin(m_angle);
        Scalar c1 = static_cast<Scalar>(1) - c;
        Scalar x = m_axis[0], y = m_axis[1], z = m_axis[2];

        res(0, 0) = c + x * x * c1;
        res(0, 1) = x * y * c1 - z * s;
        res(0, 2) = x * z * c1 + y * s;

        res(1, 0) = y * x * c1 + z * s;
        res(1, 1) = c + y * y * c1;
        res(1, 2) = y * z * c1 - x * s;

        res(2, 0) = z * x * c1 - y * s;
        res(2, 1) = z * y * c1 + x * s;
        res(2, 2) = c + z * z * c1;

        return res;
    }

    Matrix3 matrix() const { return toRotationMatrix(); }

    private:
    Scalar m_angle;
    Vector3 m_axis;
    };

    // -----------------------------------------------------------------------------
    // Quaternion
    // -----------------------------------------------------------------------------
    template<typename T, int Options = 0>
    class Quaternion {
    public:
    using Scalar = T;
    using Vector3 = Matrix<Scalar, 3, 1>;
    using Vector4 = Matrix<Scalar, 4, 1>;
    using Matrix3 = Matrix<Scalar, 3, 3>;

    // Default: identity rotation (x=0, y=0, z=0, w=1)
    Quaternion() : m_coeffs(0, 0, 0, 1) {}

    // w, x, y, z constructor (Hamilton convention: w is first parameter)
    Quaternion(Scalar w, Scalar x, Scalar y, Scalar z)
        : m_coeffs(x, y, z, w) {}

    explicit Quaternion(const Vector4 &coeffs)
        : m_coeffs(coeffs) {}

    Quaternion(const AngleAxis<Scalar> &aa) {
        Scalar halfA = aa.angle() * static_cast<Scalar>(0.5);
        Scalar s = std::sin(halfA);
        m_coeffs[0] = aa.axis()[0] * s;
        m_coeffs[1] = aa.axis()[1] * s;
        m_coeffs[2] = aa.axis()[2] * s;
        m_coeffs[3] = std::cos(halfA);
    }

    Quaternion(const Matrix3 &R) {
        Scalar tr = R(0, 0) + R(1, 1) + R(2, 2);
        if (tr > static_cast<Scalar>(0)) {
            Scalar s = std::sqrt(tr + static_cast<Scalar>(1)) * static_cast<Scalar>(2);
            m_coeffs[3] = static_cast<Scalar>(0.25) * s;
            m_coeffs[0] = (R(2, 1) - R(1, 2)) / s;
            m_coeffs[1] = (R(0, 2) - R(2, 0)) / s;
            m_coeffs[2] = (R(1, 0) - R(0, 1)) / s;
        } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
            Scalar s = std::sqrt(static_cast<Scalar>(1) + R(0, 0) - R(1, 1) - R(2, 2)) * static_cast<Scalar>(2);
            m_coeffs[3] = (R(2, 1) - R(1, 2)) / s;
            m_coeffs[0] = static_cast<Scalar>(0.25) * s;
            m_coeffs[1] = (R(0, 1) + R(1, 0)) / s;
            m_coeffs[2] = (R(0, 2) + R(2, 0)) / s;
        } else if (R(1, 1) > R(2, 2)) {
            Scalar s = std::sqrt(static_cast<Scalar>(1) + R(1, 1) - R(0, 0) - R(2, 2)) * static_cast<Scalar>(2);
            m_coeffs[3] = (R(0, 2) - R(2, 0)) / s;
            m_coeffs[0] = (R(0, 1) + R(1, 0)) / s;
            m_coeffs[1] = static_cast<Scalar>(0.25) * s;
            m_coeffs[2] = (R(1, 2) + R(2, 1)) / s;
        } else {
            Scalar s = std::sqrt(static_cast<Scalar>(1) + R(2, 2) - R(0, 0) - R(1, 1)) * static_cast<Scalar>(2);
            m_coeffs[3] = (R(1, 0) - R(0, 1)) / s;
            m_coeffs[0] = (R(0, 2) + R(2, 0)) / s;
            m_coeffs[1] = (R(1, 2) + R(2, 1)) / s;
            m_coeffs[2] = static_cast<Scalar>(0.25) * s;
        }
    }

    // Component accessors
    Scalar x() const { return m_coeffs[0]; }
    Scalar y() const { return m_coeffs[1]; }
    Scalar z() const { return m_coeffs[2]; }
    Scalar w() const { return m_coeffs[3]; }

    Scalar& x() { return m_coeffs[0]; }
    Scalar& y() { return m_coeffs[1]; }
    Scalar& z() { return m_coeffs[2]; }
    Scalar& w() { return m_coeffs[3]; }

    Vector4& coeffs() { return m_coeffs; }
    const Vector4& coeffs() const { return m_coeffs; }

    Vector3 vec() const { return Vector3(m_coeffs[0], m_coeffs[1], m_coeffs[2]); }

    // Conjugate & Inverse
    Quaternion conjugate() const {
        return Quaternion(m_coeffs[3], -m_coeffs[0], -m_coeffs[1], -m_coeffs[2]);
    }

    Quaternion inverse() const {
        Scalar sqNorm = squaredNorm();
        if (sqNorm > static_cast<Scalar>(1e-12)) {
            Scalar inv = static_cast<Scalar>(1) / sqNorm;
            return Quaternion(m_coeffs[3] * inv, -m_coeffs[0] * inv, -m_coeffs[1] * inv, -m_coeffs[2] * inv);
        }
        return conjugate();
    }

    Scalar norm() const {
        return m_coeffs.norm();
    }

    Scalar squaredNorm() const {
        return m_coeffs.squaredNorm();
    }

    void normalize() {
        m_coeffs.normalize();
    }

    Quaternion normalized() const {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    // Conversion to 3x3 rotation matrix
    Matrix3 toRotationMatrix() const {
        Matrix3 R;
        Scalar qx = m_coeffs[0], qy = m_coeffs[1], qz = m_coeffs[2], qw = m_coeffs[3];

        Scalar qx2 = qx * qx;
        Scalar qy2 = qy * qy;
        Scalar qz2 = qz * qz;

        R(0, 0) = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (qy2 + qz2);
        R(0, 1) = static_cast<Scalar>(2) * (qx * qy - qw * qz);
        R(0, 2) = static_cast<Scalar>(2) * (qx * qz + qw * qy);

        R(1, 0) = static_cast<Scalar>(2) * (qx * qy + qw * qz);
        R(1, 1) = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (qx2 + qz2);
        R(1, 2) = static_cast<Scalar>(2) * (qy * qz - qw * qx);

        R(2, 0) = static_cast<Scalar>(2) * (qx * qz - qw * qy);
        R(2, 1) = static_cast<Scalar>(2) * (qy * qz + qw * qx);
        R(2, 2) = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (qx2 + qy2);

        return R;
    }

    Matrix3 matrix() const { return toRotationMatrix(); }

    // Quaternion multiplication (Hamilton product)
    Quaternion operator*(const Quaternion &other) const {
        Scalar w1 = m_coeffs[3], x1 = m_coeffs[0], y1 = m_coeffs[1], z1 = m_coeffs[2];
        Scalar w2 = other.m_coeffs[3], x2 = other.m_coeffs[0], y2 = other.m_coeffs[1], z2 = other.m_coeffs[2];

        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        );
    }

    // Vector rotation: v' = q * v
    Vector3 operator*(const Vector3 &v) const {
        return toRotationMatrix() * v;
    }

    // Slerp
    Quaternion slerp(Scalar t, const Quaternion &other) const {
        Scalar cosTheta = m_coeffs.dot(other.m_coeffs);
        Quaternion target = other;

        if (cosTheta < static_cast<Scalar>(0)) {
            cosTheta = -cosTheta;
            target.m_coeffs = -target.m_coeffs;
        }

        if (cosTheta > static_cast<Scalar>(0.9995)) {
            // Linear interpolation for small angles
            Vector4 c = (static_cast<Scalar>(1) - t) * m_coeffs + t * target.m_coeffs;
            Quaternion res(c);
            res.normalize();
            return res;
        }

        Scalar theta = std::acos(cosTheta);
        Scalar sinTheta = std::sin(theta);
        Scalar w1 = std::sin((static_cast<Scalar>(1) - t) * theta) / sinTheta;
        Scalar w2 = std::sin(t * theta) / sinTheta;

        return Quaternion(w1 * m_coeffs + w2 * target.m_coeffs);
    }

    private:
    Vector4 m_coeffs; // stored as [x, y, z, w]
    };

    // -----------------------------------------------------------------------------
    // AlignedBox
    // -----------------------------------------------------------------------------
    template<typename T, int Dim>
    class AlignedBox {
    public:
    using Scalar = T;
    using VectorType = Matrix<Scalar, Dim, 1>;

    AlignedBox() {
        setEmpty();
    }

    AlignedBox(const VectorType &min, const VectorType &max)
        : m_min(min), m_max(max) {}

    void setEmpty() {
        m_min = VectorType::Constant(std::numeric_limits<Scalar>::max());
        m_max = VectorType::Constant(std::numeric_limits<Scalar>::lowest());
    }

    bool isEmpty() const {
        for (int i = 0; i < Dim; ++i) {
            if (m_min[i] > m_max[i]) return true;
        }
        return false;
    }

    const VectorType& min() const { return m_min; }
    VectorType& min() { return m_min; }

    const VectorType& max() const { return m_max; }
    VectorType& max() { return m_max; }

    void extend(const VectorType &p) {
        for (int i = 0; i < Dim; ++i) {
            if (p[i] < m_min[i]) m_min[i] = p[i];
            if (p[i] > m_max[i]) m_max[i] = p[i];
        }
    }

    bool contains(const VectorType &p) const {
        for (int i = 0; i < Dim; ++i) {
            if (p[i] < m_min[i] || p[i] > m_max[i]) return false;
        }
        return true;
    }


    private:
    VectorType m_min;
    VectorType m_max;
    };

    using Quaternionf = Quaternion<float>;
    using Quaterniond = Quaternion<double>;
    using AngleAxisf = AngleAxis<float>;
    using AngleAxisd = AngleAxis<double>;
    using AlignedBox2f = AlignedBox<float, 2>;
    using AlignedBox3f = AlignedBox<float, 3>;
    using AlignedBox2d = AlignedBox<double, 2>;
    using AlignedBox3d = AlignedBox<double, 3>;

} // namespace LinearMath