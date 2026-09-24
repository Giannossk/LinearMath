#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <iostream>

namespace LinearMath {

    enum {
        DontAlign = 0,
        AutoAlign = 0,
        ColMajor = 0,
        RowMajor = 1
    };

    enum ComputationInfo {
        Success = 0,
        NumericalIssue = 1,
        NoConvergence = 2,
        InvalidInput = 3
    };

    using Index = std::ptrdiff_t;
    constexpr int Dynamic = -1;

    // Forward declarations
    template<typename T, int Rows, int Cols, int Options = 0, int MaxRows = Rows, int MaxCols = Cols>
    class Matrix;

    template<typename MatrixTypeT>
    class LLT;

    template<typename MatrixTypeT>
    class LDLT;

    template<typename Derived, int BlockRows, int BlockCols>
    class Block;

    // -----------------------------------------------------------------------------
    // Block Expression
    // -----------------------------------------------------------------------------
    template<typename Derived, int BlockRows, int BlockCols>
    class Block {
    public:
        using Scalar = typename Derived::Scalar;
        static constexpr int RowsAtCompileTime = BlockRows;
        static constexpr int ColsAtCompileTime = BlockCols;

        Block(Derived &mat, int startRow, int startCol)
            : m_mat(mat), m_startRow(startRow), m_startCol(startCol) {}

        constexpr int rows() const { return BlockRows; }
        constexpr int cols() const { return BlockCols; }
        constexpr int size() const { return BlockRows * BlockCols; }

        Scalar& operator()(int r, int c) {
            return m_mat(m_startRow + r, m_startCol + c);
        }
        const Scalar& operator()(int r, int c) const {
            return m_mat(m_startRow + r, m_startCol + c);
        }

        Scalar& operator[](int i) {
            if constexpr (BlockCols == 1) return (*this)(i, 0);
            else return (*this)(0, i);
        }
        const Scalar& operator[](int i) const {
            if constexpr (BlockCols == 1) return (*this)(i, 0);
            else return (*this)(0, i);
        }

        Scalar& operator()(int i) { return (*this)[i]; }
        const Scalar& operator()(int i) const { return (*this)[i]; }

        // Conversion to concrete Matrix
        operator Matrix<Scalar, BlockRows, BlockCols>() const {
            Matrix<Scalar, BlockRows, BlockCols> res;
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    res(r, c) = (*this)(r, c);
                }
            }
            return res;
        }

        Matrix<Scalar, BlockRows, BlockCols> eval() const {
            return static_cast<Matrix<Scalar, BlockRows, BlockCols>>(*this);
        }

        Matrix<Scalar, BlockCols, BlockRows> transpose() const {
            Matrix<Scalar, BlockCols, BlockRows> res;
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    res(c, r) = (*this)(r, c);
                }
            }
            return res;
        }

        template<typename OtherDerived>
        Block& operator=(const OtherDerived &other) {
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    (*this)(r, c) = other(r, c);
                }
            }
            return *this;
        }

        template<typename OtherDerived>
        Block& operator+=(const OtherDerived &other) {
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    (*this)(r, c) += other(r, c);
                }
            }
            return *this;
        }

        template<typename OtherDerived>
        Block& operator-=(const OtherDerived &other) {
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    (*this)(r, c) -= other(r, c);
                }
            }
            return *this;
        }

        template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
        Block& operator*=(T s) {
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    (*this)(r, c) *= static_cast<Scalar>(s);
                }
            }
            return *this;
        }

        template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
        Block& operator/=(T s) {
            Scalar inv = static_cast<Scalar>(1) / static_cast<Scalar>(s);
            return (*this) *= inv;
        }

        void setZero() {
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    (*this)(r, c) = Scalar(0);
                }
            }
        }

        void setIdentity() {
            setZero();
            constexpr int minDim = (BlockRows < BlockCols) ? BlockRows : BlockCols;
            for (int i = 0; i < minDim; ++i) {
                (*this)(i, i) = Scalar(1);
            }
        }

        Scalar squaredNorm() const {
            Scalar sum = 0;
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    Scalar val = (*this)(r, c);
                    sum += val * val;
                }
            }
            return sum;
        }

        Scalar norm() const {
            return std::sqrt(squaredNorm());
        }

        void normalize() {
            Scalar n = norm();
            if (n > static_cast<Scalar>(1e-12)) {
                (*this) *= (static_cast<Scalar>(1) / n);
            }
        }

        Matrix<Scalar, BlockRows, BlockCols> normalized() const {
            Matrix<Scalar, BlockRows, BlockCols> res = *this;
            res.normalize();
            return res;
        }

        template<typename Other>
        Scalar dot(const Other &other) const {
            Scalar sum = 0;
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    sum += (*this)(r, c) * other(r, c);
                }
            }
            return sum;
        }

        template<typename Other, int R = BlockRows, int C = BlockCols,
                 typename = typename std::enable_if<R == 3 && C == 1>::type>
        Matrix<Scalar, 3, 1> cross(const Other &other) const {
            return Matrix<Scalar, 3, 1>(
                (*this)[1] * other[2] - (*this)[2] * other[1],
                (*this)[2] * other[0] - (*this)[0] * other[2],
                (*this)[0] * other[1] - (*this)[1] * other[0]
            );
        }

        Scalar& x() { return (*this)[0]; }
        const Scalar& x() const { return (*this)[0]; }
        Scalar& y() { return (*this)[1]; }
        const Scalar& y() const { return (*this)[1]; }
        Scalar& z() { return (*this)[2]; }
        const Scalar& z() const { return (*this)[2]; }
        Scalar& w() { return (*this)[3]; }
        const Scalar& w() const { return (*this)[3]; }

        Matrix<Scalar, BlockRows, BlockCols> operator-() const {
            Matrix<Scalar, BlockRows, BlockCols> res;
            for (int c = 0; c < BlockCols; ++c) {
                for (int r = 0; r < BlockRows; ++r) {
                    res(r, c) = -(*this)(r, c);
                }
            }
            return res;
        }

        template<int SubR, int SubC>
        Block<Derived, SubR, SubC> block(int r, int c) {
            return Block<Derived, SubR, SubC>(m_mat, m_startRow + r, m_startCol + c);
        }

        template<int SubR, int SubC>
        Block<const Derived, SubR, SubC> block(int r, int c) const {
            return Block<const Derived, SubR, SubC>(m_mat, m_startRow + r, m_startCol + c);
        }

    private:
        Derived &m_mat;
        int m_startRow;
        int m_startCol;
    };


    // -----------------------------------------------------------------------------
    // Matrix Class Template
    // -----------------------------------------------------------------------------
    template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class Matrix {
    public:
        using Scalar = T;
        static constexpr int RowsAtCompileTime = Rows;
        static constexpr int ColsAtCompileTime = Cols;
        static constexpr int SizeAtCompileTime = Rows * Cols;

        // Constructors & Destructor
        Matrix() {
            setZero();
        }

        ~Matrix() {
            if constexpr (Rows == Cols) {
                delete m_lltCache;
                delete m_ldltCache;
            }
        }

        Matrix(const Matrix &other) {
            for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] = other.m_data[i];
            m_lltCache = nullptr;
            m_ldltCache = nullptr;
        }

        Matrix& operator=(const Matrix &other) {
            if (this != &other) {
                for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] = other.m_data[i];
                if constexpr (Rows == Cols) {
                    delete m_lltCache; m_lltCache = nullptr;
                    delete m_ldltCache; m_ldltCache = nullptr;
                }
            }
            return *this;
        }

        Matrix(Matrix &&other) noexcept {
            for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] = other.m_data[i];
            m_lltCache = other.m_lltCache;
            m_ldltCache = other.m_ldltCache;
            other.m_lltCache = nullptr;
            other.m_ldltCache = nullptr;
        }

        Matrix& operator=(Matrix &&other) noexcept {
            if (this != &other) {
                for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] = other.m_data[i];
                if constexpr (Rows == Cols) {
                    delete m_lltCache;
                    delete m_ldltCache;
                }
                m_lltCache = other.m_lltCache;
                m_ldltCache = other.m_ldltCache;
                other.m_lltCache = nullptr;
                other.m_ldltCache = nullptr;
            }
            return *this;
        }


        // Vector constructors
        template<int R = Rows, int C = Cols, typename = typename std::enable_if<R * C == 2>::type>
        Matrix(Scalar x, Scalar y) {
            m_data[0] = x;
            m_data[1] = y;
        }

        template<int R = Rows, int C = Cols, typename = typename std::enable_if<R * C == 3>::type>
        Matrix(Scalar x, Scalar y, Scalar z) {
            m_data[0] = x;
            m_data[1] = y;
            m_data[2] = z;
        }

        template<int R = Rows, int C = Cols, typename = typename std::enable_if<R * C == 4>::type>
        Matrix(Scalar x, Scalar y, Scalar z, Scalar w) {
            m_data[0] = x;
            m_data[1] = y;
            m_data[2] = z;
            m_data[3] = w;
        }

        // Copy / conversion constructors
        template<typename OtherScalar, int OtherOptions>
        Matrix(const Matrix<OtherScalar, Rows, Cols, OtherOptions> &other) {
            for (int i = 0; i < SizeAtCompileTime; ++i) {
                m_data[i] = static_cast<Scalar>(other.data()[i]);
            }
        }

        template<typename OtherDerived, int BR, int BC>
        Matrix(const Block<OtherDerived, BR, BC> &block) {
            static_assert(BR == Rows && BC == Cols, "Block dimensions must match Matrix dimensions");
            for (int c = 0; c < Cols; ++c) {
                for (int r = 0; r < Rows; ++r) {
                    (*this)(r, c) = block(r, c);
                }
            }
        }

        template<typename OtherDerived, int BR, int BC>
        Matrix& operator=(const Block<OtherDerived, BR, BC> &block) {
            static_assert(BR == Rows && BC == Cols, "Block dimensions must match Matrix dimensions");
            for (int c = 0; c < Cols; ++c) {
                for (int r = 0; r < Rows; ++r) {
                    (*this)(r, c) = block(r, c);
                }
            }
            return *this;
        }


        // Static factories
        static Matrix Zero() {
            Matrix m;
            m.setZero();
            return m;
        }

        static Matrix Identity() {
            Matrix m;
            m.setIdentity();
            return m;
        }

        static Matrix Constant(Scalar val) {
            Matrix m;
            for (int i = 0; i < SizeAtCompileTime; ++i) m.m_data[i] = val;
            return m;
        }

        // Modifiers
        void setZero() {
            std::fill_n(m_data, SizeAtCompileTime, Scalar(0));
        }

        void setIdentity() {
            setZero();
            constexpr int minDim = (Rows < Cols) ? Rows : Cols;
            for (int i = 0; i < minDim; ++i) {
                (*this)(i, i) = Scalar(1);
            }
        }

        // Element access (column-major)
        Scalar& operator()(int r, int c) {
            return m_data[c * Rows + r];
        }
        const Scalar& operator()(int r, int c) const {
            return m_data[c * Rows + r];
        }

        Scalar& operator[](int i) {
            return m_data[i];
        }
        const Scalar& operator[](int i) const {
            return m_data[i];
        }

        Scalar& operator()(int i) {
            return m_data[i];
        }
        const Scalar& operator()(int i) const {
            return m_data[i];
        }

        Scalar& x() { return m_data[0]; }
        const Scalar& x() const { return m_data[0]; }
        Scalar& y() { return m_data[1]; }
        const Scalar& y() const { return m_data[1]; }
        Scalar& z() { return m_data[2]; }
        const Scalar& z() const { return m_data[2]; }
        Scalar& w() { return m_data[3]; }
        const Scalar& w() const { return m_data[3]; }

        Scalar* data() { return m_data; }
        const Scalar* data() const { return m_data; }

        constexpr int rows() const { return Rows; }
        constexpr int cols() const { return Cols; }
        constexpr int size() const { return SizeAtCompileTime; }

        // Evaluator
        const Matrix& eval() const { return *this; }
        Matrix& eval() { return *this; }

        // Transpose
        Matrix<Scalar, Cols, Rows, Options> transpose() const {
            Matrix<Scalar, Cols, Rows, Options> res;
            for (int c = 0; c < Cols; ++c) {
                for (int r = 0; r < Rows; ++r) {
                    res(c, r) = (*this)(r, c);
                }
            }
            return res;
        }

        // Norms and Vector operations
        Scalar squaredNorm() const {
            Scalar sum = 0;
            for (int i = 0; i < SizeAtCompileTime; ++i) sum += m_data[i] * m_data[i];
            return sum;
        }

        Scalar norm() const {
            return std::sqrt(squaredNorm());
        }

        Matrix normalized() const {
            Scalar n = norm();
            if (n > static_cast<Scalar>(1e-12)) {
                return (*this) * (static_cast<Scalar>(1) / n);
            }
            return *this;
        }

        void normalize() {
            Scalar n = norm();
            if (n > static_cast<Scalar>(1e-12)) {
                (*this) *= (static_cast<Scalar>(1) / n);
            }
        }

        Scalar dot(const Matrix &other) const {
            Scalar sum = 0;
            for (int i = 0; i < SizeAtCompileTime; ++i) sum += m_data[i] * other.m_data[i];
            return sum;
        }

        template<int R = Rows, int C = Cols, typename = typename std::enable_if<R == 3 && C == 1>::type>
        Matrix cross(const Matrix &other) const {
            return Matrix(
                m_data[1] * other.m_data[2] - m_data[2] * other.m_data[1],
                m_data[2] * other.m_data[0] - m_data[0] * other.m_data[2],
                m_data[0] * other.m_data[1] - m_data[1] * other.m_data[0]
            );
        }

        template<int R = Rows, int C = Cols, typename = typename std::enable_if<(R == 2 && C == 2) || (R == 3 && C == 3)>::type>
        Scalar determinant() const {
            if constexpr (R == 2 && C == 2) {
                return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
            } else if constexpr (R == 3 && C == 3) {
                Scalar c00 = (*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1);
                Scalar c01 = (*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2);
                Scalar c02 = (*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0);
                return (*this)(0, 0) * c00 + (*this)(0, 1) * c01 + (*this)(0, 2) * c02;
            }
        }

        template<int R = Rows, int C = Cols, typename = typename std::enable_if<(R == 2 && C == 2) || (R == 3 && C == 3)>::type>
        Matrix inverse() const {
            if constexpr (R == 2 && C == 2) {
                Scalar det = determinant();
                Scalar invDet = static_cast<Scalar>(1) / det;
                Matrix res;
                res(0, 0) =  (*this)(1, 1) * invDet;
                res(0, 1) = -(*this)(0, 1) * invDet;
                res(1, 0) = -(*this)(1, 0) * invDet;
                res(1, 1) =  (*this)(0, 0) * invDet;
                return res;
            } else if constexpr (R == 3 && C == 3) {
                Scalar c00 = (*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1);
                Scalar c01 = (*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2);
                Scalar c02 = (*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0);

                Scalar det = (*this)(0, 0) * c00 + (*this)(0, 1) * c01 + (*this)(0, 2) * c02;
                Scalar invDet = static_cast<Scalar>(1) / det;

                Matrix res;
                res(0, 0) = c00 * invDet;
                res(0, 1) = ((*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2)) * invDet;
                res(0, 2) = ((*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1)) * invDet;

                res(1, 0) = c01 * invDet;
                res(1, 1) = ((*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0)) * invDet;
                res(1, 2) = ((*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2)) * invDet;

                res(2, 0) = c02 * invDet;
                res(2, 1) = ((*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1)) * invDet;
                res(2, 2) = ((*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0)) * invDet;

                return res;
            }
        }

        // Submatrix blocks
        template<int BlockRows, int BlockCols>
        Block<Matrix, BlockRows, BlockCols> block(int startRow, int startCol) {
            return Block<Matrix, BlockRows, BlockCols>(*this, startRow, startCol);
        }

        template<int BlockRows, int BlockCols>
        Block<const Matrix, BlockRows, BlockCols> block(int startRow, int startCol) const {
            return Block<const Matrix, BlockRows, BlockCols>(*this, startRow, startCol);
        }

        Block<Matrix, Rows, 1> col(int c) {
            return Block<Matrix, Rows, 1>(*this, 0, c);
        }
        Block<const Matrix, Rows, 1> col(int c) const {
            return Block<const Matrix, Rows, 1>(*this, 0, c);
        }

        Block<Matrix, 1, Cols> row(int r) {
            return Block<Matrix, 1, Cols>(*this, r, 0);
        }
        Block<const Matrix, 1, Cols> row(int r) const {
            return Block<const Matrix, 1, Cols>(*this, r, 0);
        }

        Matrix<Scalar, (Rows < Cols ? Rows : Cols), 1> diagonal() const {
            constexpr int minDim = (Rows < Cols) ? Rows : Cols;
            Matrix<Scalar, minDim, 1> diag;
            for (int i = 0; i < minDim; ++i) diag[i] = (*this)(i, i);
            return diag;
        }


        // Compound assignment operators
        Matrix& operator+=(const Matrix &other) {
            for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] += other.m_data[i];
            return *this;
        }

        Matrix& operator-=(const Matrix &other) {
            for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] -= other.m_data[i];
            return *this;
        }

        template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
        Matrix& operator*=(T s) {
            Scalar val = static_cast<Scalar>(s);
            for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] *= val;
            return *this;
        }

        template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
        Matrix& operator/=(T s) {
            Scalar inv = static_cast<Scalar>(1) / static_cast<Scalar>(s);
            for (int i = 0; i < SizeAtCompileTime; ++i) m_data[i] *= inv;
            return *this;
        }

        // Unary minus
        Matrix operator-() const {
            Matrix res;
            for (int i = 0; i < SizeAtCompileTime; ++i) res.m_data[i] = -m_data[i];
            return res;
        }

        // Linear Solvers caching
        LLT<Matrix>& llt() const;
        LLT<Matrix>& llt();
        LDLT<Matrix>& ldlt() const;
        LDLT<Matrix>& ldlt();

    private:
        Scalar m_data[SizeAtCompileTime];
        mutable LLT<Matrix>* m_lltCache = nullptr;
        mutable LDLT<Matrix>* m_ldltCache = nullptr;
    };

    // -----------------------------------------------------------------------------
    // Matrix Binary Arithmetic Operators
    // -----------------------------------------------------------------------------
    template<typename Scalar, int Rows, int Cols, int Options>
    Matrix<Scalar, Rows, Cols, Options> operator+(const Matrix<Scalar, Rows, Cols, Options> &a,
                                                 const Matrix<Scalar, Rows, Cols, Options> &b) {
        Matrix<Scalar, Rows, Cols, Options> res;
        for (int i = 0; i < Rows * Cols; ++i) res.data()[i] = a.data()[i] + b.data()[i];
        return res;
    }

    template<typename Scalar, int Rows, int Cols, int Options>
    Matrix<Scalar, Rows, Cols, Options> operator-(const Matrix<Scalar, Rows, Cols, Options> &a,
                                                 const Matrix<Scalar, Rows, Cols, Options> &b) {
        Matrix<Scalar, Rows, Cols, Options> res;
        for (int i = 0; i < Rows * Cols; ++i) res.data()[i] = a.data()[i] - b.data()[i];
        return res;
    }

    // Matrix-Matrix Multiplication
    template<typename Scalar, int R1, int C1, int R2, int C2, int Opt1, int Opt2>
    Matrix<Scalar, R1, C2, Opt1> operator*(const Matrix<Scalar, R1, C1, Opt1> &a,
                                           const Matrix<Scalar, R2, C2, Opt2> &b) {
        static_assert(C1 == R2, "Matrix dimensions must match for multiplication");
        Matrix<Scalar, R1, C2, Opt1> res = Matrix<Scalar, R1, C2, Opt1>::Zero();
        for (int j = 0; j < C2; ++j) {
            for (int k = 0; k < C1; ++k) {
                Scalar bkj = b(k, j);
                for (int i = 0; i < R1; ++i) {
                    res(i, j) += a(i, k) * bkj;
                }
            }
        }
        return res;
    }

    // Matrix * Scalar
    template<typename Scalar, int Rows, int Cols, int Options, typename T,
             typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Matrix<Scalar, Rows, Cols, Options> operator*(T s, const Matrix<Scalar, Rows, Cols, Options> &m) {
        Matrix<Scalar, Rows, Cols, Options> res;
        Scalar val = static_cast<Scalar>(s);
        for (int i = 0; i < Rows * Cols; ++i) res.data()[i] = val * m.data()[i];
        return res;
    }

    template<typename Scalar, int Rows, int Cols, int Options, typename T,
             typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Matrix<Scalar, Rows, Cols, Options> operator*(const Matrix<Scalar, Rows, Cols, Options> &m, T s) {
        return static_cast<Scalar>(s) * m;
    }

    template<typename Scalar, int Rows, int Cols, int Options, typename T,
             typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Matrix<Scalar, Rows, Cols, Options> operator/(const Matrix<Scalar, Rows, Cols, Options> &m, T s) {
        Matrix<Scalar, Rows, Cols, Options> res;
        Scalar inv = static_cast<Scalar>(1) / static_cast<Scalar>(s);
        for (int i = 0; i < Rows * Cols; ++i) res.data()[i] = m.data()[i] * inv;
        return res;
    }

    // Matrix * Block
    template<typename Scalar, int R1, int C1, int Opt1, typename Derived, int BR, int BC>
    Matrix<Scalar, R1, BC, Opt1> operator*(const Matrix<Scalar, R1, C1, Opt1> &a,
                                           const Block<Derived, BR, BC> &b) {
        static_assert(C1 == BR, "Matrix dimensions must match for multiplication");
        Matrix<Scalar, R1, BC, Opt1> res = Matrix<Scalar, R1, BC, Opt1>::Zero();
        for (int j = 0; j < BC; ++j) {
            for (int k = 0; k < C1; ++k) {
                Scalar bkj = b(k, j);
                for (int i = 0; i < R1; ++i) {
                    res(i, j) += a(i, k) * bkj;
                }
            }
        }
        return res;
    }

    // Block * Matrix
    template<typename Derived, int BR, int BC, typename Scalar, int R2, int C2, int Opt2>
    Matrix<Scalar, BR, C2, Opt2> operator*(const Block<Derived, BR, BC> &a,
                                           const Matrix<Scalar, R2, C2, Opt2> &b) {
        static_assert(BC == R2, "Matrix dimensions must match for multiplication");
        Matrix<Scalar, BR, C2, Opt2> res = Matrix<Scalar, BR, C2, Opt2>::Zero();
        for (int j = 0; j < C2; ++j) {
            for (int k = 0; k < BC; ++k) {
                Scalar bkj = b(k, j);
                for (int i = 0; i < BR; ++i) {
                    res(i, j) += a(i, k) * bkj;
                }
            }
        }
        return res;
    }

    // Block * Block
    template<typename D1, int R1, int C1, typename D2, int R2, int C2>
    Matrix<typename D1::Scalar, R1, C2> operator*(const Block<D1, R1, C1> &a,
                                                 const Block<D2, R2, C2> &b) {
        using Scalar = typename D1::Scalar;
        static_assert(C1 == R2, "Matrix dimensions must match for multiplication");
        Matrix<Scalar, R1, C2> res = Matrix<Scalar, R1, C2>::Zero();
        for (int j = 0; j < C2; ++j) {
            for (int k = 0; k < C1; ++k) {
                Scalar bkj = b(k, j);
                for (int i = 0; i < R1; ++i) {
                    res(i, j) += a(i, k) * bkj;
                }
            }
        }
        return res;
    }

    // Block + Matrix
    template<typename Derived, int BR, int BC, typename Scalar, int Opt>
    Matrix<Scalar, BR, BC, Opt> operator+(const Block<Derived, BR, BC> &a,
                                          const Matrix<Scalar, BR, BC, Opt> &b) {
        Matrix<Scalar, BR, BC, Opt> res;
        for (int c = 0; c < BC; ++c)
            for (int r = 0; r < BR; ++r)
                res(r, c) = a(r, c) + b(r, c);
        return res;
    }

    template<typename Scalar, int BR, int BC, int Opt, typename Derived>
    Matrix<Scalar, BR, BC, Opt> operator+(const Matrix<Scalar, BR, BC, Opt> &a,
                                          const Block<Derived, BR, BC> &b) {
        return b + a;
    }

    // Block - Matrix
    template<typename Derived, int BR, int BC, typename Scalar, int Opt>
    Matrix<Scalar, BR, BC, Opt> operator-(const Block<Derived, BR, BC> &a,
                                          const Matrix<Scalar, BR, BC, Opt> &b) {
        Matrix<Scalar, BR, BC, Opt> res;
        for (int c = 0; c < BC; ++c)
            for (int r = 0; r < BR; ++r)
                res(r, c) = a(r, c) - b(r, c);
        return res;
    }

    template<typename Scalar, int BR, int BC, int Opt, typename Derived>
    Matrix<Scalar, BR, BC, Opt> operator-(const Matrix<Scalar, BR, BC, Opt> &a,
                                          const Block<Derived, BR, BC> &b) {
        Matrix<Scalar, BR, BC, Opt> res;
        for (int c = 0; c < BC; ++c)
            for (int r = 0; r < BR; ++r)
                res(r, c) = a(r, c) - b(r, c);
        return res;
    }

    // Block * Scalar
    template<typename Derived, int BR, int BC, typename T,
             typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Matrix<typename Derived::Scalar, BR, BC> operator*(T s, const Block<Derived, BR, BC> &b) {
        using Scalar = typename Derived::Scalar;
        Matrix<Scalar, BR, BC> res;
        Scalar val = static_cast<Scalar>(s);
        for (int c = 0; c < BC; ++c)
            for (int r = 0; r < BR; ++r)
                res(r, c) = val * b(r, c);
        return res;
    }

    template<typename Derived, int BR, int BC, typename T,
             typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Matrix<typename Derived::Scalar, BR, BC> operator*(const Block<Derived, BR, BC> &b, T s) {
        return s * b;
    }

    template<typename Derived, int BR, int BC, typename T,
             typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Matrix<typename Derived::Scalar, BR, BC> operator/(const Block<Derived, BR, BC> &b, T s) {
        using Scalar = typename Derived::Scalar;
        Scalar inv = static_cast<Scalar>(1) / static_cast<Scalar>(s);
        return inv * b;
    }

    // -----------------------------------------------------------------------------
    // Diagonal Matrix
    // -----------------------------------------------------------------------------
    template<typename T, int Size>
    class DiagonalMatrix {
    public:
        using Scalar = T;

        DiagonalMatrix() {
            m_diag.setZero();
        }

        template<int S = Size, typename = typename std::enable_if<S == 3>::type>
        DiagonalMatrix(Scalar s0, Scalar s1, Scalar s2)
            : m_diag(s0, s1, s2) {}

        DiagonalMatrix(const Matrix<Scalar, Size, 1> &vec)
            : m_diag(vec) {}

        const Matrix<Scalar, Size, 1>& diagonal() const { return m_diag; }
        Matrix<Scalar, Size, 1>& diagonal() { return m_diag; }

        template<int Cols, int Opt>
        Matrix<Scalar, Size, Cols, Opt> operator*(const Matrix<Scalar, Size, Cols, Opt> &mat) const {
            Matrix<Scalar, Size, Cols, Opt> res;
            for (int c = 0; c < Cols; ++c) {
                for (int r = 0; r < Size; ++r) {
                    res(r, c) = m_diag[r] * mat(r, c);
                }
            }
            return res;
        }

    private:
        Matrix<Scalar, Size, 1> m_diag;
    };


    // Common Typedefs
    using Vector2f = Matrix<float, 2, 1>;
    using Vector3f = Matrix<float, 3, 1>;
    using Vector4f = Matrix<float, 4, 1>;
    using Vector2d = Matrix<double, 2, 1>;
    using Vector3d = Matrix<double, 3, 1>;
    using Vector4d = Matrix<double, 4, 1>;
    using Vector2i = Matrix<int, 2, 1>;
    using Vector3i = Matrix<int, 3, 1>;
    using Vector4i = Matrix<int, 4, 1>;

    using Matrix2f = Matrix<float, 2, 2>;
    using Matrix3f = Matrix<float, 3, 3>;
    using Matrix4f = Matrix<float, 4, 4>;
    using Matrix2d = Matrix<double, 2, 2>;
    using Matrix3d = Matrix<double, 3, 3>;
    using Matrix4d = Matrix<double, 4, 4>;

} // namespace LinearMath