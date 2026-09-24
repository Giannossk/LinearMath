#pragma once
#include "Core.h"

namespace LinearMath {
    // -----------------------------------------------------------------------------
    // LLT (Cholesky Decomposition)
    // -----------------------------------------------------------------------------
    template<typename MatrixTypeT>
    class LLT {
    public:
        using MatrixType = MatrixTypeT;
        using Scalar = typename MatrixType::Scalar;
        static constexpr int Rows = MatrixType::RowsAtCompileTime;
        static constexpr int Cols = MatrixType::ColsAtCompileTime;
        static_assert(Rows == Cols, "LLT requires a square matrix");

        LLT() : m_info(Success) {}
        explicit LLT(const MatrixType &A) { compute(A); }

        LLT& compute(const MatrixType &A) {
            m_L = MatrixType::Zero();
            for (int j = 0; j < Rows; ++j) {
                Scalar sum = 0;
                for (int k = 0; k < j; ++k) {
                    sum += m_L(j, k) * m_L(j, k);
                }
                Scalar diff = A(j, j) - sum;
                if (diff <= static_cast<Scalar>(0)) {
                    m_info = NumericalIssue;
                    return *this;
                }
                m_L(j, j) = std::sqrt(diff);
                Scalar invLjj = static_cast<Scalar>(1) / m_L(j, j);

                for (int i = j + 1; i < Rows; ++i) {
                    Scalar s = 0;
                    for (int k = 0; k < j; ++k) {
                        s += m_L(i, k) * m_L(j, k);
                    }
                    m_L(i, j) = (A(i, j) - s) * invLjj;
                }
            }
            m_info = Success;
            return *this;
        }

        ComputationInfo info() const { return m_info; }

        template<typename Rhs>
        Rhs solve(const Rhs &b) const {
            Rhs x = b;
            int rhsCols = x.cols();
            // Forward solve: L y = b
            for (int c = 0; c < rhsCols; ++c) {
                for (int i = 0; i < Rows; ++i) {
                    Scalar sum = 0;
                    for (int k = 0; k < i; ++k) sum += m_L(i, k) * x(k, c);
                    x(i, c) = (b(i, c) - sum) / m_L(i, i);
                }
            }
            // Back solve: L^T x = y
            for (int c = 0; c < rhsCols; ++c) {
                for (int i = Rows - 1; i >= 0; --i) {
                    Scalar sum = 0;
                    for (int k = i + 1; k < Rows; ++k) sum += m_L(k, i) * x(k, c);
                    x(i, c) = (x(i, c) - sum) / m_L(i, i);
                }
            }
            return x;
        }

    private:
        MatrixType m_L;
        ComputationInfo m_info;
    };

    // -----------------------------------------------------------------------------
    // LDLT (Robust Cholesky without square roots)
    // -----------------------------------------------------------------------------
    template<typename MatrixTypeT>
    class LDLT {
    public:
        using MatrixType = MatrixTypeT;
        using Scalar = typename MatrixType::Scalar;
        static constexpr int Rows = MatrixType::RowsAtCompileTime;
        static constexpr int Cols = MatrixType::ColsAtCompileTime;
        static_assert(Rows == Cols, "LDLT requires a square matrix");

        LDLT() : m_info(Success) {}
        explicit LDLT(const MatrixType &A) { compute(A); }

        LDLT& compute(const MatrixType &A) {
            m_L = MatrixType::Identity();
            m_D = Matrix<Scalar, Rows, 1>::Zero();

            for (int j = 0; j < Rows; ++j) {
                Scalar sum = 0;
                for (int k = 0; k < j; ++k) {
                    sum += m_L(j, k) * m_L(j, k) * m_D[k];
                }
                Scalar dj = A(j, j) - sum;
                if (std::abs(dj) < static_cast<Scalar>(1e-14)) {
                    dj = (dj < 0) ? -static_cast<Scalar>(1e-14) : static_cast<Scalar>(1e-14);
                }
                m_D[j] = dj;
                Scalar invDj = static_cast<Scalar>(1) / dj;

                for (int i = j + 1; i < Rows; ++i) {
                    Scalar s = 0;
                    for (int k = 0; k < j; ++k) {
                        s += m_L(i, k) * m_L(j, k) * m_D[k];
                    }
                    m_L(i, j) = (A(i, j) - s) * invDj;
                }
            }
            m_info = Success;
            return *this;
        }

        ComputationInfo info() const { return m_info; }

        template<typename Rhs>
        Rhs solve(const Rhs &b) const {
            Rhs x = b;
            int rhsCols = x.cols();
            // 1. Solve L z = b
            for (int c = 0; c < rhsCols; ++c) {
                for (int i = 0; i < Rows; ++i) {
                    Scalar sum = 0;
                    for (int k = 0; k < i; ++k) sum += m_L(i, k) * x(k, c);
                    x(i, c) = b(i, c) - sum;
                }
            }
            // 2. Solve D y = z
            for (int c = 0; c < rhsCols; ++c) {
                for (int i = 0; i < Rows; ++i) {
                    x(i, c) /= m_D[i];
                }
            }
            // 3. Solve L^T x = y
            for (int c = 0; c < rhsCols; ++c) {
                for (int i = Rows - 1; i >= 0; --i) {
                    Scalar sum = 0;
                    for (int k = i + 1; k < Rows; ++k) sum += m_L(k, i) * x(k, c);
                    x(i, c) -= sum;
                }
            }
            return x;
        }

    private:
        MatrixType m_L;
        Matrix<Scalar, Rows, 1> m_D;
        ComputationInfo m_info;
    };

    // Implementations of Matrix::llt() and Matrix::ldlt()
    template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    inline LLT<Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>>&
    Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>::llt() const {
        if (!m_lltCache) m_lltCache = new LLT<Matrix>();
        m_lltCache->compute(*this);
        return *m_lltCache;
    }

    template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    inline LLT<Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>>&
    Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>::llt() {
        if (!m_lltCache) m_lltCache = new LLT<Matrix>();
        m_lltCache->compute(*this);
        return *m_lltCache;
    }

    template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    inline LDLT<Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>>&
    Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>::ldlt() const {
        if (!m_ldltCache) m_ldltCache = new LDLT<Matrix>();
        m_ldltCache->compute(*this);
        return *m_ldltCache;
    }

    template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    inline LDLT<Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>>&
    Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>::ldlt() {
        if (!m_ldltCache) m_ldltCache = new LDLT<Matrix>();
        m_ldltCache->compute(*this);
        return *m_ldltCache;
    }


} // namespace LinearMath