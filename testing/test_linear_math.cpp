#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <cassert>

#include "LinearMath.h"

using namespace LinearMath;

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define TEST_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "Near check failed: " #a " (" << (a) << ") vs " #b " (" << (b) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

void test_vectors() {
    std::cout << "Testing Vectors..." << std::endl;

    Vector3r v1(1.0f, 2.0f, 3.0f);
    TEST_NEAR(v1.x(), 1.0f, 1e-6f);
    TEST_NEAR(v1.y(), 2.0f, 1e-6f);
    TEST_NEAR(v1.z(), 3.0f, 1e-6f);

    Vector3r v2(4.0f, 5.0f, 6.0f);
    Vector3r vSum = v1 + v2;
    TEST_NEAR(vSum[0], 5.0f, 1e-6f);
    TEST_NEAR(vSum[1], 7.0f, 1e-6f);
    TEST_NEAR(vSum[2], 9.0f, 1e-6f);

    Real d = v1.dot(v2);
    TEST_NEAR(d, 1.0f * 4.0f + 2.0f * 5.0f + 3.0f * 6.0f, 1e-5f);

    Vector3r c = v1.cross(v2);
    // (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
    TEST_NEAR(c.x(), -3.0f, 1e-5f);
    TEST_NEAR(c.y(),  6.0f, 1e-5f);
    TEST_NEAR(c.z(), -3.0f, 1e-5f);

    Vector3r vNorm = v1.normalized();
    TEST_NEAR(vNorm.norm(), 1.0f, 1e-5f);
    TEST_NEAR(v1.squaredNorm(), 1.0f + 4.0f + 9.0f, 1e-5f);

    Vector3r scaled = 2.0f * v1;
    TEST_NEAR(scaled.x(), 2.0f, 1e-5f);
    TEST_NEAR(scaled.y(), 4.0f, 1e-5f);
    TEST_NEAR(scaled.z(), 6.0f, 1e-5f);
}

void test_matrices() {
    std::cout << "Testing Matrices..." << std::endl;

    Matrix3r m = Matrix3r::Identity();
    TEST_NEAR(m(0, 0), 1.0f, 1e-6f);
    TEST_NEAR(m(1, 1), 1.0f, 1e-6f);
    TEST_NEAR(m(2, 2), 1.0f, 1e-6f);
    TEST_NEAR(m(0, 1), 0.0f, 1e-6f);

    Vector3r v(1.0f, 2.0f, 3.0f);
    Vector3r res = m * v;
    TEST_NEAR(res.x(), 1.0f, 1e-6f);
    TEST_NEAR(res.y(), 2.0f, 1e-6f);
    TEST_NEAR(res.z(), 3.0f, 1e-6f);

    Matrix3r A;
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;

    Matrix3r AT = A.transpose();
    TEST_NEAR(AT(0, 1), 4.0f, 1e-6f);
    TEST_NEAR(AT(1, 0), 2.0f, 1e-6f);

    LinearMath::Matrix<Real, 2, 3> B;
    B(0, 0) = 1; B(0, 1) = 2; B(0, 2) = 3;
    B(1, 0) = 4; B(1, 1) = 5; B(1, 2) = 6;

    LinearMath::Matrix<Real, 3, 2> BT = B.transpose();
    LinearMath::Matrix<Real, 2, 2> BBT = B * BT;
    // BBT(0,0) = 1*1 + 2*2 + 3*3 = 14
    // BBT(0,1) = 1*4 + 2*5 + 3*6 = 32
    // BBT(1,0) = 32
    // BBT(1,1) = 16 + 25 + 36 = 77
    TEST_NEAR(BBT(0, 0), 14.0f, 1e-5f);
    TEST_NEAR(BBT(0, 1), 32.0f, 1e-5f);
    TEST_NEAR(BBT(1, 0), 32.0f, 1e-5f);
    TEST_NEAR(BBT(1, 1), 77.0f, 1e-5f);
}

void test_blocks() {
    std::cout << "Testing Submatrix Blocks..." << std::endl;

    LinearMath::Matrix<Real, 6, 6> K;
    K.setZero();

    Matrix3r block0;
    block0.setIdentity();
    block0 *= 5.0f;

    K.block<3, 3>(0, 0) = block0;
    K.block<3, 3>(3, 3) += block0 * 2.0f;

    TEST_NEAR(K(0, 0), 5.0f, 1e-6f);
    TEST_NEAR(K(1, 1), 5.0f, 1e-6f);
    TEST_NEAR(K(3, 3), 10.0f, 1e-6f);
    TEST_NEAR(K(4, 4), 10.0f, 1e-6f);

    LinearMath::Matrix<Real, 6, 1> vec6;
    vec6.setZero();
    vec6[0] = 1; vec6[1] = 2; vec6[2] = 3;
    vec6[3] = 4; vec6[4] = 5; vec6[5] = 6;

    Vector3r top = vec6.block<3, 1>(0, 0);
    TEST_NEAR(top.x(), 1.0f, 1e-6f);
    TEST_NEAR(top.y(), 2.0f, 1e-6f);
    TEST_NEAR(top.z(), 3.0f, 1e-6f);

    Vector3r bottom = vec6.block<3, 1>(3, 0);
    TEST_NEAR(bottom.x(), 4.0f, 1e-6f);
    TEST_NEAR(bottom.y(), 5.0f, 1e-6f);
    TEST_NEAR(bottom.z(), 6.0f, 1e-6f);

    // Block vector operations
    Vector3r c = K.block<3, 1>(0, 0).cross(Vector3r(0, 1, 0));
    TEST_NEAR(c.z(), 5.0f, 1e-5f);
}

void test_llt() {
    std::cout << "Testing LLT Solver..." << std::endl;

    Matrix3r A;
    A.setZero();
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 1;
    A(1, 0) = 1; A(1, 1) = 3; A(1, 2) = 1;
    A(2, 0) = 1; A(2, 1) = 1; A(2, 2) = 2;

    Vector3r b(6.0f, 5.0f, 4.0f);

    auto& llt = A.llt();
    TEST_ASSERT(llt.info() == LinearMath::Success);

    Vector3r x = llt.solve(b);

    Vector3r Ax = A * x;
    TEST_NEAR(Ax.x(), b.x(), 1e-4f);
    TEST_NEAR(Ax.y(), b.y(), 1e-4f);
    TEST_NEAR(Ax.z(), b.z(), 1e-4f);
}

void test_ldlt() {
    std::cout << "Testing LDLT Solver..." << std::endl;

    Matrix6r A = Matrix6r::Identity();
    A *= 2.0f;
    A(0, 1) = 1.0f; A(1, 0) = 1.0f;
    A(3, 4) = 0.5f; A(4, 3) = 0.5f;

    LinearMath::LDLT<Matrix6r> ldlt;
    ldlt.compute(A);
    TEST_ASSERT(ldlt.info() == LinearMath::Success);

    Vector6r b;
    for (int i = 0; i < 6; ++i) b[i] = static_cast<Real>(i + 1);

    Vector6r x = ldlt.solve(b);
    Vector6r Ax = A * x;

    for (int i = 0; i < 6; ++i) {
        TEST_NEAR(Ax[i], b[i], 1e-4f);
    }
}

void test_quaternions() {
    std::cout << "Testing Quaternions..." << std::endl;

    Quaternionr q1(1.0f, 0.0f, 0.0f, 0.0f); // Identity (w=1, x=0, y=0, z=0)
    TEST_NEAR(q1.w(), 1.0f, 1e-6f);
    TEST_NEAR(q1.x(), 0.0f, 1e-6f);

    // 90 degree rotation around Z axis
    Real angle = static_cast<Real>(M_PI / 2.0);
    Vector3r axis(0.0f, 0.0f, 1.0f);
    AngleAxisr aa(angle, axis);
    Quaternionr qRot(aa);

    Vector3r v(1.0f, 0.0f, 0.0f);
    Vector3r vRot = qRot * v;

    // After 90 deg rotation around Z, (1, 0, 0) -> (0, 1, 0)
    TEST_NEAR(vRot.x(), 0.0f, 1e-5f);
    TEST_NEAR(vRot.y(), 1.0f, 1e-5f);
    TEST_NEAR(vRot.z(), 0.0f, 1e-5f);

    Matrix3r R = qRot.toRotationMatrix();
    Vector3r vRotMat = R * v;
    TEST_NEAR(vRotMat.x(), 0.0f, 1e-5f);
    TEST_NEAR(vRotMat.y(), 1.0f, 1e-5f);
    TEST_NEAR(vRotMat.z(), 0.0f, 1e-5f);

    // Inverse
    Quaternionr qInv = qRot.inverse();
    Vector3r vBack = qInv * vRot;
    TEST_NEAR(vBack.x(), 1.0f, 1e-5f);
    TEST_NEAR(vBack.y(), 0.0f, 1e-5f);
    TEST_NEAR(vBack.z(), 0.0f, 1e-5f);
}

void test_diagonal_matrix() {
    std::cout << "Testing DiagonalMatrix..." << std::endl;

    LinearMath::DiagonalMatrix<Real, 3> diag(2.0f, 3.0f, 4.0f);
    Matrix3r m = Matrix3r::Identity();

    Matrix3r res = diag * m;
    TEST_NEAR(res(0, 0), 2.0f, 1e-6f);
    TEST_NEAR(res(1, 1), 3.0f, 1e-6f);
    TEST_NEAR(res(2, 2), 4.0f, 1e-6f);
}

void test_aligned_box() {
    std::cout << "Testing AlignedBox..." << std::endl;

    AlignedBox3r box;
    TEST_ASSERT(box.isEmpty());

    box.extend(Vector3r(0.0f, 0.0f, 0.0f));
    box.extend(Vector3r(10.0f, 10.0f, 10.0f));

    TEST_ASSERT(!box.isEmpty());
    TEST_ASSERT(box.contains(Vector3r(5.0f, 5.0f, 5.0f)));
    TEST_ASSERT(!box.contains(Vector3r(15.0f, 5.0f, 5.0f)));
}

void test_matrix_inverse() {
    std::cout << "Testing Matrix Inverse & Determinant..." << std::endl;

    // 2x2 test
    Matrix2r M2;
    M2(0, 0) = 4.0f; M2(0, 1) = 7.0f;
    M2(1, 0) = 2.0f; M2(1, 1) = 6.0f;

    Real det2 = M2.determinant();
    TEST_NEAR(det2, 10.0f, 1e-5f);

    Matrix2r inv2 = M2.inverse();
    Matrix2r I2 = M2 * inv2;
    TEST_NEAR(I2(0, 0), 1.0f, 1e-5f);
    TEST_NEAR(I2(0, 1), 0.0f, 1e-5f);
    TEST_NEAR(I2(1, 0), 0.0f, 1e-5f);
    TEST_NEAR(I2(1, 1), 1.0f, 1e-5f);

    // 3x3 test
    Matrix3r M3;
    M3(0, 0) = 1.0f; M3(0, 1) = 2.0f; M3(0, 2) = 3.0f;
    M3(1, 0) = 0.0f; M3(1, 1) = 1.0f; M3(1, 2) = 4.0f;
    M3(2, 0) = 5.0f; M3(2, 1) = 6.0f; M3(2, 2) = 0.0f;

    Real det3 = M3.determinant();
    TEST_NEAR(det3, 1.0f, 1e-5f);

    Matrix3r inv3 = M3.inverse();
    Matrix3r I3 = M3 * inv3;
    TEST_NEAR(I3(0, 0), 1.0f, 1e-5f);
    TEST_NEAR(I3(0, 1), 0.0f, 1e-5f);
    TEST_NEAR(I3(0, 2), 0.0f, 1e-5f);
    TEST_NEAR(I3(1, 0), 0.0f, 1e-5f);
    TEST_NEAR(I3(1, 1), 1.0f, 1e-5f);
    TEST_NEAR(I3(1, 2), 0.0f, 1e-5f);
    TEST_NEAR(I3(2, 0), 0.0f, 1e-5f);
    TEST_NEAR(I3(2, 1), 0.0f, 1e-5f);
    TEST_NEAR(I3(2, 2), 1.0f, 1e-5f);
}

void test_matrix_utilities() {
    std::cout << "Testing Matrix Utilities (crossProductMatrix, norms, cotTheta)..." << std::endl;

    // Cross product matrix
    Vector3r a(1.0f, 2.0f, 3.0f);
    Vector3r b(4.0f, 5.0f, 6.0f);
    Matrix3r a_hat;
    matrix::crossProductMatrix(a, a_hat);
    Vector3r cross_mat = a_hat * b;
    Vector3r cross_vec = a.cross(b);
    TEST_NEAR(cross_mat.x(), cross_vec.x(), 1e-5f);
    TEST_NEAR(cross_mat.y(), cross_vec.y(), 1e-5f);
    TEST_NEAR(cross_mat.z(), cross_vec.z(), 1e-5f);

    // Overload
    Matrix3r a_hat2 = matrix::crossProductMatrix(a);
    TEST_NEAR(a_hat2(0, 1), a_hat(0, 1), 1e-6f);
    TEST_NEAR(a_hat2(1, 0), a_hat(1, 0), 1e-6f);

    // cotTheta and norms
    Vector3r vx(1.0f, 0.0f, 0.0f);
    Vector3r vy(0.0f, 1.0f, 0.0f);
    TEST_NEAR(matrix::cotTheta(vx, vy), 0.0f, 1e-5f);

    Matrix3r M;
    M(0, 0) = 1.0f; M(0, 1) = -2.0f; M(0, 2) = 3.0f;
    M(1, 0) = 0.0f; M(1, 1) =  1.0f; M(1, 2) = 1.0f;
    M(2, 0) = 2.0f; M(2, 1) = -1.0f; M(2, 2) = 1.0f;
    TEST_NEAR(matrix::oneNorm(M), 5.0f, 1e-5f);
    TEST_NEAR(matrix::infNorm(M), 6.0f, 1e-5f);
}

void test_decompositions() {
    std::cout << "Testing Decompositions (Eigen, Polar, SVD)..." << std::endl;

    // Eigen decomposition
    Matrix3r S;
    S(0, 0) = 2.0f; S(0, 1) = 1.0f; S(0, 2) = 0.0f;
    S(1, 0) = 1.0f; S(1, 1) = 3.0f; S(1, 2) = 1.0f;
    S(2, 0) = 0.0f; S(2, 1) = 1.0f; S(2, 2) = 2.0f;
    Matrix3r V;
    Vector3r D;
    decomposition::eigenDecomposition(S, V, D);
    Matrix3r IV = V * V.transpose();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TEST_NEAR(IV(r, c), (r == c ? 1.0f : 0.0f), 1e-4f);
        }
    }

    // Polar decomposition
    Quaternionr q(AngleAxisr(0.5f, Vector3r(0.0f, 1.0f, 0.0f)));
    Matrix3r R_true = q.toRotationMatrix();
    Matrix3r A = R_true * S;
    Matrix3r R_dec, U_dec, D_dec;
    decomposition::polarDecomposition(A, R_dec, U_dec, D_dec);
    Matrix3r IR = R_dec * R_dec.transpose();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TEST_NEAR(IR(r, c), (r == c ? 1.0f : 0.0f), 1e-3f);
        }
    }

    // Polar decomposition stable
    Matrix3r R_stable;
    decomposition::polarDecompositionStable(A, 1e-6f, R_stable);
    Matrix3r IR_stable = R_stable * R_stable.transpose();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TEST_NEAR(IR_stable(r, c), (r == c ? 1.0f : 0.0f), 1e-3f);
        }
    }

    // SVD with inversion handling
    Vector3r sigma;
    Matrix3r U_svd, VT_svd;
    decomposition::svdWithInversionHandling(A, sigma, U_svd, VT_svd);
    Matrix3r diagSigma = Matrix3r::Zero();
    diagSigma(0, 0) = sigma[0];
    diagSigma(1, 1) = sigma[1];
    diagSigma(2, 2) = sigma[2];
    Matrix3r A_reconstructed = U_svd * diagSigma * VT_svd;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TEST_NEAR(A(r, c), A_reconstructed(r, c), 1e-3f);
        }
    }

    // Extract rotation
    Quaternionr q_extracted;
    decomposition::extractRotation(A, q_extracted, 20);
    TEST_NEAR(std::abs(q_extracted.coeffs().dot(q.coeffs())), 1.0f, 1e-3f);
}

void test_kinematics_matrix() {
    std::cout << "Testing Kinematics Matrix (computeMatrixK, G, Q, QHat)..." << std::endl;

    // Test computeMatrixK
    Vector3r connector(1.0f, 0.5f, -0.2f);
    Vector3r x(0.0f, 0.0f, 0.0f);
    Real invMass = 1.5f;
    Matrix3r invInertia = Matrix3r::Identity();
    Matrix3r K;
    matrix::computeMatrixK(connector, invMass, x, invInertia, K);

    // K must be symmetric: K = K^T
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TEST_NEAR(K(r, c), K(c, r), 1e-6f);
        }
    }
    // Zero invMass gives zero K
    Matrix3r K_zero;
    matrix::computeMatrixK(connector, 0.0f, x, invInertia, K_zero);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TEST_NEAR(K_zero(r, c), 0.0f, 1e-6f);
        }
    }

    // Test computeMatrixQ: Q(q) * p = q * p
    Quaternionr q(0.7071068f, 0.0f, 0.7071068f, 0.0f); // 90 deg around Y
    Quaternionr p(0.5f, 0.5f, 0.5f, 0.5f);
    Matrix4r Q = matrix::computeMatrixQ(q);

    // Represent p as vector (w, x, y, z)
    LinearMath::Matrix<Real, 4, 1> p_vec(p.w(), p.x(), p.y(), p.z());
    LinearMath::Matrix<Real, 4, 1> qp_vec = Q * p_vec;

    Quaternionr prod = q * p;
    TEST_NEAR(qp_vec[0], prod.w(), 1e-5f);
    TEST_NEAR(qp_vec[1], prod.x(), 1e-5f);
    TEST_NEAR(qp_vec[2], prod.y(), 1e-5f);
    TEST_NEAR(qp_vec[3], prod.z(), 1e-5f);

    // Test computeMatrixQHat: QHat(q) * p = p * q
    Matrix4r QHat = matrix::computeMatrixQHat(q);
    LinearMath::Matrix<Real, 4, 1> pq_vec = QHat * p_vec;
    Quaternionr prod2 = p * q;
    TEST_NEAR(pq_vec[0], prod2.w(), 1e-5f);
    TEST_NEAR(pq_vec[1], prod2.x(), 1e-5f);
    TEST_NEAR(pq_vec[2], prod2.y(), 1e-5f);
    TEST_NEAR(pq_vec[3], prod2.z(), 1e-5f);

    // Test computeMatrixG: dot(q) = 0.5 * G(q) * omega
    // If omega = (0, 1, 0), q * [0, 0, 1, 0] gives 2 * dot(q)
    Matrix<Real, 4, 3, DontAlign> G = matrix::computeMatrixG(q);
    Vector3r omega(0.0f, 2.0f, 0.0f);
    LinearMath::Matrix<Real, 4, 1> q_dot = G * omega;
    // q * (0, 0, 2, 0) with w=0:
    Quaternionr omega_quat(0.0f, 0.0f, 2.0f, 0.0f);
    Quaternionr q_times_omega = q * omega_quat;
    TEST_NEAR(q_dot[0], 0.5f * q_times_omega.w(), 1e-5f);
    TEST_NEAR(q_dot[1], 0.5f * q_times_omega.x(), 1e-5f);
    TEST_NEAR(q_dot[2], 0.5f * q_times_omega.y(), 1e-5f);
    TEST_NEAR(q_dot[3], 0.5f * q_times_omega.z(), 1e-5f);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running LinearMath Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_vectors();
    test_matrices();
    test_blocks();
    test_matrix_inverse();
    test_llt();
    test_ldlt();
    test_quaternions();
    test_diagonal_matrix();
    test_aligned_box();
    test_matrix_utilities();
    test_decompositions();
    test_kinematics_matrix();

    std::cout << "========================================" << std::endl;
    std::cout << "All LinearMath tests passed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}

