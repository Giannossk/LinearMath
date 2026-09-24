#pragma once

#include <cmath>
#include "types.h"
#include "Dense.h"

namespace LinearMath {

class matrix
{
public:
	static Real infNorm(const Matrix3r &A)
	{
		const Real sum1 = std::abs(A(0, 0)) + std::abs(A(0, 1)) + std::abs(A(0, 2));
		const Real sum2 = std::abs(A(1, 0)) + std::abs(A(1, 1)) + std::abs(A(1, 2));
		const Real sum3 = std::abs(A(2, 0)) + std::abs(A(2, 1)) + std::abs(A(2, 2));
		Real maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		return maxSum;
	}

	static Real oneNorm(const Matrix3r &A)
	{
		const Real sum1 = std::abs(A(0, 0)) + std::abs(A(1, 0)) + std::abs(A(2, 0));
		const Real sum2 = std::abs(A(0, 1)) + std::abs(A(1, 1)) + std::abs(A(2, 1));
		const Real sum3 = std::abs(A(0, 2)) + std::abs(A(1, 2)) + std::abs(A(2, 2));
		Real maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		return maxSum;
	}

	static Real cotTheta(const Vector3r &v, const Vector3r &w)
	{
		const Real cosTheta = v.dot(w);
		const Real sinTheta = (v.cross(w)).norm();
		return (cosTheta / sinTheta);
	}

	/** Computes the cross product matrix of a vector.
	 * @param  v     input vector
	 * @param  v_hat resulting cross product matrix
	 */
	static void crossProductMatrix(const Vector3r &v, Matrix3r &v_hat)
	{
		v_hat(0, 0) = static_cast<Real>(0.0);  v_hat(0, 1) = -v[2];                  v_hat(0, 2) = v[1];
		v_hat(1, 0) = v[2];                   v_hat(1, 1) = static_cast<Real>(0.0);  v_hat(1, 2) = -v[0];
		v_hat(2, 0) = -v[1];                  v_hat(2, 1) = v[0];                   v_hat(2, 2) = static_cast<Real>(0.0);
	}

	static Matrix3r crossProductMatrix(const Vector3r &v)
	{
		Matrix3r v_hat;
		crossProductMatrix(v, v_hat);
		return v_hat;
	}

	/** Compute matrix K for single connector point:
	 * K = invMass * I - (r_hat * invInertia * r_hat)
	 */
	static void computeMatrixK(
		const Vector3r &connector,
		const Real invMass,
		const Vector3r &x,
		const Matrix3r &inertiaInverseW,
		Matrix3r &K)
	{
		if (invMass != static_cast<Real>(0.0))
		{
			const Vector3r v = connector - x;
			const Real a = v[0];
			const Real b = v[1];
			const Real c = v[2];

			// J is symmetric
			const Real j11 = inertiaInverseW(0, 0);
			const Real j12 = inertiaInverseW(0, 1);
			const Real j13 = inertiaInverseW(0, 2);
			const Real j22 = inertiaInverseW(1, 1);
			const Real j23 = inertiaInverseW(1, 2);
			const Real j33 = inertiaInverseW(2, 2);

			K(0, 0) = c*c*j22 - b*c*(j23 + j23) + b*b*j33 + invMass;
			K(0, 1) = -(c*c*j12) + a*c*j23 + b*c*j13 - a*b*j33;
			K(0, 2) = b*c*j12 - a*c*j22 - b*b*j13 + a*b*j23;
			K(1, 0) = K(0, 1);
			K(1, 1) = c*c*j11 - a*c*(j13 + j13) + a*a*j33 + invMass;
			K(1, 2) = -(b*c*j11) + a*c*j12 + a*b*j13 - a*a*j23;
			K(2, 0) = K(0, 2);
			K(2, 1) = K(1, 2);
			K(2, 2) = b*b*j11 - a*b*(j12 + j12) + a*a*j22 + invMass;
		}
		else
			K.setZero();
	}

	static Matrix3r computeMatrixK(
		const Vector3r &connector,
		const Real invMass,
		const Vector3r &x,
		const Matrix3r &inertiaInverseW)
	{
		Matrix3r K;
		computeMatrixK(connector, invMass, x, inertiaInverseW, K);
		return K;
	}

	/** Compute matrix K for two connector points */
	static void computeMatrixK(
		const Vector3r &connector0,
		const Vector3r &connector1,
		const Real invMass,
		const Vector3r &x,
		const Matrix3r &inertiaInverseW,
		Matrix3r &K)
	{
		if (invMass != static_cast<Real>(0.0))
		{
			const Vector3r v0 = connector0 - x;
			const Real a = v0[0];
			const Real b = v0[1];
			const Real c = v0[2];

			const Vector3r v1 = connector1 - x;
			const Real d = v1[0];
			const Real e = v1[1];
			const Real f = v1[2];

			// J is symmetric
			const Real j11 = inertiaInverseW(0, 0);
			const Real j12 = inertiaInverseW(0, 1);
			const Real j13 = inertiaInverseW(0, 2);
			const Real j22 = inertiaInverseW(1, 1);
			const Real j23 = inertiaInverseW(1, 2);
			const Real j33 = inertiaInverseW(2, 2);

			K(0, 0) = c*f*j22 - c*e*j23 - b*f*j23 + b*e*j33 + invMass;
			K(0, 1) = -(c*f*j12) + c*d*j23 + b*f*j13 - b*d*j33;
			K(0, 2) = c*e*j12 - c*d*j22 - b*e*j13 + b*d*j23;
			K(1, 0) = -(c*f*j12) + c*e*j13 + a*f*j23 - a*e*j33;
			K(1, 1) = c*f*j11 - c*d*j13 - a*f*j13 + a*d*j33 + invMass;
			K(1, 2) = -(c*e*j11) + c*d*j12 + a*e*j13 - a*d*j23;
			K(2, 0) = b*f*j12 - b*e*j13 - a*f*j22 + a*e*j23;
			K(2, 1) = -(b*f*j11) + b*d*j13 + a*f*j12 - a*d*j23;
			K(2, 2) = b*e*j11 - b*d*j12 - a*e*j12 + a*d*j22 + invMass;
		}
		else
			K.setZero();
	}

	static Matrix3r computeMatrixK(
		const Vector3r &connector0,
		const Vector3r &connector1,
		const Real invMass,
		const Vector3r &x,
		const Matrix3r &inertiaInverseW)
	{
		Matrix3r K;
		computeMatrixK(connector0, connector1, invMass, x, inertiaInverseW, K);
		return K;
	}

	/** Compute matrix that is required to transform quaternion in
	 * a 3D representation: dot(q) = 0.5 * G(q) * omega.
	 * Coordinate order is (w, x, y, z).
	 */
	template<int Opt = DontAlign>
	static void computeMatrixG(const Quaternionr &q, Matrix<Real, 4, 3, Opt> &G)
	{
		G(0, 0) = -static_cast<Real>(0.5) * q.x();
		G(0, 1) = -static_cast<Real>(0.5) * q.y();
		G(0, 2) = -static_cast<Real>(0.5) * q.z();

		G(1, 0) =  static_cast<Real>(0.5) * q.w();
		G(1, 1) =  static_cast<Real>(0.5) * q.z();
		G(1, 2) = -static_cast<Real>(0.5) * q.y();

		G(2, 0) = -static_cast<Real>(0.5) * q.z();
		G(2, 1) =  static_cast<Real>(0.5) * q.w();
		G(2, 2) =  static_cast<Real>(0.5) * q.x();

		G(3, 0) =  static_cast<Real>(0.5) * q.y();
		G(3, 1) = -static_cast<Real>(0.5) * q.x();
		G(3, 2) =  static_cast<Real>(0.5) * q.w();
	}

	static Matrix<Real, 4, 3, DontAlign> computeMatrixG(const Quaternionr &q)
	{
		Matrix<Real, 4, 3, DontAlign> G;
		computeMatrixG(q, G);
		return G;
	}

	/** Left quaternion multiplication matrix: Q(q) * p = q * p
	 * Coordinate order is (w, x, y, z).
	 */
	template<int Opt = DontAlign>
	static void computeMatrixQ(const Quaternionr &q, Matrix<Real, 4, 4, Opt> &Q)
	{
		Q(0, 0) =  q.w();
		Q(0, 1) = -q.x();
		Q(0, 2) = -q.y();
		Q(0, 3) = -q.z();

		Q(1, 0) =  q.x();
		Q(1, 1) =  q.w();
		Q(1, 2) = -q.z();
		Q(1, 3) =  q.y();

		Q(2, 0) =  q.y();
		Q(2, 1) =  q.z();
		Q(2, 2) =  q.w();
		Q(2, 3) = -q.x();

		Q(3, 0) =  q.z();
		Q(3, 1) = -q.y();
		Q(3, 2) =  q.x();
		Q(3, 3) =  q.w();
	}

	static Matrix4r computeMatrixQ(const Quaternionr &q)
	{
		Matrix4r Q;
		computeMatrixQ(q, Q);
		return Q;
	}

	/** Right quaternion multiplication matrix: QHat(q) * p = p * q
	 * Coordinate order is (w, x, y, z).
	 */
	template<int Opt = DontAlign>
	static void computeMatrixQHat(const Quaternionr &q, Matrix<Real, 4, 4, Opt> &Q)
	{
		Q(0, 0) =  q.w();
		Q(0, 1) = -q.x();
		Q(0, 2) = -q.y();
		Q(0, 3) = -q.z();

		Q(1, 0) =  q.x();
		Q(1, 1) =  q.w();
		Q(1, 2) =  q.z();
		Q(1, 3) = -q.y();

		Q(2, 0) =  q.y();
		Q(2, 1) = -q.z();
		Q(2, 2) =  q.w();
		Q(2, 3) =  q.x();

		Q(3, 0) =  q.z();
		Q(3, 1) =  q.y();
		Q(3, 2) = -q.x();
		Q(3, 3) =  q.w();
	}

	static Matrix4r computeMatrixQHat(const Quaternionr &q)
	{
		Matrix4r Q;
		computeMatrixQHat(q, Q);
		return Q;
	}
};

} // namespace LinearMath
