#include <cmath>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "diffOptHelper.h"

/*
    We consider F(p) = (p - a)^T @ R @ D @ R.T @ (p - a) where a is a vector and
    R = [[[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]].
    D = diag(D11, D22, D33).
    Note A = R @ D @ R.T for simplicity.
    The parameters of F are x = [qx, qy, qz, qw, ax, ay, az]. This class calculates
    various derivatives of F w.r.t. x and p.
*/

Eigen::VectorXd rimonMethod(const Eigen::MatrixXd& A, const Eigen::VectorXd& a, const Eigen::MatrixXd& B, const Eigen::VectorXd& b) {
    int nv = A.rows();
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A); // Compute the Cholesky decomposition of A
    Eigen::MatrixXd A_sqrt = lltOfA.matrixL(); // Extract the lower triangular matrix

    // C = inv(A_sqrt) * B * inv(A_sqrt).T
    Eigen::MatrixXd A_sqrt_inv = A_sqrt.inverse();
    Eigen::MatrixXd C = A_sqrt_inv * B * A_sqrt_inv.transpose();
    
    Eigen::VectorXd c = A_sqrt.transpose() * (b - a);
    
    Eigen::LLT<Eigen::MatrixXd> lltOfC(C); // Cholesky decomposition of C
    Eigen::MatrixXd C_sqrt = lltOfC.matrixL(); // Lower triangular matrix of C
    Eigen::VectorXd c_tilde = C_sqrt.triangularView<Eigen::Lower>().solve(c);
    
    // Computing C_tilde
    Eigen::MatrixXd C_sqrt_inv = C_sqrt.inverse();
    Eigen::MatrixXd C_tilde = C_sqrt_inv * C_sqrt_inv.transpose();
    
    // Construct matrix M
    Eigen::MatrixXd M(2*nv, 2*nv);
    M << C_tilde, -Eigen::MatrixXd::Identity(nv, nv),
        -c_tilde * c_tilde.transpose(), C_tilde; // doubt
    
    // Compute the smallest eigenvalue of M
    Eigen::EigenSolver<Eigen::MatrixXd> es(M);
    double lambda_min = es.eigenvalues().real().minCoeff();
    
    // Solve for x_rimon
    Eigen::VectorXd x_rimon = (lambda_min * C - Eigen::MatrixXd::Identity(nv, nv)).ldlt().solve(C * c);
    x_rimon = a + lambda_min * A_sqrt.transpose().triangularView<Eigen::Upper>().solve(x_rimon);
    
    return x_rimon;
}

xt::xarray<double> rimonMethodXtensor(const xt::xarray<double>& A, const xt::xarray<double>& a, const xt::xarray<double>& B, const xt::xarray<double>& b) {
    int nv = A.shape()[0];

    xt::xarray<double> A_sqrt = xt::linalg::cholesky(A); 

    xt::xarray<double> A_sqrt_inv = xt::linalg::inv(A_sqrt);
    
    // Eigen::MatrixXd C = A_sqrt_inv * B * A_sqrt_inv.transpose();
    xt::xarray<double> C = xt::linalg::dot(xt::linalg::dot(A_sqrt_inv, B), xt::transpose(A_sqrt_inv));

    // Eigen::VectorXd c = A_sqrt.transpose() * (b - a);
    xt::xarray<double> c = xt::linalg::dot(xt::transpose(A_sqrt), b - a);

    // Placeholder for Cholesky decomposition on C
    xt::xarray<double> C_sqrt = xt::linalg::cholesky(C); 
    
    // Eigen::VectorXd c_tilde = C_sqrt.triangularView<Eigen::Lower>().solve(c);
    xt::xarray<double> c_tilde = xt::linalg::solve(C_sqrt, c); 

    // Eigen::MatrixXd C_sqrt_inv = C_sqrt.inverse();
    xt::xarray<double> C_sqrt_inv = xt::linalg::inv(C_sqrt);
    
    // Eigen::MatrixXd C_tilde = C_sqrt_inv * C_sqrt_inv.transpose();
    xt::xarray<double> C_tilde = xt::linalg::dot(C_sqrt_inv, xt::transpose(C_sqrt_inv));

    // Construct matrix M
    xt::xarray<double> M = xt::zeros<double>({2*nv, 2*nv}); // Initialize M with zeros
    xt::view(M, xt::range(0, nv), xt::range(0, nv)) = C_tilde;
    xt::view(M, xt::range(nv, 2*nv), xt::range(nv, 2*nv)) = C_tilde;
    xt::view(M, xt::range(0, nv), xt::range(nv, 2*nv)) = -xt::eye(nv);
    xt::view(M, xt::range(nv, 2*nv), xt::range(0, nv)) = xt::linalg::outer(-c_tilde, xt::transpose(c_tilde));

    // Compute the smallest eigenvalue of M
    double lambda_min = xt::amin(xt::real(xt::linalg::eigvals(M)))();

    // Solve for x_rimon
    xt::xarray<double> x_rimon = xt::linalg::solve(lambda_min * C - xt::eye(nv), xt::linalg::dot(C, c)); 
    x_rimon = a + lambda_min * xt::linalg::solve(xt::transpose(A_sqrt), x_rimon);
    
    return x_rimon;
}

double ellipsoid_F(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute F(p)
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3

    return xt::linalg::dot(p - a, xt::linalg::dot(A, p - a))(0);
}

xt::xarray<double> ellipsoid_dp(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute dF/dp
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3

    return 2 * xt::linalg::dot(A, p - a); // Since A is symmetric, A.transpose() is the same as A
}

xt::xarray<double> ellipsoid_dpdp(const xt::xarray<double>& A) {
    // Compute d^2F/dpdp
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3

    return 2 * A; // Since A is symmetric, A.transpose() is the same as A
}

xt::xarray<double> ellipsoid_dpdpdp() {
    // Compute d^3F/dpdpdp

    return xt::zeros<double>({3, 3, 3});;
}

xt::xarray<double> ellipsoid_dy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute dF/dy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    
    int dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> vector = a - p;
    xt::xarray<double> outer_prod = xt::linalg::outer(vector, vector);
    xt::xarray<double> F_dy = xt::zeros<double>({dim_y});
    F_dy(0) = outer_prod(0, 0);
    F_dy(1) = 2 * outer_prod(0, 1);
    F_dy(2) = 2 * outer_prod(0, 2);
    F_dy(3) = outer_prod(1, 1);
    F_dy(4) = 2 * outer_prod(1, 2);
    F_dy(5) = outer_prod(2, 2);
    xt::view(F_dy, xt::range(dim_A_flat, dim_y)) = 2 * xt::linalg::dot(A, vector);

    return F_dy;
}

xt::xarray<double> ellipsoid_dpdy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute d^2F/dpdy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;

    xt::xarray<double> F_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> vector = p - a;
    F_dpdy(0,0) = 2 * vector(0);
    F_dpdy(0,1) = 2 * vector(1);
    F_dpdy(0,2) = 2 * vector(2);
    F_dpdy(1,1) = 2 * vector(0);
    F_dpdy(1,3) = 2 * vector(1);
    F_dpdy(1,4) = 2 * vector(2);
    F_dpdy(2,2) = 2 * vector(0);
    F_dpdy(2,4) = 2 * vector(1);
    F_dpdy(2,5) = 2 * vector(2);
    xt::view(F_dpdy, xt::all(), xt::range(dim_A_flat, dim_y)) = -2 * A;

    return F_dpdy;
}

xt::xarray<double> ellipsoid_dydy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute d^2F/dydy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> F_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> vector = a - p;
    xt::xarray<double> tmp = xt::zeros<double>({dim_p, dim_A_flat});
    tmp(0,0) = 2 * vector(0);
    tmp(0,1) = 2 * vector(1);
    tmp(0,2) = 2 * vector(2);
    tmp(1,1) = 2 * vector(0);
    tmp(1,3) = 2 * vector(1);
    tmp(1,4) = 2 * vector(2);
    tmp(2,2) = 2 * vector(0);
    tmp(2,4) = 2 * vector(1);
    tmp(2,5) = 2 * vector(2);
    xt::view(F_dydy, xt::range(dim_A_flat, dim_y), xt::range(0, dim_A_flat)) = tmp;
    xt::view(F_dydy, xt::range(0, dim_A_flat), xt::range(dim_A_flat, dim_y)) = xt::transpose(tmp);
    xt::view(F_dydy, xt::range(dim_A_flat, dim_y), xt::range(dim_A_flat, dim_y)) = 2 * A;

    return F_dydy;
}

xt::xarray<double> ellipsoid_dpdpdy(const xt::xarray<double>& A) {
    // Compute d^3F/dpdpdy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9;
    xt::xarray<double> F_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    F_dpdpdy(0,0,0) = 2;
    F_dpdpdy(0,1,1) = 2;
    F_dpdpdy(1,0,1) = 2;
    F_dpdpdy(0,2,2) = 2;
    F_dpdpdy(2,0,2) = 2;
    F_dpdpdy(1,1,3) = 2;
    F_dpdpdy(1,2,4) = 2;
    F_dpdpdy(2,1,4) = 2;
    F_dpdpdy(2,2,5) = 2;

    return F_dpdpdy;
}

xt::xarray<double> ellipsoid_dpdydy(const xt::xarray<double>& A) {
    // Compute d^3F/dpdpdy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> F_dpdydy = xt::zeros<double>({dim_p, dim_y, dim_y});
    xt::xarray<double> tmp = xt::zeros<double>({dim_p, dim_p, dim_A_flat});
    tmp(0,0,0) = -2;
    tmp(0,1,1) = -2;
    tmp(1,0,1) = -2;
    tmp(0,2,2) = -2;
    tmp(2,0,2) = -2;
    tmp(1,1,3) = -2;
    tmp(1,2,4) = -2;
    tmp(2,1,4) = -2;
    tmp(2,2,5) = -2;

    xt::view(F_dpdydy, xt::all(), xt::range(dim_A_flat, dim_y), xt::range(0, dim_A_flat)) = tmp;
    xt::view(F_dpdydy, xt::all(), xt::range(0, dim_A_flat), xt::range(dim_A_flat, dim_y)) = xt::transpose(tmp, {0,2,1});

    return F_dpdydy;
}

xt::xarray<double> ellipsoid_RDRT_dq(const xt::xarray<double>& q, const xt::xarray<double>& D, const xt::xarray<double>& R){
    // Compute d(R @ D @ R.T)/dq
    // q: input vector of dimension 4
    // D: diagonal matrix of dimension 3 x 3
    // R: rotation matrix of dimension 3 x 3
    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    double a = D(0,0), b = D(1,1), c = D(2,2);
    double r11 = R(0,0), r12 = R(0,1), r13 = R(0,2);
    double r21 = R(1,0), r22 = R(1,1), r23 = R(1,2);
    double r31 = R(2,0), r32 = R(2,1), r33 = R(2,2);

    xt::xarray<double> M11_dq {8*a*qx*r11 + 4*b*qy*r12 + 4*c*qz*r13,
                                4*b*qx*r12 + 4*c*qw*r13,
                                -4*b*qw*r12 + 4*c*qx*r13,
                                8*a*qw*r11 - 4*b*qz*r12 + 4*c*qy*r13};

    xt::xarray<double> M12_dq {4*a*qx*r21 + 2*a*qy*r11 + 2*b*qy*r22 - 2*c*qw*r13 + 2*c*qz*r23,
                                2*a*qx*r11 + 2*b*qx*r22 + 4*b*qy*r12 + 2*c*qw*r23 + 2*c*qz*r13,
                                2*a*qw*r11 - 2*b*qw*r22 + 2*c*qx*r23 + 2*c*qy*r13,
                                4*a*qw*r21 + 2*a*qz*r11 + 4*b*qw*r12 - 2*b*qz*r22 - 2*c*qx*r13 + 2*c*qy*r23};

    xt::xarray<double> M13_dq {4*a*qx*r31 + 2*a*qz*r11 + 2*b*qw*r12 + 2*b*qy*r32 + 2*c*qz*r33,
                                -2*a*qw*r11 + 2*b*qx*r32 + 2*b*qz*r12 + 2*c*qw*r33,
                                2*a*qx*r11 - 2*b*qw*r32 + 2*b*qy*r12 + 2*c*qx*r33 + 4*c*qz*r13,
                                4*a*qw*r31 - 2*a*qy*r11 + 2*b*qx*r12 - 2*b*qz*r32 + 4*c*qw*r13 + 2*c*qy*r33};

    xt::xarray<double> M22_dq {4*a*qy*r21 - 4*c*qw*r23,
                                4*a*qx*r21 + 8*b*qy*r22 + 4*c*qz*r23,
                                4*a*qw*r21 + 4*c*qy*r23,
                                4*a*qz*r21 + 8*b*qw*r22 - 4*c*qx*r23};

    xt::xarray<double> M23_dq {2*a*qy*r31 + 2*a*qz*r21 + 2*b*qw*r22 - 2*c*qw*r33,
                                -2*a*qw*r21 + 2*a*qx*r31 + 4*b*qy*r32 + 2*b*qz*r22 + 2*c*qz*r33,
                                2*a*qw*r31 + 2*a*qx*r21 + 2*b*qy*r22 + 2*c*qy*r33 + 4*c*qz*r23,
                                -2*a*qy*r21 + 2*a*qz*r31 + 4*b*qw*r32 + 2*b*qx*r22 + 4*c*qw*r23 - 2*c*qx*r33};

    xt::xarray<double> M33_dq {4*a*qz*r31 + 4*b*qw*r32,
                                -4*a*qw*r31 + 4*b*qz*r32,
                                4*a*qx*r31 + 4*b*qy*r32 + 8*c*qz*r33,
                                -4*a*qy*r31 + 4*b*qx*r32 + 8*c*qw*r33};                        

    xt::xarray<double> RDRT_dq = xt::zeros<double>({6, 4});
    xt::view(RDRT_dq, 0, xt::all()) = M11_dq;
    xt::view(RDRT_dq, 1, xt::all()) = M12_dq;
    xt::view(RDRT_dq, 2, xt::all()) = M13_dq;
    xt::view(RDRT_dq, 3, xt::all()) = M22_dq;
    xt::view(RDRT_dq, 4, xt::all()) = M23_dq;
    xt::view(RDRT_dq, 5, xt::all()) = M33_dq;

    return RDRT_dq;
}

xt::xarray<double> ellipsoid_RDRT_dqdq(const xt::xarray<double>& q, const xt::xarray<double>& D){
    // Compute d^2(R @ D @ R.T)/dqdq
    // q: input vector of dimension 4
    // D: diagonal matrix of dimension 3 x 3
    // R: rotation matrix of dimension 3 x 3

    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    double a = D(0,0), b = D(1,1), c = D(2,2);

    double pxx = qx*qx, pxy = qx*qy, pxz = qx*qz, pxw = qx*qw, pyy = qy*qy;
    double pyz = qy*qz, pyw = qy*qw, pzz = qz*qz, pzw = qz*qw, pww = qw*qw;

    double M11_dqdq_11 = 16*a*pww + 48*a*pxx - 8*a + 8*b*pyy + 8*c*pzz;
    double M11_dqdq_12 = 16*b*pxy - 8*b*pzw + 8*c*pzw;
    double M11_dqdq_13 = -8*b*pyw + 16*c*pxz + 8*c*pyw;
    double M11_dqdq_14 = 32*a*pxw - 8*b*pyz + 8*c*pyz;
    double M11_dqdq_22 = 8*b*pxx + 8*c*pww;
    double M11_dqdq_23 = 8*pxw*(-b + c);
    double M11_dqdq_24 = -8*b*pxz + 8*c*pxz + 16*c*pyw;
    double M11_dqdq_33 = 8*b*pww + 8*c*pxx;
    double M11_dqdq_34 = -8*b*pxy + 16*b*pzw + 8*c*pxy;
    double M11_dqdq_44 = 48*a*pww + 16*a*pxx - 8*a + 8*b*pzz + 8*c*pyy;

    xt::xarray<double> M11_dqdq {{M11_dqdq_11, M11_dqdq_12, M11_dqdq_13, M11_dqdq_14},
                                {M11_dqdq_12, M11_dqdq_22, M11_dqdq_23, M11_dqdq_24},
                                {M11_dqdq_13, M11_dqdq_23, M11_dqdq_33, M11_dqdq_34},
                                {M11_dqdq_14, M11_dqdq_24, M11_dqdq_34, M11_dqdq_44}};

    double M12_dqdq_11 = 24*a*pxy + 8*a*pzw - 8*c*pzw;
    double M12_dqdq_12 = 4*a*pww + 12*a*pxx - 2*a + 4*b*pww + 12*b*pyy - 2*b - 4*c*pww + 4*c*pzz;
    double M12_dqdq_13 = 8*a*pxw - 8*c*pxw + 8*c*pyz;
    double M12_dqdq_14 = 8*a*pxz + 8*a*pyw + 8*b*pyw - 8*c*pxz - 8*c*pyw;
    double M12_dqdq_22 = 24*b*pxy - 8*b*pzw + 8*c*pzw;
    double M12_dqdq_23 = -8*b*pyw + 8*c*pxz + 8*c*pyw;
    double M12_dqdq_24 = 8*a*pxw + 8*b*pxw - 8*b*pyz - 8*c*pxw + 8*c*pyz;
    double M12_dqdq_33 = 8*c*pxy;
    double M12_dqdq_34 = 12*a*pww + 4*a*pxx - 2*a - 12*b*pww - 4*b*pyy + 2*b - 4*c*pxx + 4*c*pyy;
    double M12_dqdq_44 = 8*a*pxy + 24*a*pzw + 8*b*pxy - 24*b*pzw - 8*c*pxy;

    xt::xarray<double> M12_dqdq {{M12_dqdq_11, M12_dqdq_12, M12_dqdq_13, M12_dqdq_14},
                                {M12_dqdq_12, M12_dqdq_22, M12_dqdq_23, M12_dqdq_24},
                                {M12_dqdq_13, M12_dqdq_23, M12_dqdq_33, M12_dqdq_34},
                                {M12_dqdq_14, M12_dqdq_24, M12_dqdq_34, M12_dqdq_44}};

    double M13_dqdq_11 = 24*a*pxz - 8*a*pyw + 8*b*pyw;
    double M13_dqdq_12 = -8*a*pxw + 8*b*pxw + 8*b*pyz;
    double M13_dqdq_13 = 4*a*pww + 12*a*pxx - 2*a - 4*b*pww + 4*b*pyy + 4*c*pww + 12*c*pzz - 2*c;
    double M13_dqdq_14 = -8*a*pxy + 8*a*pzw + 8*b*pxy - 8*b*pzw + 8*c*pzw;
    double M13_dqdq_22 = 8*b*pxz;
    double M13_dqdq_23 = 8*b*pxy - 8*b*pzw + 8*c*pzw;
    double M13_dqdq_24 = -12*a*pww - 4*a*pxx + 2*a + 4*b*pxx - 4*b*pzz + 12*c*pww + 4*c*pzz - 2*c;
    double M13_dqdq_33 = -8*b*pyw + 24*c*pxz + 8*c*pyw;
    double M13_dqdq_34 = 8*a*pxw - 8*b*pxw - 8*b*pyz + 8*c*pxw + 8*c*pyz;
    double M13_dqdq_44 = 8*a*pxz - 24*a*pyw - 8*b*pxz + 8*c*pxz + 24*c*pyw;

    xt::xarray<double> M13_dqdq {{M13_dqdq_11, M13_dqdq_12, M13_dqdq_13, M13_dqdq_14},
                                {M13_dqdq_12, M13_dqdq_22, M13_dqdq_23, M13_dqdq_24},
                                {M13_dqdq_13, M13_dqdq_23, M13_dqdq_33, M13_dqdq_34},
                                {M13_dqdq_14, M13_dqdq_24, M13_dqdq_34, M13_dqdq_44}};
    
    double M22_dqdq_11 = 8*a*pyy + 8*c*pww;
    double M22_dqdq_12 = 16*a*pxy + 8*a*pzw - 8*c*pzw;
    double M22_dqdq_13 = 8*pyw*(a - c);
    double M22_dqdq_14 = 8*a*pyz + 16*c*pxw - 8*c*pyz;
    double M22_dqdq_22 = 8*a*pxx + 16*b*pww + 48*b*pyy - 8*b + 8*c*pzz;
    double M22_dqdq_23 = 8*a*pxw - 8*c*pxw + 16*c*pyz;
    double M22_dqdq_24 = 8*a*pxz + 32*b*pyw - 8*c*pxz;
    double M22_dqdq_33 = 8*a*pww + 8*c*pyy;
    double M22_dqdq_34 = 8*a*pxy + 16*a*pzw - 8*c*pxy;
    double M22_dqdq_44 = 8*a*pzz + 48*b*pww + 16*b*pyy - 8*b + 8*c*pxx;

    xt::xarray<double> M22_dqdq {{M22_dqdq_11, M22_dqdq_12, M22_dqdq_13, M22_dqdq_14},
                                {M22_dqdq_12, M22_dqdq_22, M22_dqdq_23, M22_dqdq_24},
                                {M22_dqdq_13, M22_dqdq_23, M22_dqdq_33, M22_dqdq_34},
                                {M22_dqdq_14, M22_dqdq_24, M22_dqdq_34, M22_dqdq_44}};

    double M23_dqdq_11 = 8*a*pyz;
    double M23_dqdq_12 = 8*a*pxz - 8*a*pyw + 8*b*pyw;
    double M23_dqdq_13 = 8*a*pxy + 8*a*pzw - 8*c*pzw;
    double M23_dqdq_14 = -4*a*pyy + 4*a*pzz + 12*b*pww + 4*b*pyy - 2*b - 12*c*pww - 4*c*pzz + 2*c;
    double M23_dqdq_22 = -8*a*pxw + 8*b*pxw + 24*b*pyz;
    double M23_dqdq_23 = -4*a*pww + 4*a*pxx + 4*b*pww + 12*b*pyy - 2*b + 4*c*pww + 12*c*pzz - 2*c;
    double M23_dqdq_24 = -8*a*pxy - 8*a*pzw + 8*b*pxy + 8*b*pzw + 8*c*pzw;
    double M23_dqdq_33 = 8*a*pxw - 8*c*pxw + 24*c*pyz;
    double M23_dqdq_34 = 8*a*pxz - 8*a*pyw + 8*b*pyw - 8*c*pxz + 8*c*pyw;
    double M23_dqdq_44 = -8*a*pyz + 24*b*pxw + 8*b*pyz - 24*c*pxw + 8*c*pyz;

    xt::xarray<double> M23_dqdq {{M23_dqdq_11, M23_dqdq_12, M23_dqdq_13, M23_dqdq_14},
                                {M23_dqdq_12, M23_dqdq_22, M23_dqdq_23, M23_dqdq_24},
                                {M23_dqdq_13, M23_dqdq_23, M23_dqdq_33, M23_dqdq_34},
                                {M23_dqdq_14, M23_dqdq_24, M23_dqdq_34, M23_dqdq_44}};

    double M33_dqdq_11 = 8*a*pzz + 8*b*pww;
    double M33_dqdq_12 = 8*pzw*(-a + b);
    double M33_dqdq_13 = 16*a*pxz - 8*a*pyw + 8*b*pyw;
    double M33_dqdq_14 = -8*a*pyz + 16*b*pxw + 8*b*pyz;
    double M33_dqdq_22 = 8*a*pww + 8*b*pzz;
    double M33_dqdq_23 = -8*a*pxw + 8*b*pxw + 16*b*pyz;
    double M33_dqdq_24 = -8*a*pxz + 16*a*pyw + 8*b*pxz;
    double M33_dqdq_33 = 8*a*pxx + 8*b*pyy + 16*c*pww + 48*c*pzz - 8*c;
    double M33_dqdq_34 = -8*a*pxy + 8*b*pxy + 32*c*pzw;
    double M33_dqdq_44 = 8*a*pyy + 8*b*pxx + 48*c*pww + 16*c*pzz - 8*c;

    xt::xarray<double> M33_dqdq {{M33_dqdq_11, M33_dqdq_12, M33_dqdq_13, M33_dqdq_14},
                                {M33_dqdq_12, M33_dqdq_22, M33_dqdq_23, M33_dqdq_24},
                                {M33_dqdq_13, M33_dqdq_23, M33_dqdq_33, M33_dqdq_34},
                                {M33_dqdq_14, M33_dqdq_24, M33_dqdq_34, M33_dqdq_44}};

    xt::xarray<double> RDRT_dqdq = xt::zeros<double>({6,4,4});
    xt::view(RDRT_dqdq, 0, xt::all(), xt::all()) = M11_dqdq;
    xt::view(RDRT_dqdq, 1, xt::all(), xt::all()) = M12_dqdq;
    xt::view(RDRT_dqdq, 2, xt::all(), xt::all()) = M13_dqdq;
    xt::view(RDRT_dqdq, 3, xt::all(), xt::all()) = M22_dqdq;
    xt::view(RDRT_dqdq, 4, xt::all(), xt::all()) = M23_dqdq;
    xt::view(RDRT_dqdq, 5, xt::all(), xt::all()) = M33_dqdq;

    return RDRT_dqdq;
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientEllipsoids(const xt::xarray<double>& a,
    const xt::xarray<double>& q, const xt::xarray<double>& D, const xt::xarray<double>& R,
    const xt::xarray<double>& B, const xt::xarray<double>& b){
    // calculate the gradient of the ellipsoid function at p_riomon wrt x = [qx, qy, qz, qw, a1, a2, a3]
    // a: center of the ellipsoid, dimension 3
    // q: input vector of dimension 4
    // D: diagonal matrix of dimension 3 x 3
    // R: rotation matrix of dimension 3 x 3
    // B: real symmetric quadratic coefficient matrix, dimension 3 x 3
    // b: input vector of dimension 3

    int dim_p = 3, dim_y = 9, dim_x = 7, dim_A_flat = 6, dim_q = 4;
    xt::xarray<double> A = xt::linalg::dot(R, xt::linalg::dot(D, xt::transpose(R))); // shape dim_p x dim_p
    xt::xarray<double> p = rimonMethodXtensor(A, a, B, b); // shape dim_p
    double F1 = ellipsoid_F(p, a, A); // scalar
    xt::xarray<double> F1_dp = ellipsoid_dp(p, a, A); // shape dim_p
    xt::xarray<double> F2_dp = ellipsoid_dp(p, b, B); // shape dim_p
    xt::xarray<double> F1_dy = ellipsoid_dy(p, a, A); // shape dim_y
    xt::xarray<double> F2_dy = xt::zeros<double>({dim_y});
    xt::xarray<double> F1_dpdp = ellipsoid_dpdp(A); // shape dim_p x dim_p
    xt::xarray<double> F2_dpdp = ellipsoid_dpdp(B); // shape dim_p x dim_p
    xt::xarray<double> F1_dpdy = ellipsoid_dpdy(p, a, A); // shape dim_p x dim_y
    xt::xarray<double> F2_dpdy = xt::zeros<double>({dim_p, dim_y});
    double dual_var = getDualVariable(F1_dp, F2_dp);
    xt::xarray<double> alpha_dy = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dy, F2_dy, F1_dpdp, F2_dpdp, F1_dpdy, F2_dpdy);

    xt::xarray<double> M_dq = ellipsoid_RDRT_dq(q, D, R); // shape dim_A_flat x dim_q
    xt::xarray<double> y_dx = xt::zeros<double>({dim_y, dim_x}); // shape dim_y x dim_x
    xt::view(y_dx, xt::range(0, dim_A_flat), xt::range(0, dim_q)) = M_dq;
    y_dx(6,4) = 1;
    y_dx(7,5) = 1;
    y_dx(8,6) = 1;

    xt::xarray<double> alpha_dx = xt::linalg::dot(alpha_dy, y_dx); // shape dim_x

    return std::make_tuple(F1, p, alpha_dx);
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipsoids(
    const xt::xarray<double>& a, const xt::xarray<double>& q, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b){
    // calculate the gradient and hessian of the ellipsoid function at p_riomon wrt x = [qx, qy, qz, qw, a1, a2, a3]
    // a: center of the ellipsoid, dimension 3
    // q: input vector of dimension 4
    // D: diagonal matrix of dimension 3 x 3
    // R: rotation matrix of dimension 3 x 3
    // B: real symmetric quadratic coefficient matrix, dimension 3 x 3
    // b: input vector of dimension 3

    int dim_p = 3, dim_y = 9, dim_x = 7, dim_A_flat = 6, dim_q = 4;
    xt::xarray<double> A = xt::linalg::dot(R, xt::linalg::dot(D, xt::transpose(R))); // shape dim_p x dim_p
    xt::xarray<double> p = rimonMethodXtensor(A, a, B, b); // shape dim_p
    double F1 = ellipsoid_F(p, a, A); // scalar
    xt::xarray<double> F1_dp = ellipsoid_dp(p, a, A); // shape dim_p
    xt::xarray<double> F2_dp = ellipsoid_dp(p, b, B); // shape dim_p
    xt::xarray<double> F1_dy = ellipsoid_dy(p, a, A); // shape dim_y
    xt::xarray<double> F2_dy = xt::zeros<double>({dim_y});
    xt::xarray<double> F1_dpdp = ellipsoid_dpdp(A); // shape dim_p x dim_p
    xt::xarray<double> F2_dpdp = ellipsoid_dpdp(B); // shape dim_p x dim_p
    xt::xarray<double> F1_dpdy = ellipsoid_dpdy(p, a, A); // shape dim_p x dim_y
    xt::xarray<double> F2_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> F1_dydy = ellipsoid_dydy(p, a, A); // shape dim_x x dim_x
    xt::xarray<double> F2_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> F1_dpdpdp = ellipsoid_dpdpdp(); // shape dim_p x dim_p x dim_p
    xt::xarray<double> F2_dpdpdp = ellipsoid_dpdpdp(); // shape dim_p x dim_p x dim_p
    xt::xarray<double> F1_dpdpdy = ellipsoid_dpdpdy(A); // shape dim_p x dim_p x dim_y
    xt::xarray<double> F2_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    xt::xarray<double> F1_dpdydy = ellipsoid_dpdydy(A); // shape dim_p x dim_y x dim_y
    xt::xarray<double> F2_dpdydy = xt::zeros<double>({dim_p, dim_y, dim_y});

    double dual_var = getDualVariable(F1_dp, F2_dp);
    xt::xarray<double> alpha_dy, alpha_dydy;
    std::tie(alpha_dy, alpha_dydy) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dy, F2_dy, F1_dpdp, F2_dpdp, F1_dpdy, F2_dpdy, F1_dydy, F2_dydy,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdy, F2_dpdpdy, F1_dpdydy, F2_dpdydy);

    xt::xarray<double> M_dq = ellipsoid_RDRT_dq(q, D, R); // shape dim_A_flat x dim_q
    xt::xarray<double> M_dqdq = ellipsoid_RDRT_dqdq(q, D); // shape dim_A_flat x dim_q x dim_q

    xt::xarray<double> y_dx = xt::zeros<double>({dim_y, dim_x}); // shape dim_y x dim_x
    xt::view(y_dx, xt::range(0, dim_A_flat), xt::range(0, dim_q)) = M_dq;
    y_dx(6,4) = 1;
    y_dx(7,5) = 1;
    y_dx(8,6) = 1;

    xt::xarray<double> alpha_dx = xt::linalg::dot(alpha_dy, y_dx); // shape dim_x

    xt::xarray<double> y_dxdx = xt::zeros<double>({dim_y, dim_x, dim_x}); // shape dim_y x dim_x x dim_x
    xt::view(y_dxdx, xt::range(0, dim_A_flat), xt::range(0, dim_q), xt::range(0, dim_q)) = M_dqdq;

    xt::xarray<double> alpha_dxdx = xt::linalg::dot(xt::transpose(y_dx), xt::linalg::dot(alpha_dydy, y_dx));
    alpha_dxdx += xt::linalg::tensordot(alpha_dy, y_dxdx, {0}, {0});

    return std::make_tuple(F1, p, alpha_dx, alpha_dxdx);
}
