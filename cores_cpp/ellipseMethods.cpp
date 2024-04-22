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
#include "ellipsoidMethods.h"

/*
    We consider F(p) = (p - a)^T @ R @ D @ R.T @ (p - a) where a is a vector and
    R = [[cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]].
    D = diag(D11, D22).
    Note A = R @ D @ R.T for simplicity.
    The parameters of F are x = [theta, ax, ay]. This class calculates
    various derivatives of F w.r.t. x and p.
*/

double ellipse_F(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute F(p)
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2

    return xt::linalg::dot(p - a, xt::linalg::dot(A, p - a))(0);
}

xt::xarray<double> ellipse_dp(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute dF/dp
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2

    return 2 * xt::linalg::dot(A, p - a); // Since A is symmetric, A.transpose() is the same as A
}

xt::xarray<double> ellipse_dpdp(const xt::xarray<double>& A) {
    // Compute d^2F/dpdp
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2

    return 2 * A; // Since A is symmetric, A.transpose() is the same as A
}

xt::xarray<double> ellipse_dpdpdp() {
    // Compute d^3F/dpdpdp

    return xt::zeros<double>({2, 2, 2});;
}

xt::xarray<double> ellipse_dy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute dF/dy, where y = [A11, A12, A22, a1, a2]
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2
    
    int dim_y = 5, dim_A_flat = 3;
    xt::xarray<double> vector = a - p;
    xt::xarray<double> outer_prod = xt::linalg::outer(vector, vector);
    xt::xarray<double> F_dy = xt::zeros<double>({dim_y});
    F_dy(0) = outer_prod(0, 0);
    F_dy(1) = 2 * outer_prod(0, 1);
    F_dy(2) = outer_prod(1, 1);
    xt::view(F_dy, xt::range(dim_A_flat, dim_y)) = 2 * xt::linalg::dot(A, vector);

    return F_dy;
}

xt::xarray<double> ellipse_dpdy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute d^2F/dpdy, where y = [A11, A12, A22, a1, a2]
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2
    int dim_p = 2, dim_y = 5, dim_A_flat = 3;

    xt::xarray<double> F_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> vector = p - a;
    F_dpdy(0,0) = 2 * vector(0);
    F_dpdy(0,1) = 2 * vector(1);
    F_dpdy(1,1) = 2 * vector(0);
    F_dpdy(1,2) = 2 * vector(1);
    xt::view(F_dpdy, xt::all(), xt::range(dim_A_flat, dim_y)) = -2 * A;

    return F_dpdy;
}

xt::xarray<double> ellipse_dydy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute d^2F/dydy, where y = [A11, A12, A22, a1, a2]
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2
    int dim_p = 2, dim_y = 5, dim_A_flat = 3;
    xt::xarray<double> F_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> vector = a - p;
    xt::xarray<double> tmp = xt::zeros<double>({dim_p, dim_A_flat});
    tmp(0,0) = 2 * vector(0);
    tmp(0,1) = 2 * vector(1);
    tmp(1,1) = 2 * vector(0);
    tmp(1,2) = 2 * vector(1);
    xt::view(F_dydy, xt::range(dim_A_flat, dim_y), xt::range(0, dim_A_flat)) = tmp;
    xt::view(F_dydy, xt::range(0, dim_A_flat), xt::range(dim_A_flat, dim_y)) = xt::transpose(tmp);
    xt::view(F_dydy, xt::range(dim_A_flat, dim_y), xt::range(dim_A_flat, dim_y)) = 2 * A;

    return F_dydy;
}

xt::xarray<double> ellipse_dpdpdy(const xt::xarray<double>& A) {
    // Compute d^3F/dpdpdy, where y = [A11, A12, A22, a1, a2]
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2
    int dim_p = 2, dim_y = 5;
    xt::xarray<double> F_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    F_dpdpdy(0,0,0) = 2;
    F_dpdpdy(0,1,1) = 2;
    F_dpdpdy(1,0,1) = 2;
    F_dpdpdy(1,1,2) = 2;

    return F_dpdpdy;
}

xt::xarray<double> ellipse_dpdydy(const xt::xarray<double>& A) {
    // Compute d^3F/dpdpdy, where y = [A11, A12, A22, a1, a2]
    // p: input vector of dimension 2
    // a: center of the ellipse, dimension 2
    // A: real symmetric quadratic coefficient matrix, dimension 2 x 2
    int dim_p = 2, dim_y = 5, dim_A_flat = 3;
    xt::xarray<double> F_dpdydy = xt::zeros<double>({dim_p, dim_y, dim_y});
    xt::xarray<double> tmp = xt::zeros<double>({dim_p, dim_p, dim_A_flat});
    tmp(0,0,0) = -2;
    tmp(0,1,1) = -2;
    tmp(1,0,1) = -2;
    tmp(1,1,2) = -2;

    xt::view(F_dpdydy, xt::all(), xt::range(dim_A_flat, dim_y), xt::range(0, dim_A_flat)) = tmp;
    xt::view(F_dpdydy, xt::all(), xt::range(0, dim_A_flat), xt::range(dim_A_flat, dim_y)) = xt::transpose(tmp, {0,2,1});

    return F_dpdydy;
}

xt::xarray<double> ellipse_RDRT_dtheta(const double theta, const xt::xarray<double>& D){
    // Compute d(R @ D @ R.T)/dtheta
    // theta: orientation
    // D: diagonal matrix of dimension 2 x 2

    double a = D(0,0), b = D(1,1);

    xt::xarray<double> RDRT_dtheta {(-a + b)*sin(2*theta),
                                    (a - b)*cos(2*theta),
                                    (a - b)*sin(2*theta)};
    RDRT_dtheta = xt::view(RDRT_dtheta, xt::all(), xt::newaxis());
    return RDRT_dtheta;
}

xt::xarray<double> ellipse_RDRT_dthetadtheta(const double theta, const xt::xarray<double>& D){
    // Compute d^2(R @ D @ R.T)/dqdq
    // theta: orientation
    // D: diagonal matrix of dimension 2 x 2

    double a = D(0,0), b = D(1,1);

    xt::xarray<double> RDRT_dthetadtheta {2*(-a + b)*cos(2*theta),
                                            2*(-a + b)*sin(2*theta),
                                            2*(a - b)*cos(2*theta)};
    RDRT_dthetadtheta = xt::view(RDRT_dthetadtheta, xt::all(), xt::newaxis(), xt::newaxis());
    return RDRT_dthetadtheta;
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientEllipses(const xt::xarray<double>& a,
    const double theta, const xt::xarray<double>& D, const xt::xarray<double>& R,
    const xt::xarray<double>& B, const xt::xarray<double>& b){
    // calculate the gradient of the ellipse function at p_riomon wrt x = [theta, a1, a2]
    // a: center of the ellipse, dimension 2
    // theta: orientation
    // D: diagonal matrix of dimension 2 x 2
    // R: rotation matrix of dimension 2 x 2
    // B: real symmetric quadratic coefficient matrix, dimension 2 x 2
    // b: input vector of dimension 2

    int dim_p = 2, dim_y = 5, dim_x = 3, dim_A_flat = 3, dim_theta = 1;
    xt::xarray<double> A = xt::linalg::dot(R, xt::linalg::dot(D, xt::transpose(R))); // shape dim_p x dim_p
    xt::xarray<double> p = rimonMethodXtensor(A, a, B, b); // shape dim_p
    double F1 = ellipse_F(p, a, A); // scalar
    xt::xarray<double> F1_dp = ellipse_dp(p, a, A); // shape dim_p
    xt::xarray<double> F2_dp = ellipse_dp(p, b, B); // shape dim_p
    xt::xarray<double> F1_dy = ellipse_dy(p, a, A); // shape dim_y
    xt::xarray<double> F2_dy = xt::zeros<double>({dim_y});
    xt::xarray<double> F1_dpdp = ellipse_dpdp(A); // shape dim_p x dim_p
    xt::xarray<double> F2_dpdp = ellipse_dpdp(B); // shape dim_p x dim_p
    xt::xarray<double> F1_dpdy = ellipse_dpdy(p, a, A); // shape dim_p x dim_y
    xt::xarray<double> F2_dpdy = xt::zeros<double>({dim_p, dim_y});
    double dual_var = getDualVariable(F1_dp, F2_dp);
    xt::xarray<double> alpha_dy = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dy, F2_dy, F1_dpdp, F2_dpdp, F1_dpdy, F2_dpdy);

    xt::xarray<double> M_dq = ellipse_RDRT_dtheta(theta, D); // shape dim_A_flat x dim_theta
    xt::xarray<double> y_dx = xt::zeros<double>({dim_y, dim_x}); // shape dim_y x dim_x
    xt::view(y_dx, xt::range(0, dim_A_flat), xt::range(0, dim_theta)) = M_dq;
    y_dx(3,1) = 1;
    y_dx(4,2) = 1;

    xt::xarray<double> alpha_dx = xt::linalg::dot(alpha_dy, y_dx); // shape dim_x

    return std::make_tuple(F1, p, alpha_dx);
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipses(
    const xt::xarray<double>& a, const double theta, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b){
    // calculate the gradient and hessian of the ellipse function at p_riomon wrt x = [theta, a1, a2]
    // a: center of the ellipse, dimension 2
    // theta: orientation
    // D: diagonal matrix of dimension 2 x 2
    // R: rotation matrix of dimension 2 x 2
    // B: real symmetric quadratic coefficient matrix, dimension 2 x 2
    // b: input vector of dimension 2

    int dim_p = 2, dim_y = 5, dim_x = 3, dim_A_flat = 3, dim_theta = 1;
    xt::xarray<double> A = xt::linalg::dot(R, xt::linalg::dot(D, xt::transpose(R))); // shape dim_p x dim_p
    xt::xarray<double> p = rimonMethodXtensor(A, a, B, b); // shape dim_p
    double F1 = ellipse_F(p, a, A); // scalar
    xt::xarray<double> F1_dp = ellipse_dp(p, a, A); // shape dim_p
    xt::xarray<double> F2_dp = ellipse_dp(p, b, B); // shape dim_p
    xt::xarray<double> F1_dy = ellipse_dy(p, a, A); // shape dim_y
    xt::xarray<double> F2_dy = xt::zeros<double>({dim_y});
    xt::xarray<double> F1_dpdp = ellipse_dpdp(A); // shape dim_p x dim_p
    xt::xarray<double> F2_dpdp = ellipse_dpdp(B); // shape dim_p x dim_p
    xt::xarray<double> F1_dpdy = ellipse_dpdy(p, a, A); // shape dim_p x dim_y
    xt::xarray<double> F2_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> F1_dydy = ellipse_dydy(p, a, A); // shape dim_x x dim_x
    xt::xarray<double> F2_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> F1_dpdpdp = ellipse_dpdpdp(); // shape dim_p x dim_p x dim_p
    xt::xarray<double> F2_dpdpdp = ellipse_dpdpdp(); // shape dim_p x dim_p x dim_p
    xt::xarray<double> F1_dpdpdy = ellipse_dpdpdy(A); // shape dim_p x dim_p x dim_y
    xt::xarray<double> F2_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    xt::xarray<double> F1_dpdydy = ellipse_dpdydy(A); // shape dim_p x dim_y x dim_y
    xt::xarray<double> F2_dpdydy = xt::zeros<double>({dim_p, dim_y, dim_y});

    double dual_var = getDualVariable(F1_dp, F2_dp);
    xt::xarray<double> alpha_dy, alpha_dydy;
    std::tie(alpha_dy, alpha_dydy) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dy, F2_dy, F1_dpdp, F2_dpdp, F1_dpdy, F2_dpdy, F1_dydy, F2_dydy,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdy, F2_dpdpdy, F1_dpdydy, F2_dpdydy);

    xt::xarray<double> M_dq = ellipse_RDRT_dtheta(theta, D); // shape dim_A_flat x dim_theta
    xt::xarray<double> M_dqdq = ellipse_RDRT_dthetadtheta(theta, D); // shape dim_A_flat x dim_theta x dim_theta

    xt::xarray<double> y_dx = xt::zeros<double>({dim_y, dim_x}); // shape dim_y x dim_x
    xt::view(y_dx, xt::range(0, dim_A_flat), xt::range(0, dim_theta)) = M_dq;
    y_dx(3,1) = 1;
    y_dx(4,2) = 1;

    xt::xarray<double> alpha_dx = xt::linalg::dot(alpha_dy, y_dx); // shape dim_x

    xt::xarray<double> y_dxdx = xt::zeros<double>({dim_y, dim_x, dim_x}); // shape dim_y x dim_x x dim_x
    xt::view(y_dxdx, xt::range(0, dim_A_flat), xt::range(0, dim_theta), xt::range(0, dim_theta)) = M_dqdq;

    xt::xarray<double> alpha_dxdx = xt::linalg::dot(xt::transpose(y_dx), xt::linalg::dot(alpha_dydy, y_dx));
    alpha_dxdx += xt::linalg::tensordot(alpha_dy, y_dxdx, {0}, {0});

    return std::make_tuple(F1, p, alpha_dx, alpha_dxdx);
}
