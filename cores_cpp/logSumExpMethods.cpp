#include <cmath>
#include <tuple>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "diffOptHelper.h"
#include "ellipsoidMethods.h"
#include "ellipseMethods.h"


std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>
getLogSumExpDerivatives(const xt::xarray<double>& p, const xt::xarray<double>& A,
const xt::xarray<double>& b, const double kappa) {
    // Compute F(p) = log[sum(exp[k(A p + b)])/len(A)] + 1
    // p: input vector of dimension 2 or 3
    // A: real matrix of dimension N x dim(p)
    // b: real vector of dimension N

    int dim_p = p.shape()[0], dim_z = A.shape()[0];
    xt::xarray<double> z = kappa * (xt::linalg::dot(A, p) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();
    double F = log(sum_z) + c - log(double(dim_z)) + 1;

    xt::xarray<double> zT_A = xt::linalg::dot(z, A); // shape dim_p
    xt::xarray<double> F_dp = kappa * zT_A / sum_z; // shape dim_p

    xt::xarray<double> diag_z = xt::diag(z);
    xt::xarray<double> diag_z_A = xt::linalg::dot(diag_z, A);
    xt::xarray<double> AT_diag_z_A = xt::linalg::dot(xt::transpose(A), diag_z_A);
    xt::xarray<double> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
    xt::xarray<double> F_dpdp = pow(kappa,2) * (AT_diag_z_A/sum_z - AT_z_zT_A/pow(sum_z,2));

    xt::xarray<double> F_dpdpdp = xt::zeros<double>({dim_p, dim_p, dim_p});
    xt::xarray<double> z_dp = kappa * diag_z_A;
    // part 1 of F_dpdpdp
    for (int i = 0; i < dim_p; ++i){
        xt::xarray<double> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xarray<double> tmp = pow(kappa,2)*xt::linalg::dot(xt::transpose(A), xt::linalg::dot(xt::diag(z_dp_i) ,A))/sum_z;
        tmp -= pow(kappa,2) * xt::sum(z_dp_i)() * AT_diag_z_A/pow(sum_z,2);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) += tmp; 
    }
    // part 2 of F_dpdpdp
    xt::xarray<double> big = xt::zeros<double>({dim_z, dim_z, dim_z});
    xt::xarray<double> identity = xt::eye(dim_z);
    for (int i = 0; i < dim_z; ++i){
        xt::xarray<double> e_i = xt::view(identity, i, xt::all());
        xt::view(big, xt::all(), xt::all(), i) += xt::linalg::outer(e_i, z) + xt::linalg::outer(z, e_i);
    }

    for (int i = 0; i < dim_p; ++i){
        xt::xarray<double> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xarray<double> tmp = xt::linalg::tensordot(big, z_dp_i, {2}, {0}); // shape dim_z x dim_z
        tmp = pow(kappa,2) * xt::linalg::dot(xt::transpose(A), xt::linalg::dot(tmp, A))/ pow(sum_z,2); // shape dim_p x dim_p
        tmp -= 2 * pow(kappa,2) * xt::sum(z_dp_i)() * AT_z_zT_A / pow(sum_z,3);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) -= tmp;
    }

    return std::make_tuple(F, F_dp, F_dpdp, F_dpdpdp);
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipsoidAndLogSumExp(
    const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& q, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b, const double kappa){
    // calculate the gradient and hessian of the ellipsoid function at p wrt x = [qx, qy, qz, qw, a1, a2, a3]
    // a: center of the ellipsoid, dimension 3
    // q: input vector of dimension 4
    // D: diagonal matrix of dimension 3 x 3
    // R: rotation matrix of dimension 3 x 3
    // B: coefficient matrix of logSumExp function of dimension N x 3
    // b: coefficient matrix of logSumExp function of dimension N
    // kappa: scalar

    int dim_p = 3, dim_y = 9, dim_x = 7, dim_A_flat = 6, dim_q = 4;
    xt::xarray<double> A = xt::linalg::dot(R, xt::linalg::dot(D, xt::transpose(R))); // shape dim_p x dim_p
    double F1 = ellipsoid_F(p, a, A); // scalar
    xt::xarray<double> F1_dp = ellipsoid_dp(p, a, A); // shape dim_p
    xt::xarray<double> F1_dy = ellipsoid_dy(p, a, A); // shape dim_y
    xt::xarray<double> F2_dy = xt::zeros<double>({dim_y});
    xt::xarray<double> F1_dpdp = ellipsoid_dpdp(A); // shape dim_p x dim_p
    xt::xarray<double> F1_dpdy = ellipsoid_dpdy(p, a, A); // shape dim_p x dim_y
    xt::xarray<double> F2_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> F1_dydy = ellipsoid_dydy(p, a, A); // shape dim_x x dim_x
    xt::xarray<double> F2_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> F1_dpdpdp = ellipsoid_dpdpdp(); // shape dim_p x dim_p x dim_p
    xt::xarray<double> F1_dpdpdy = ellipsoid_dpdpdy(A); // shape dim_p x dim_p x dim_y
    xt::xarray<double> F2_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    xt::xarray<double> F1_dpdydy = ellipsoid_dpdydy(A); // shape dim_p x dim_y x dim_y
    xt::xarray<double> F2_dpdydy = xt::zeros<double>({dim_p, dim_y, dim_y});

    double F2;
    xt::xarray<double> F2_dp, F2_dpdp, F2_dpdpdp;
    std::tie(F2, F2_dp, F2_dpdp, F2_dpdpdp) = getLogSumExpDerivatives(p, B, b, kappa);

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

    return std::make_tuple(F1, alpha_dx, alpha_dxdx);
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipseAndLogSumExp(
    const xt::xarray<double>& p, const xt::xarray<double>& a, const double theta, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b, const double kappa){
    // calculate the gradient and hessian of the ellipsoid function at p wrt x = [theta, a1, a2]
    // a: center of the ellipsoid, dimension 2
    // theta: orientation
    // D: diagonal matrix of dimension 2 x 2
    // R: rotation matrix of dimension 2 x 2
    // B: coefficient matrix of logSumExp function of dimension N x 2
    // b: coefficient matrix of logSumExp function of dimension N
    // kappa: scalar

    int dim_p = 2, dim_y = 5, dim_x = 3, dim_A_flat = 3, dim_theta = 1;
    xt::xarray<double> A = xt::linalg::dot(R, xt::linalg::dot(D, xt::transpose(R))); // shape dim_p x dim_p
    double F1 = ellipse_F(p, a, A); // scalar
    xt::xarray<double> F1_dp = ellipse_dp(p, a, A); // shape dim_p
    xt::xarray<double> F1_dy = ellipse_dy(p, a, A); // shape dim_y
    xt::xarray<double> F2_dy = xt::zeros<double>({dim_y});
    xt::xarray<double> F1_dpdp = ellipse_dpdp(A); // shape dim_p x dim_p
    xt::xarray<double> F1_dpdy = ellipse_dpdy(p, a, A); // shape dim_p x dim_y
    xt::xarray<double> F2_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> F1_dydy = ellipse_dydy(p, a, A); // shape dim_x x dim_x
    xt::xarray<double> F2_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> F1_dpdpdp = ellipse_dpdpdp(); // shape dim_p x dim_p x dim_p
    xt::xarray<double> F1_dpdpdy = ellipse_dpdpdy(A); // shape dim_p x dim_p x dim_y
    xt::xarray<double> F2_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    xt::xarray<double> F1_dpdydy = ellipse_dpdydy(A); // shape dim_p x dim_y x dim_y
    xt::xarray<double> F2_dpdydy = xt::zeros<double>({dim_p, dim_y, dim_y});

    double F2;
    xt::xarray<double> F2_dp, F2_dpdp, F2_dpdpdp;
    std::tie(F2, F2_dp, F2_dpdp, F2_dpdpdp) = getLogSumExpDerivatives(p, B, b, kappa);

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

    return std::make_tuple(F1, alpha_dx, alpha_dxdx);
}