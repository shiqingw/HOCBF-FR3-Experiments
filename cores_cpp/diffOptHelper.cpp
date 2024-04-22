#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <tuple>

double getDualVariable(const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp){
    double F1_dp_norm = xt::linalg::norm(F1_dp, 2);
    double F2_dp_norm = xt::linalg::norm(F2_dp, 2);

    return F1_dp_norm/F2_dp_norm;
}

xt::xarray<double> getGradientGeneral(double dual_var,
    const xt::xarray<double>& F1_dp,const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx){
    
    int dim_p = xt::adapt(F1_dpdx.shape())(0);
    xt::xarray<double> b1 = - F1_dpdx - dual_var * F2_dpdx; // shape (dim_p, dim_x)
    xt::xarray<double> b2 = xt::view(-F2_dx, xt::newaxis(), xt::all()); // shape (1, dim_x)
    xt::xarray<double> b = xt::concatenate(xt::xtuple(b1, b2), 0); // shape (dim_p+1, dim_x)

    xt::xarray<double> A11 = F1_dpdp + dual_var * F2_dpdp; // shape (dim_p, dim_p)
    xt::xarray<double> A = xt::zeros<double>({dim_p+1, dim_p+1});
    xt::view(A, xt::range(0, dim_p), xt::range(0, dim_p)) = A11;
    xt::view(A, xt::range(0, dim_p), dim_p) = F2_dp;
    xt::view(A, dim_p, xt::range(0, dim_p)) = F2_dp;

    xt::xarray<double> grad = xt::linalg::solve(A, b); // shape (dim_p+1, dim_x)
    xt::xarray<double> p_dx = xt::view(grad, xt::range(0, dim_p), xt::all());
    xt::xarray<double> alpha_dx = F1_dx + xt::linalg::dot(F1_dp, p_dx); // shape (dim_x)

    return alpha_dx;
}

std::tuple<xt::xarray<double>,xt::xarray<double>> getGradientAndHessianGeneral(double dual_var,
    const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx,
    const xt::xarray<double>& F1_dxdx, const xt::xarray<double>& F2_dxdx,
    const xt::xarray<double>& F1_dpdpdp, const xt::xarray<double>& F2_dpdpdp,
    const xt::xarray<double>& F1_dpdpdx, const xt::xarray<double>& F2_dpdpdx,
    const xt::xarray<double>& F1_dpdxdx, const xt::xarray<double>& F2_dpdxdx){
    
    int dim_p = xt::adapt(F1_dpdx.shape())(0);
    int dim_x = xt::adapt(F1_dpdx.shape())(1);
    xt::xarray<double> b1 = - F1_dpdx - dual_var * F2_dpdx; // shape (dim_p, dim_x)
    xt::xarray<double> b2 = xt::view(-F2_dx, xt::newaxis(), xt::all()); // shape (1, dim_x)
    xt::xarray<double> b = xt::concatenate(xt::xtuple(b1, b2), 0); // shape (dim_p+1, dim_x)

    xt::xarray<double> A11 = F1_dpdp + dual_var * F2_dpdp; // shape (dim_p, dim_p)
    xt::xarray<double> A = xt::zeros<double>({dim_p+1, dim_p+1});
    xt::view(A, xt::range(0, dim_p), xt::range(0, dim_p)) = A11;
    xt::view(A, xt::range(0, dim_p), dim_p) = F2_dp;
    xt::view(A, dim_p, xt::range(0, dim_p)) = F2_dp;

    xt::xarray<double> grad = xt::linalg::solve(A, b); // shape (dim_p+1, dim_x)
    xt::xarray<double> p_dx = xt::view(grad, xt::range(0, dim_p), xt::all()); // shape (dim_p, dim_x)
    xt::xarray<double> dual_dx = xt::view(grad, dim_p, xt::all()); // shape (dim_x)
    xt::xarray<double> alpha_dx = F1_dx + xt::linalg::dot(F1_dp, p_dx); // shape (dim_x)

    xt::xarray<double> F1_dpdxdp = xt::transpose(F1_dpdpdx, {0, 2, 1}); // shape (dim_p, dim_x, dim_p)
    xt::xarray<double> F2_dpdxdp = xt::transpose(F2_dpdpdx, {0, 2, 1}); // shape (dim_p, dim_x, dim_p)
    xt::xarray<double> b1_dx_1 = -F1_dpdxdx - xt::linalg::tensordot(F1_dpdxdp, p_dx, {2}, {0}); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> tmp1 = F2_dpdxdx + xt::linalg::tensordot(F2_dpdxdp, p_dx, {2}, {0}); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> b1_dx_2 = - xt::linalg::tensordot(xt::view(F2_dpdx, xt::all(), xt::all(), xt::newaxis()),
                                xt::view(dual_dx, xt::newaxis(), xt::all()), {2}, {0}) - dual_var*tmp1; // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> b1_dx = b1_dx_1 + b1_dx_2; // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> F2_dxdp = xt::transpose(F2_dpdx); // shape (dim_x, dim_p)
    xt::xarray<double> b2_dx = -F2_dxdx - xt::linalg::dot(F2_dxdp, p_dx); // shape (dim_x, dim_x)
    xt::xarray<double> b_dx = xt::concatenate(xt::xtuple(b1_dx, xt::view(b2_dx, xt::newaxis(), xt::all(), xt::all())), 0); // shape (dim_p+1, dim_x, dim_x)

    xt::xarray<double> A11_dx_1 = F1_dpdpdx + xt::linalg::tensordot(F1_dpdpdp, p_dx, {2}, {0}); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> tmp2 = F2_dpdpdx + xt::linalg::tensordot(F2_dpdpdp, p_dx, {2}, {0}); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> A11_dx_2 = xt::linalg::tensordot(xt::view(F2_dpdp, xt::all(), xt::all(), xt::newaxis()),
                                xt::view(dual_dx, xt::newaxis(), xt::all()), {2}, {0}) + dual_var*tmp2; // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> A11_dx = A11_dx_1 + A11_dx_2; // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> A21_dx = F2_dpdx + xt::linalg::dot(F2_dpdp, p_dx); // shape (dim_p, dim_x)
    xt::xarray<double> A_dx = xt::zeros<double>({dim_p+1, dim_p+1, dim_x});
    xt::view(A_dx, xt::range(0, dim_p), xt::range(0, dim_p), xt::all()) = A11_dx;
    xt::view(A_dx, dim_p, xt::range(0, dim_p), xt::all()) = A21_dx;
    xt::view(A_dx, xt::range(0, dim_p), dim_p, xt::all()) = A21_dx;

    xt::xarray<double> hessian = xt::zeros<double>({dim_p+1, dim_x, dim_x});
    for (int i = 0; i < dim_x; ++i){
        xt::xarray<double> A_dx_i = xt::view(A_dx, xt::all(), xt::all(), i); // shape (dim_p+1, dim_p+1)
        xt::xarray<double> b_dx_i = xt::view(b_dx, xt::all(), xt::all(), i); // shape (dim_p+1, dim_x)
        xt::view(hessian, xt::all(), xt::all(), i) = - xt::linalg::solve(A, xt::linalg::dot(A_dx_i, grad)) + xt::linalg::solve(A, b_dx_i);
    }
    xt::xarray<double> F1_dxdp = xt::transpose(F1_dpdx); // shape (dim_x, dim_p)
    xt::xarray<double> p_dxdx = xt::view(hessian, xt::range(0, dim_p), xt::all(), xt::all()); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> alpha_dxdx = F1_dxdx + xt::linalg::dot(F1_dxdp, p_dx) + xt::linalg::dot(xt::transpose(p_dx), F1_dpdx); // shape (dim_x, dim_x)
    alpha_dxdx += xt::linalg::dot(xt::transpose(p_dx), xt::linalg::dot(F1_dpdp, p_dx))
                    + xt::linalg::tensordot(F1_dp, p_dxdx, {0}, {0}); // shape (dim_x, dim_x)
    
    return std::make_tuple(alpha_dx, alpha_dxdx);
}