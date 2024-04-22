#include <xtensor/xarray.hpp>
#include <tuple>
double getDualVariable(const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp);
xt::xarray<double> getGradientGeneral(double dual_var, const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx);
std::tuple<xt::xarray<double>,xt::xarray<double>> getGradientAndHessianGeneral(double dual_var, const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx,
    const xt::xarray<double>& F1_dxdx, const xt::xarray<double>& F2_dxdx,
    const xt::xarray<double>& F1_dpdpdp, const xt::xarray<double>& F2_dpdpdp,
    const xt::xarray<double>& F1_dpdpdx, const xt::xarray<double>& F2_dpdpdx,
    const xt::xarray<double>& F1_dpdxdx, const xt::xarray<double>& F2_dpdxdx);