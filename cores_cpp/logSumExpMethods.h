#include <tuple>
#include <xtensor/xarray.hpp>
std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>
getLogSumExpDerivatives(const xt::xarray<double>& p, const xt::xarray<double>& A,
const xt::xarray<double>& b, const double kappa);

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipsoidAndLogSumExp(
    const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& q, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b, const double kappa);

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipseAndLogSumExp(
    const xt::xarray<double>& p, const xt::xarray<double>& a, const double theta, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b, const double kappa);