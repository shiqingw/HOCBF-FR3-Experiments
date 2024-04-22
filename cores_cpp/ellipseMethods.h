#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
double ellipse_F(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipse_dp(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipse_dpdp(const xt::xarray<double>& A);
xt::xarray<double> ellipse_dpdpdp();
xt::xarray<double> ellipse_dy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipse_dpdy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipse_dydy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipse_dpdpdy(const xt::xarray<double>& A);
xt::xarray<double> ellipse_dpdydy(const xt::xarray<double>& A);
xt::xarray<double> ellipse_RDRT_dtheta(const double theta, const xt::xarray<double>& D);
xt::xarray<double> ellipse_RDRT_dthetadtheta(const double theta, const xt::xarray<double>& D);
std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientEllipses(const xt::xarray<double>& a,
    const double theta, const xt::xarray<double>& D, const xt::xarray<double>& R,
    const xt::xarray<double>& B, const xt::xarray<double>& b);
std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipses(
    const xt::xarray<double>& a, const double theta, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b);