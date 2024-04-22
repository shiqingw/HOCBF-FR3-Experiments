#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
Eigen::VectorXd rimonMethod(const Eigen::MatrixXd& A, const Eigen::VectorXd& a, const Eigen::MatrixXd& B, const Eigen::VectorXd& b);
xt::xarray<double> rimonMethodXtensor(const xt::xarray<double>& A, const xt::xarray<double>& a, const xt::xarray<double>& B, const xt::xarray<double>& b);
double ellipsoid_F(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dp(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dpdp(const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dpdpdp();
xt::xarray<double> ellipsoid_dy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dpdy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dydy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dpdpdy(const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_dpdydy(const xt::xarray<double>& A);
xt::xarray<double> ellipsoid_RDRT_dq(const xt::xarray<double>& q, const xt::xarray<double>& D, const xt::xarray<double>& R);
xt::xarray<double> ellipsoid_RDRT_dqdq(const xt::xarray<double>& q, const xt::xarray<double>& D);
std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientEllipsoids(const xt::xarray<double>& a,
    const xt::xarray<double>& q, const xt::xarray<double>& D, const xt::xarray<double>& R,
    const xt::xarray<double>& B, const xt::xarray<double>& b);
std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipsoids(
    const xt::xarray<double>& a, const xt::xarray<double>& q, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b);