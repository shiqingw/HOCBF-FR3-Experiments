#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <xtensor-python/pyarray.hpp>

#include "ellipsoidMethods.h"
#include "ellipseMethods.h"
#include "logSumExpMethods.h"
#include "diffOptHelper.h"

namespace py = pybind11;

PYBIND11_MODULE(diffOptCpp, m) {
    xt::import_numpy();
    m.doc() = "diffOptCpp";

    m.def("getDualVariable", &getDualVariable, "getDualVariable based on xtensor.");
    m.def("getGradientGeneral", &getGradientGeneral, "getGradientGeneral based on xtensor.");
    m.def("getGradientAndHessianGeneral", &getGradientAndHessianGeneral, "getGradientAndHessianGeneral based on xtensor.");

    m.def("rimonMethod", &rimonMethod, "Rimon method based on Eigen.");
    m.def("rimonMethodXtensor", &rimonMethodXtensor, "Rimon method based on xtensor.");
    m.def("ellipsoid_F", &ellipsoid_F, "ellipsoid_value based on xtensor.");
    m.def("ellipsoid_dp", &ellipsoid_dp, "ellipsoid_dp based on xtensor.");
    m.def("ellipsoid_dpdp", &ellipsoid_dpdp, "ellipsoid_dpdp based on xtensor.");
    m.def("ellipsoid_dpdpdp", &ellipsoid_dpdpdp, "ellipsoid_dpdpdp based on xtensor.");
    m.def("ellipsoid_dy", &ellipsoid_dy, "ellipsoid_dy based on xtensor.");
    m.def("ellipsoid_dpdy", &ellipsoid_dpdy, "ellipsoid_dpdy based on xtensor.");
    m.def("ellipsoid_dydy", &ellipsoid_dydy, "ellipsoid_dydy based on xtensor.");
    m.def("ellipsoid_dpdpdy", &ellipsoid_dpdpdy, "ellipsoid_dpdpdy based on xtensor.");
    m.def("ellipsoid_dpdydy", &ellipsoid_dpdydy, "ellipsoid_dpdydy based on xtensor.");
    m.def("ellipsoid_RDRT_dq", &ellipsoid_RDRT_dq, "RDRT_dq based on xtensor.");
    m.def("ellipsoid_RDRT_dqdq", &ellipsoid_RDRT_dqdq, "RDRT_dqdq based on xtensor.");
    m.def("getGradientEllipsoids", &getGradientEllipsoids, "getGradientEllipsoid based on xtensor.");
    m.def("getGradientAndHessianEllipsoids", &getGradientAndHessianEllipsoids, "getGradientAndHessianEllipsoid based on xtensor.");

    m.def("getLogSumExpDerivatives", &getLogSumExpDerivatives, "getLogSumExpDerivatives based on xtensor.");
    m.def("getGradientAndHessianEllipsoidAndLogSumExp", &getGradientAndHessianEllipsoidAndLogSumExp, "getGradientAndHessianEllipsoidAndLogSumExp based on xtensor.");
    m.def("getGradientAndHessianEllipseAndLogSumExp", &getGradientAndHessianEllipseAndLogSumExp, "getGradientAndHessianEllipseAndLogSumExp based on xtensor.");

    m.def("ellipse_F", &ellipse_F, "ellipse_F based on xtensor");
    m.def("ellipse_dp", &ellipse_dp, "ellipse_dp based on xtensor");
    m.def("ellipse_dpdp", &ellipse_dpdp, "ellipse_dpdp based on xtensor");
    m.def("ellipse_dpdpdp", &ellipse_dpdpdp, "ellipse_dpdpdp based on xtensor");
    m.def("ellipse_dy", &ellipse_dy, "ellipse_dy based on xtensor");
    m.def("ellipse_dpdy", &ellipse_dpdy, "ellipse_dpdy based on xtensor");
    m.def("ellipse_dydy", &ellipse_dydy, "ellipse_dydy based on xtensor");
    m.def("ellipse_dpdpdy", &ellipse_dpdpdy, "ellipse_dpdpdy based on xtensor");
    m.def("ellipse_dpdydy", &ellipse_dpdydy, "ellipse_dpdydy based on xtensor");
    m.def("ellipse_RDRT_dtheta", &ellipse_RDRT_dtheta, "ellipse_RDRT_dtheta based on xtensor");
    m.def("ellipse_RDRT_dthetadtheta", &ellipse_RDRT_dthetadtheta, "ellipse_RDRT_dthetadtheta based on xtensor");
    m.def("getGradientEllipses", &getGradientEllipses, "getGradientEllipses based on xtensor");
    m.def("getGradientAndHessianEllipses", &getGradientAndHessianEllipses, "getGradientAndHessianEllipses based on xtensor");

}
