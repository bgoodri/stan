#ifndef STAN__MATH__MATRIX__REAL_SCHUR_HPP
#define STAN__MATH__MATRIX__REAL_SCHUR_HPP

#include <stan/math/prim/scal/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {
  namespace math {

    template <typename T>
    std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
    real_schur(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      stan::math::check_nonzero_size("eigenvectors_sym", "m", m);
      stan::math::check_square("pseudoeigensystem", "m", m);

      Eigen::RealSchur<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > solver(m);
      std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > out(2);
      out[0] = solver.matrixU();
      out[1] = solver.matrixT();
      return out;
    }

  }
}
#endif
