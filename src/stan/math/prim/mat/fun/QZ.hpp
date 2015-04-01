#ifndef STAN__MATH__MATRIX__QZ_HPP
#define STAN__MATH__MATRIX__QZ_HPP

#include <stan/math/prim/scal/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/promote_common.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <vector>

namespace stan {

  namespace math {

    template <typename T_A, typename T_B>
    std::vector<Eigen::Matrix<typename stan::return_type<T_A, T_B>::type,
                Eigen::Dynamic,Eigen::Dynamic> >
    QZ(const Eigen::Matrix<T_A,Eigen::Dynamic,Eigen::Dynamic>& A,
       const Eigen::Matrix<T_B,Eigen::Dynamic,Eigen::Dynamic>& B) {
      stan::math::check_nonzero_size("QZ", "A", A);
      stan::math::check_nonzero_size("QZ", "B", B);
      stan::math::check_square("QZ", "A", A);
      stan::math::check_square("QZ", "B", B);
      stan::math::check_matching_dims("QZ", "A", A, "B", B);
      typedef typename stan::return_type<T_A,T_B>::type T_return_type;

      Eigen::RealQZ<Eigen::Matrix<T_return_type,Eigen::Dynamic,Eigen::Dynamic> >
        solver(promote_common<Eigen::Matrix<T_A,Eigen::Dynamic,Eigen::Dynamic>,
                              Eigen::Matrix<T_B,Eigen::Dynamic,Eigen::Dynamic> >(A),
               promote_common<Eigen::Matrix<T_A,Eigen::Dynamic,Eigen::Dynamic>,
                              Eigen::Matrix<T_B,Eigen::Dynamic,Eigen::Dynamic> >(B));
      std::vector<Eigen::Matrix<T_return_type,Eigen::Dynamic,Eigen::Dynamic> >
        out(4);
      out[0] = solver.matrixS();
      out[1] = solver.matrixT();
      out[2] = solver.matrixQ();
      out[3] = solver.matrixZ();
      return out;
    }

  }
}

