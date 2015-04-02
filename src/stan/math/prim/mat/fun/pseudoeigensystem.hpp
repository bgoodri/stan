#ifndef STAN__MATH__MATRIX__PSEUDOEIGENSYSTEM_HPP
#define STAN__MATH__MATRIX__PSEUDOEIGENSYSTEM_HPP

#include <stan/math/prim/scal/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/real_schur.hpp>
#include <stan/math/prim/scal/fun/abs.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <vector>
#include <iostream>
namespace stan {
  namespace math {

    // These functions are mostly copied from EigenSolver.h in Eigen
    // Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
    // Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

    template <typename T>
    inline
    Eigen::Matrix<T,2,1> cdiv(const T& xr, const T& xi,
                              const T& yr, const T& yi) {
      T r, d;
      Eigen::Matrix<T,2,1> out;
      if (abs(yr) > abs(yi)) {
        r = yi / yr;
        d = yr + r * yi;
        out.coeffRef(0) = (xr + r * xi) / d;
        out.coeffRef(1) = (xi - r * xr) / d;
      }
      else {
        r = yr / yi;
        d = yi + r * yr;
        out.coeffRef(0) = (r * xr + xi) / d;
        out.coeffRef(1) = (r * xi - xr) / d;
      }
      return out;
    }

    template <typename T>
    inline
    std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
    make_pseudoeigensystem(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> m_matT,
                           Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> m_eivec,
                           const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> m_eivalues,
                           int size) {
      typedef int Index;
      using stan::math::abs;
      using stan::math::sum;
      using stan::math::dot_product;
      using stan::math::square;
//      using stan::math::max;

      T norm(0);
      int mark(0);
      for (Index j = 0; j < size; ++j) {
        Eigen::Matrix<T,Eigen::Dynamic,1> seg = m_matT.row(j).segment(mark, size-mark).cwiseAbs();
        norm += sum(seg);
        mark = j;
      }

      // Backsubstitute to find vectors of upper triangular form
      if(norm > 0) for (Index n = size-1; n >= 0; n--) {
        T p = m_eivalues.coeff(n,0);
        T q = m_eivalues.coeff(n,1);

        // real vector
        if (q == T(0)) {
          T lastr(0), lastw(0);
          Index l = n;

          m_matT.coeffRef(n,n) = 1.0;
          for (Index i = n-1; i >= 0; i--) {
            T w = m_matT.coeff(i,i) - p;
            T r = dot_product(m_matT.row(i).segment(l,n-l+1).eval(),
                              m_matT.col(n).segment(l, n-l+1).eval());

            if (m_eivalues.coeff(i,1) < 0.0) {
              lastw = w;
              lastr = r;
            }
            else {
              l = i;
              if (m_eivalues.coeff(i,1) == 0.0) {
                if (w != 0.0)
                  m_matT.coeffRef(i,n) = -r / w;
                else
                  m_matT.coeffRef(i,n) = -r / (EPSILON * norm);
              }
              else { // Solve real equations
                T x = m_matT.coeff(i,i+1);
                T y = m_matT.coeff(i+1,i);
                T denom = square(m_eivalues.coeff(i,0) - p) +
                          square(m_eivalues.coeff(i,1));
                T t = (x * lastr - lastw * r) / denom;
                m_matT.coeffRef(i,n) = t;
                if (abs(x) > abs(lastw))
                  m_matT.coeffRef(i+1,n) = (-r - w * t) / x;
                else
                  m_matT.coeffRef(i+1,n) = (-lastr - y * t) / lastw;
              }

              // Overflow control
              T t = abs(m_matT.coeff(i,n));
              if ((EPSILON * t) * t > T(1))
                m_matT.col(n).tail(size-i) /= t;
            }
          }
        }
        else if (q < T(0) && n > 0) { // Complex vector
          T lastra(0), lastsa(0), lastw(0);
          Index l = n-1;
          Eigen::Matrix<T,2,1> cc;
          // Last vector component imaginary so matrix is triangular
          if (abs(m_matT.coeff(n,n-1)) > abs(m_matT.coeff(n-1,n))) {
            m_matT.coeffRef(n-1,n-1) = q / m_matT.coeff(n,n-1);
            m_matT.coeffRef(n-1,n) = -(m_matT.coeff(n,n) - p) / m_matT.coeff(n,n-1);
          }
          else {
            cc = cdiv<T>(0.0,-m_matT.coeff(n-1,n),m_matT.coeff(n-1,n-1)-p,q);
            m_matT.coeffRef(n-1,n-1) = cc(0,0);
            m_matT.coeffRef(n-1,n) = cc(1,0);
          }
          m_matT.coeffRef(n,n-1) = 0.0;
          m_matT.coeffRef(n,n) = 1.0;
          for (Index i = n-2; i >= 0; i--) {
            T ra = dot_product(m_matT.row(i).segment(l, n-l+1).eval(),
                               m_matT.col(n-1).segment(l, n-l+1).eval());
            T sa = dot_product(m_matT.row(i).segment(l, n-l+1).eval(),
                               m_matT.col(n).segment(l, n-l+1).eval());
            T w = m_matT.coeff(i,i) - p;

            if (m_eivalues.coeff(i,1) < 0.0) {
              lastw = w;
              lastra = ra;
              lastsa = sa;
            }
            else {
              l = i;
              if (m_eivalues.coeff(i,1) == T(0)) {
                cc = cdiv<T>(-ra,-sa,w,q);
                m_matT.coeffRef(i,n-1) = cc(0);
                m_matT.coeffRef(i,n) = cc(1);
              }
              else {
                // Solve complex equations
                T x = m_matT.coeff(i,i+1);
                T y = m_matT.coeff(i+1,i);
                T vr = square(m_eivalues.coeff(i,0) - p) +
                       square(m_eivalues.coeff(i,1)) - square(q);
                T vi = (m_eivalues.coeff(i,0) - p) * T(2) * q;
                if ((vr == 0.0) && (vi == 0.0))
                  vr = EPSILON * norm * (abs(w) + abs(q) + abs(x) + abs(y) + abs(lastw));

                cc = cdiv<T>(x*lastra-lastw*ra+q*sa,x*lastsa-lastw*sa-q*ra,vr,vi);
                m_matT.coeffRef(i,n-1) = cc(0);
                m_matT.coeffRef(i,n) = cc(1);
                if (abs(x) > (abs(lastw) + abs(q))) {
                  m_matT.coeffRef(i+1,n-1) = (-ra - w * m_matT.coeff(i,n-1) + q * m_matT.coeff(i,n)) / x;
                  m_matT.coeffRef(i+1,n) = (-sa - w * m_matT.coeff(i,n) - q * m_matT.coeff(i,n-1)) / x;
                }
                else {
                  cc = cdiv<T>(-lastra-y*m_matT.coeff(i,n-1),-lastsa-y*m_matT.coeff(i,n),lastw,q);
                  m_matT.coeffRef(i+1,n-1) = cc(0);
                  m_matT.coeffRef(i+1,n) = cc(1);
                }
              }

              // Overflow control FIXME: can't get max() to work
              T t = abs(m_matT.coeff(i,n-1));
              if(t < abs(m_matT.coeff(i,n))) t = abs(m_matT.coeff(i,n));
              if ((EPSILON * t) * t > T(1))
                m_matT.block(i, n-1, size-i, 2) /= t;

            }
          }

          // We handled a pair of complex conjugate eigenvalues, so need to skip them both
          n--;
        }
      }

      // Back transformation to get eigenvectors of original matrix
      for (Index j = size-1; j >= 0; j--)
        m_eivec.col(j) = m_eivec.leftCols(j+1) * m_matT.col(j).segment(0, j+1);

      // create pseudo eigenvalue matrix
      const int K = m_eivec.rows();
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> matD =
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Zero(K,K);
      for (int i = 0; i < K; i++) {
        if (m_eivalues.coeff(i,1) == 0.0)
          matD.coeffRef(i,i) = m_eivalues.coeff(i,0);
        else {
          matD.template block<2,2>(i,i) <<
            m_eivalues.coeff(i,0), m_eivalues.coeff(i,1),
           -m_eivalues.coeff(i,1), m_eivalues.coeff(i,0);
          ++i;
        }
      }

      // compute order of eigenvalues
      Eigen::VectorXi o(K);
      int zero_mark = 0;
      int k = 0;
      while (k < K) {
        if (m_eivalues(k,2) == 0) {
          o(k) = zero_mark;
          zero_mark++;
          k++;
          continue;
        }
        if (m_eivalues(k,2) == INFTY) {
          o(k) = k;
          k++;
          continue;
        }
        o(k) = 0;
        for (int j = 0; j < K; j++) o(k) += m_eivalues(j,2) < m_eivalues(k,2);
        if (k < (K - 1) && m_eivalues(k,2) == m_eivalues(k+1,2)) {
          o(k+1) = o(k) + 1;
          k += 2;
        }
        else if (k == (K - 1)) {
          o(k) = k;
          k++;
        }
        else k++;
      }

      // reorder by decreasing modulus
      Eigen::PermutationMatrix<Eigen::Dynamic> P(o);
      matD.applyOnTheLeft(P);
      matD.applyOnTheRight(P.transpose());
      m_eivec.applyOnTheRight(P.transpose());

      std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > out(2);
      out[0] = m_eivec;
      out[1] = matD;
      return out;
    }

    template <typename T>
    std::vector<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
    pseudoeigensystem(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      stan::math::check_nonzero_size("pseudoeigensystem", "m", m);
      stan::math::check_square("pseudoeigensystem", "m", m);
      const int K = m.rows();
      typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixT;
      std::vector<MatrixT> out = real_schur(m);
      MatrixT m_matT = out[1];

      MatrixT m_eivalues(K, 3);
      using stan::math::abs;
      using std::sqrt;
      // the 3 columns of m_eivalues are: real part, imaginary part, modulus
      int i = 0;
      while (i < K) {
        if (i == (K - 1) || m_matT.coeff(i+1, i) == 0) {
          m_eivalues.coeffRef(i,0) = m_matT.coeff(i, i);
          m_eivalues.coeffRef(i,1) = 0;
          m_eivalues.coeffRef(i,2) = abs(m_matT.coeff(i, i));
          i++;
        }
        else {
          T p = 0.5 * (m_matT.coeff(i, i) * m_matT.coeff(i+1, i+1));
          T z = sqrt(abs(square(p) + (m_matT.coeff(i+1, i) *
                                    m_matT.coeff(i, i+1))));
          m_eivalues.coeffRef(i,0) = m_matT.coeff(i, i) + p;
          m_eivalues.coeffRef(i,1) = z;
          m_eivalues.coeffRef(i,2) = sqrt(square(m_eivalues.coeffRef(i,0)) +
                                          square(m_eivalues.coeffRef(i,1)));
          m_eivalues.coeffRef(i+1,0) = m_matT.coeff(i, i) + p;
          m_eivalues.coeffRef(i+1,1) = -z;
          m_eivalues.coeffRef(i+1,2) = m_eivalues.coeffRef(i,2);
          i += 2;
        }
      }

      return make_pseudoeigenvectors(m_matT, out[2], m_eivalues, K);
    }

  }
}
#endif
