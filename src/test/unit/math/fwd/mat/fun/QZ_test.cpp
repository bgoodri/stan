#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/util.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev.hpp>
#include <stan/math/fwd.hpp>
#include <stan/math/prim/mat/fun/QZ.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixQZ, excepts_fd) {
  stan::agrad::matrix_fd m0;
  stan::agrad::matrix_fd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_fd ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::QZ;
  EXPECT_THROW(QZ(m0,m0),std::invalid_argument);
  EXPECT_NO_THROW(QZ(ev_m1, ev_m1));
  EXPECT_THROW(QZ(m0,m1),std::invalid_argument);
}
TEST(AgradFwdMatrixQZ, excepts_ffd) {
  stan::agrad::matrix_ffd m0;
  stan::agrad::matrix_ffd m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_ffd ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::QZ;
  EXPECT_THROW(QZ(m0,m0),std::invalid_argument);
  EXPECT_NO_THROW(QZ(ev_m1, ev_m1));
  EXPECT_THROW(QZ(m0,m1),std::invalid_argument);
}
TEST(AgradFwdMatrixQZ, excepts_fv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_fv ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::QZ;
  EXPECT_THROW(QZ(m0,m0),std::invalid_argument);
  EXPECT_NO_THROW(QZ(ev_m1, ev_m1));
  EXPECT_THROW(QZ(m0,m1),std::invalid_argument);
}
TEST(AgradFwdMatrixQZ, excepts_ffv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::agrad::matrix_ffv ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::QZ;
  EXPECT_THROW(QZ(m0,m0),std::invalid_argument);
  EXPECT_NO_THROW(QZ(ev_m1, ev_m1));
  EXPECT_THROW(QZ(m0,m1),std::invalid_argument);
}

TEST(AgradFwdMatrixQZ, matrix_fd) {
  stan::agrad::matrix_fd m0;
  stan::agrad::matrix_fd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_fd res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_);
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_);
  EXPECT_FLOAT_EQ(0, res0(0,0).d_);
  EXPECT_FLOAT_EQ(0, res0(0,1).d_);
  EXPECT_FLOAT_EQ(0, res0(1,0).d_);
  EXPECT_FLOAT_EQ(0, res0(1,1).d_);
}

TEST(AgradFwdMatrixQZ, matrix_ffd) {
  stan::agrad::matrix_ffd m0;
  stan::agrad::matrix_ffd m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_ffd res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_.val_);
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_);
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_);
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_);
}

TEST(AgradFwdMatrixQZ, matrix_fv_1st_deriv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_fv res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_.val());
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val());


  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res0(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.17677669/2,h[0]);
  EXPECT_FLOAT_EQ(0.17677669/2,h[1]);
  EXPECT_FLOAT_EQ(-0.17677669/2,h[2]);
  EXPECT_FLOAT_EQ(-0.17677669/2,h[2]);
}
TEST(AgradFwdMatrixQZ, matrix_fv_2nd_deriv) {
  stan::agrad::matrix_fv m0;
  stan::agrad::matrix_fv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_fv res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_.val());
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val());


  AVEC z = createAVEC(m1(0,0).val_,m1(0,1).val_,m1(1,0).val_,m1(1,1).val_);
  VEC h;
  res0(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.088388346/2,h[0]);
  EXPECT_FLOAT_EQ(-0.088388346/2,h[1]);
  EXPECT_FLOAT_EQ(0.088388346/2,h[2]);
  EXPECT_FLOAT_EQ(0.088388346/2,h[3]);
}

TEST(AgradFwdMatrixQZ, matrix_ffv_1st_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_ffv res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_.val());


  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(0,0).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(0.17677669/2,h[0]);
  EXPECT_FLOAT_EQ(0.17677669/2,h[1]);
  EXPECT_FLOAT_EQ(-0.17677669/2,h[2]);
  EXPECT_FLOAT_EQ(-0.17677669/2,h[3]);
}
TEST(AgradFwdMatrixQZ, matrix_ffv_2nd_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_ffv res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_.val());


  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(0,0).d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.088388346/2,h[0]);
  EXPECT_FLOAT_EQ(-0.088388346/2,h[1]);
  EXPECT_FLOAT_EQ(0.088388346/2,h[2]);
  EXPECT_FLOAT_EQ(0.088388346/2,h[3]);
}

TEST(AgradFwdMatrixQZ, matrix_ffv_3rd_deriv) {
  stan::agrad::matrix_ffv m0;
  stan::agrad::matrix_ffv m1(2,2);
  m1 << 1, 2, 2,1;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;
  stan::math::matrix_d m2(2,2);
  m2 << 1, 0, 0, 1;

  stan::agrad::matrix_ffv res0 = stan::math::QZ(m1,m2)[2];

  EXPECT_FLOAT_EQ(0.70710677, res0(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(-0.70710677, res0(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(0.70710677, res0(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res0(1,1).d_.val_.val());


  AVEC z = createAVEC(m1(0,0).val_.val_,m1(0,1).val_.val_,
                      m1(1,0).val_.val_,m1(1,1).val_.val_);
  VEC h;
  res0(0,0).d_.d_.grad(z,h);
  EXPECT_NEAR(0.088388346/2,h[0],1e-8);
  EXPECT_NEAR(0.088388346/2,h[1],1e-8);
  EXPECT_NEAR(-0.088388346/2,h[2],1e-8);
  EXPECT_NEAR(-0.088388346/2,h[3],1e-8);
}

