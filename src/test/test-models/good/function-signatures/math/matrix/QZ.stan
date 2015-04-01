data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix[2];

  transformed_data_matrix <- QZ(d_matrix,d_matrix);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix[2];

  transformed_param_matrix <- QZ(d_matrix,p_matrix);
  transformed_param_matrix <- QZ(p_matrix,d_matrix);
  transformed_param_matrix <- QZ(p_matrix,p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
