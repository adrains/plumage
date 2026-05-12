
// The Stannon: a three-label, second-order model implementation of the Cannon.


data {
    int<lower=1> S; // size of the training set

    real y[S]; // training set pseudo-continuum-normalised flux values
    real y_var[S]; // variance on the training set pseudo-continuum-normalised flux values 

    matrix[S, 10] DM; // design matrix
}


parameters {
    matrix[1, 10] theta; // the theta coefficients / spectral derivatives
    real<lower=0> s2; // intrinsic variance for each pixel.
}

transformed parameters {
    real total_y_err[S];
    for (s in 1:S) {
        total_y_err[s] = pow(s2 + y_var[s], 0.5);
    }
}


model {
    for (s in 1:S) {
        y[s] ~ normal(theta * DM[s]', total_y_err[s]);
    }
}