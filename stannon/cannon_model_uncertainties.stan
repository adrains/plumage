
// The Stannon: a three-label, second-order model of The Cannon using many pixels

data {
    int<lower=1> S; // size of the training set
    int<lower=1> P; // number of pixels
    int<lower=1> L; // number of labels
    int<lower=1> C; // number of model coefficients per px

    matrix[S, P] y; // training set pseudo-continuum-normalised flux values
    matrix[S, P] y_var; // variance on the training set pseudo-continuum-normalized flux values

    matrix[S, L] label_means; // *whitened* mean label values (e.g., teff estimate after rescaling)
    matrix[S, L] label_variances; // *whitened* label variances

    matrix[S, C] design_matrix; // Design matrix that we're passing in
}

parameters {
    matrix[P, C] theta; // spectral derivatives
    real<lower=0> s2[P]; // intrinsic variance at each pixel

    matrix[S, L] true_labels; // true values of the labels
}

transformed parameters {

    matrix[S, P] total_y_err;

    /*
    // Build the design matrix
    matrix[S, 10] design_matrix;

    for (s in 1:S) {
        design_matrix[s, 1] = 1.0;
        design_matrix[s, 2] = true_labels[s, 1];
        design_matrix[s, 3] = true_labels[s, 2];
        design_matrix[s, 4] = true_labels[s, 3];
        design_matrix[s, 5] = pow(true_labels[s, 1], 2); // teff^2
        design_matrix[s, 6] = true_labels[s, 2] * true_labels[s, 1]; // logg * teff
        design_matrix[s, 7] = true_labels[s, 3] * true_labels[s, 1]; // feh * teff
        design_matrix[s, 8] = pow(true_labels[s, 2], 2); // logg^2
        design_matrix[s, 9] = true_labels[s, 2] * true_labels[s, 3]; // logg * feh
        design_matrix[s, 10] = pow(true_labels[s, 3], 2); // feh^2
    } */

    for (s in 1:S) {
        for (p in 1:P) {
            total_y_err[s, p] = pow(s2[p] + y_var[s, p], 0.5);
        }
    }
}


model {

    // Priors on the labels
    for (s in 1:S) {
        for (l in 1:L) {
            label_means[s, l] ~ normal(true_labels[s], pow(label_variances[s, l], 0.5));
        }
    }


    for (s in 1:S) {
        for (p in 1:P) {
            y[s, p] ~ normal(theta[p] * design_matrix[s]', total_y_err[s, p]);
        }
    }
}