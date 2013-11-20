#ifndef DF_KERNELS_H_
#define DF_KERNELS_H_

// C wrapper functions to call cuda kernels
void leapfrog_wrapper(Estate *xi, Estate *xf, Eparameters *p, Mstate *M, double *input, int *upstream, double t);
void leapfrog_copy_wrapper(int i, int res_len, double *result, Estate *xi, Estate *xf, double *input, int *upstream);

#endif
