#ifndef KIM_KERNELS_H_
#define KIM_KERNELS_H_

// C wrapper functions to call cuda kernels
void rhs_wrapper(double t, Mstate *s, Mstate *s_dot, Mparameters *p, Estate *x, double *input, int *upstream, int LIGHT);
void lincomb_wrapper(double c, Mstate *s, Mstate *s1, Mstate *s2);
void rk4_wrapper(Mstate *s, Mstate *k1, Mstate *k2, Mstate *k3, Mstate *k4, double t);
void record_result_wrapper(int i, Mresult *r, Mstate *s);
	
#endif
