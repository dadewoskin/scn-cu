#ifndef KIM_AUX_H_
#define KIM_AUX_H_

// C++ auxilliary functions for kim model
void vec2Mstate(double *vec, Mstate *M, int n);
void Mstate2vec(Mstate *M, int n, double *vec);
void vec2Mparams(double *vec, Mparameters *p, int n);
void Mparams2vec(Mparameters *p, int n, double *vec);
void vec2Mresult(double *vec, Mresult *r, int n);
void Mresult2vec(Mresult *r, int n, double *vec);

int Minitialize_repeat(Mstate *M, const char *name);
int Minitialize(Mstate *M, const char *name);
int Mpinitialize_repeat(Mparameters *p, const char *name);
int Mpinitialize(Mparameters *p, const char *name);

int write_Mresult(double *output, FILE *outfile);
int write_Mfinal(Mstate *M, FILE *outfile);
int write_Mparams(Mparameters *p, const char *name);

#endif
