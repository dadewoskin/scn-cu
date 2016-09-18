#ifndef KIM_AUX_H_
#define KIM_AUX_H_

// C++ auxilliary functions for kim model
void vec2Mstate(mclock_t *vec, Mstate *M, int n);
void Mstate2vec(Mstate *M, int n, mclock_t *vec);
void vec2Mparams(mclock_t *vec, Mparameters *p, int n);
void Mparams2vec(Mparameters *p, int n, mclock_t *vec);
void vec2Mresult(mclock_t *vec, Mresult *r, int n);
void Mresult2vec(Mresult *r, int n, mclock_t *vec);

int Minitialize_repeat(Mstate *M, const char *name);
int Minitialize(Mstate *M, const char *name);
int Mcheck_init(Mstate *M, Mparameters *p);
int Mpinitialize_repeat(Mparameters *p, const char *name);
int Mpinitialize(Mparameters *p, const char *name);

int write_Mresult(mclock_t *output, std::ofstream& outfile);
int write_Mfinal(Mstate *M, std::ofstream& outfile);
int write_Mparams(Mparameters *p, const char *name);

#endif
