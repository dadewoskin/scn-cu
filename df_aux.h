#ifndef DF_AUX_H_
#define DF_AUX_H_

// C++ auxilliary functions for Diekman-Forger model
void vec2Estate(double *vec, Estate *E, int n);
void Estate2vec(Estate *E, int n, double *vec);
void vec2Eparams(double *vec, Eparameters *p, int n);
void Eparams2vec(Eparameters *p, int n, double *vec);

int Einitialize_repeat(Estate *E, const char *name);
int Einitialize(Estate *E, const char *name);
int Epinitialize_repeat(Eparameters *p, const char *name);
int Epinitialize(Eparameters *p, const char *name);

int write_Eresult(double *output, FILE *outfile, int record, int Mt_step, int summary);
//int write_Eresult(double *output, FILE *Voutfile, FILE *Coutfile, FILE *Ooutfile, int Mt_step);
int write_Efinal(Estate *E, FILE *outfile);
int write_Eparams(Eparameters *p, const char *name);

#endif
