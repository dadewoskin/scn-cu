#ifndef DF_AUX_H_
#define DF_AUX_H_

// C++ auxilliary functions for Diekman-Forger model
void vec2Estate(ephys_t *vec, Estate *E, int n);
void Estate2vec(Estate *E, int n, ephys_t *vec);
void vec2Eparams(ephys_t *vec, Eparameters *p, int n);
void Eparams2vec(Eparameters *p, int n, ephys_t *vec);

int Einitialize_repeat(Estate *E, const char *name);
int Einitialize(Estate *E, const char *name);
int Epinitialize_repeat(Eparameters *p, const char *name);
int Epinitialize(Eparameters *p, const char *name);

int write_Eresult(ephys_t *output, std::ofstream& outfile, int record, int Mt_step, int summary);
int write_Efinal(Estate *E, std::ofstream& outfile);
int write_Eparams(Eparameters *p, const char *name);

#endif
