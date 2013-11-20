#ifndef SCN_AUX_H_
#define SCN_AUX_H_

// C++ auxilliary functions for scn model
int read_cac(double *input, int len);
int read_connect(char *filename, double *C);
int make_rconnect(double *C, double pctconnect);
int write_connect(double *C, const char *name);
int define_grid(int *grid);

#endif
