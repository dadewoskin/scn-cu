#ifndef SCN_AUX_H_
#define SCN_AUX_H_

// C++ auxilliary functions for scn model
int write_array(ephys_t *output, int len, FILE *outfile);
int read_cac(ephys_t *input, int len);
int read_connect(char *filename, ephys_t *C);
int make_rconnect(ephys_t *C, double pctconnect);
int write_connect(ephys_t *C, const char *name);
int define_grid(int *grid);

#endif
