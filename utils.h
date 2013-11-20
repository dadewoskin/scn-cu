#ifndef _UTILS_H_
#define _UTILS_H_

// C++ utilities
FILE* open_file(const char *path, const char *mode);
int mod(int x, int y);
double randn(double mu=0.0, double sigma=1.0);
double randu(double min, double max);

#endif
