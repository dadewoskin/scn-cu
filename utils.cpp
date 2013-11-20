/* Contains utilities: */
/* OPEN_FILE: opens a file
	INPUT: path to file
	RETURNS: file handle
  MOD: a modular arithmetic function
	INPUT: two integers, x and y
	RETURNS: x mod y

 RANDN: a random number generator (normally distributed)
	INPUT: mean mu and standard deviation sigma (defaults set at 0 and 1 respectively
	RETURNS: a random number (double precision)
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

FILE* open_file(const char *path, const char *mode) {
	FILE *handle;
	if ((handle = fopen(path, mode)) == NULL) {
		printf("can not open file %s", path);
	}
	else
		return handle;
}

int mod(int x, int y){
	return ((x % y) + y) % y;
}

#define PI 3.14159265358979323846

//random number generator from http://www.dreamincode.net/code/snippet1446.htm
double randn(double mu=0.0, double sigma=1.0) {
	static bool deviateAvailable=false;	//        flag
	static float storedDeviate;	//        deviate from previous calculation
	double dist, angle;

	//        If no deviate has been stored, the standard Box-Muller transformation is
	//        performed, producing two independent normally-distributed random
	//        deviates.  One is stored for the next round, and one is returned.
	if (!deviateAvailable) {

		//        choose a pair of uniformly distributed deviates, one for the
		//        distance and one for the angle, and perform transformations
		dist=sqrt( -2.0 * log(double(rand()) / double(RAND_MAX)) );
		angle=2.0 * PI * (double(rand()) / double(RAND_MAX));

		//        calculate and store first deviate and set flag
		storedDeviate=dist*cos(angle);
		deviateAvailable=true;

		//        calcaulate return second deviate
		return dist * sin(angle) * sigma + mu;
	}

	//        If a deviate is available from a previous call to this function, it is
	//        returned, and the flag is set to false.
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}

double randu(double min, double max) {
/* return a random number between 0 and limit inclusive.
 *  */
//	double retval = ((double) rand() / (RAND_MAX+1)) * (max-min+1) + min;
	double retval = ((double) rand() / (RAND_MAX));

	return retval;
}
