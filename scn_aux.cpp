#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "scn.h"
#include "kim.h"
#include "utils.h"

int read_cac(double *input, int len)
{
	FILE *myfile;
	char filename[50];
 	double buffer;
	int i;
	int j;
  	
	sprintf(filename, "cacs%0.3f.txt",Mdt);
	printf("%s\n",filename);
	myfile=fopen(filename, "r");

	if (myfile == NULL)
  	{
    		perror ("The following error occurred");
		return 1;
 	}

	printf("Reading in calcium time-series.\n");
	for(i = 0; i < len; i++)
	{
		fscanf(myfile, "%lf", &buffer);
		input[i] = buffer;
		if(!mod(i,1000))
			printf("%.15f\n", input[i]);
	}

	fclose(myfile);
	printf("Successfully read in calcium.\n");
	return 0;
}

int read_connect(char *filename, double *C)
{
	FILE *myfile;
 	double buffer;
	int i=0;
	int j=0;
  	
	char connectpath[50];
//	sprintf(connectpath,"./connectivity/%s",filename);
	sprintf(connectpath,"%s",filename);
	
	//Open connection to file
	myfile=fopen(connectpath, "r");

	if (myfile == NULL)
  	{
    		perror ("The following error occurred");
		return 1;
 	}

	//Read in data and store in matrix in column-major order
	printf("Reading in connectivity.\n");

	while (fscanf(myfile, "%lf", &buffer) != EOF) {
		C[j+i*ncells] = buffer;
		j++;
		if (j==ncells) {
			j=0;
			i++;
		}
	}
	
	//Check that amount of data in file matches expected size of matrix
	if (i != ncells || j != 0) {
		printf("Error: file did not contain a %dx%d matrix\n", ncells, ncells);
		return 1;
	}
	
	//Print out matrix
//	print_matrix(M, ncells, ncells);
/*	for (j=0; j<ncells*ncells;j++){
		printf("%lf\t", C[j]);
	}
	printf("\n");
*/
/*	j = 0;
	for (i=0; i<ncells; i++) {
		printf("%lf\t", C[j+i*ncells]);
	}
*/
	//Close connection to file and return to main
	fclose(myfile);
	printf("Successfully read in matrix.\n");
	return 0;
}

int make_rconnect(double *C, double pctconnect)
{
	double rnum;
  	
	//Read in data and store in matrix in column-major order
	printf("Creating random connectivity matrix:\n");
	printf("\tPct connect = %lf", pctconnect);

	for (int i = 0; i < ncells; i++) {
		for( int j = 0; j < ncells; j++ ) {
			rnum  = randu(0,1);
//			printf("%lf\t",rnum);
			if( rnum < pctconnect )
				C[j+i*ncells] = 1;
		}
	}
	
	//Print out matrix
/*	for (int i = 0; i < ncells; i++) {
		for( int j = 0; j < ncells; j ) {
			printf("%lf\t", C[j+i*ncells]);
		}
		printf("\n");
	}
*/
	//Close connection to file and return to main
	printf("Successfully created matrix.\n");
	return 0;
}


int write_connect(double *C, const char *name)
{
	//write the column-major matrix C to file
	FILE *file;
	int i, j;
	char filename[50];
  	sprintf(filename,"./connectivity/generated/%s",name);

	file=fopen(filename, "w");

	if (file == NULL)
  	{
    		perror ("The following error occurred");
		return 1;
 	}

	for (int i = 0; i < ncells; i++) {
		for( int j = 0; j < ncells; j++ ) {
			fprintf(file,"%lf\t", C[i+j*ncells]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
	return 0;
}

/*
int define_grid(double *grid)
{
	FILE *myfile;
 	double buffer;
	int i;
	int j;
  	
	myfile=fopen("grid.txt", "r");

	if (myfile == NULL)
  	{
    		perror ("The following error occurred");
		return 1;
 	}

	printf("Reading in grid.\n");
//	fscanf(myfile, "%d\t%d", nrow,ncols);
	for(i = 0; i < dim1; i++)
	{
		for (j = 0 ; j < dim2; j++)
		{
			fscanf(myfile, "%d", &buffer);
			grid[j+i*dim2] = buffer;
			//printf("%.15f ", input[j+i*WIDTH]);
		}
		//printf("\n");
	}

	fclose(myfile);
	printf("Successfully read in %dx%d grid.\n",dim1,dim2);
	return 0;
}
*/

