/* Host functions needed for initializing state and parameter structs for the Diekman-Forger model */

/* Includes functions:
    FILL_STATE: sets all elements of state variable arrays contained in struct (s) to values from array (value)
		of length (length)
    READ_PARAMS: reads in WIDTH number of parameters from file "parameters.txt" in local folder
		 and saves them in array (input)
    WRITE_PARAMS: writes WIDTH number of parameters from array (output) to file "parameters.txt" in local folder
		  each row in this file contains parameters for one cell, and each column is a different parameter
    FILL_PARAMS: sets all elements of parameter arrays contained in struct (p) to values from array (value)
		 of length (length). if NEW = 1, generates a new parameter set and writes it to a file. if NEW = 0,
		 reads in the parameters from file.
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "scn.h"
#include "df.h"
#include "kim.h"
#include "parameters.h"
#include "utils.h"
#include <algorithm>

void vec2Estate(double *vec, Estate *E, int n)
{
	E->V[n] = vec[0];
	E->m[n] = vec[1];
	E->h[n] = vec[2];
	E->n[n] = vec[3];
	E->rl[n] = vec[4];
	E->rnl[n] = vec[5];
	E->fnl[n] = vec[6];
	E->s[n] = vec[7];
	E->cas[n] = vec[8];
	E->cac[n] = vec[9];
	E->out[n] = vec[10];
	E->gaba[n] = vec[11];
	E->y[n] = vec[12];
}

void Estate2vec(Estate *E, int n, double *vec)
{
	vec[0] = E->V[n];
	vec[1] = E->m[n];
	vec[2] = E->h[n];
	vec[3] = E->n[n];
	vec[4] = E->rl[n];
	vec[5] = E->rnl[n];
	vec[6] = E->fnl[n];
	vec[7] = E->s[n];
	vec[8] = E->cas[n];
	vec[9] = E->cac[n];
	vec[10] = E->out[n];
	vec[11] = E->gaba[n];
	vec[12] = E->y[n];
}

void vec2Eparams(double *vec, Eparameters *p, int n)
{
	p->c[n] = vec[0];
	p->gna[n] = vec[1];
	p->gk[n] = vec[2];
	p->gcal[n] = vec[3];
	p->gcanl[n] = vec[4];
	p->gkca[n] = vec[5];
	p->gkleak[n] = vec[6];
	p->gnaleak[n] = vec[7];
	p->Ena[n] = vec[8];
	p->Ek[n] = vec[9];
	p->Eca[n] = vec[10];
	p->lambda[n] = vec[11];
	p->clk[n] = vec[12];
	p->Egaba[n] = vec[13];
}

void Eparams2vec(Eparameters *p, int n, double *vec)
{
	vec[0] = p->c[n];
	vec[1] = p->gna[n];
	vec[2] = p->gk[n];
	vec[3] = p->gcal[n];
	vec[4] = p->gcanl[n];
	vec[5] = p->gkca[n];
	vec[6] = p->gkleak[n];
	vec[7] = p->gnaleak[n];
	vec[8] = p->Ena[n];
	vec[9] = p->Ek[n];
	vec[10] = p->Eca[n];
	vec[11] = p->lambda[n];
	vec[12] = p->clk[n];
	vec[13] = p->Egaba[n];
}

int Einitialize_repeat(Estate *E, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Envars], init[Envars];

        printf("Reading in E initial conditions to repeat from file %s\n", name);
	infile = open_file(name, "r");
	while ( (fscanf(infile, "%lf,", &buffer) != EOF) && (count < Envars) )
	{
		input[count] = buffer;
//		printf("%lf\t",input[count]);
		count++;
	}

	if (count < Envars)
	{
		printf("\nWARNING: default E initialization file contained only %d values but there are %d states. Remaining states have been initialized to zero.\n",count,Envars);
		for (int i=count+1; i<Envars ;i++)
			input[i]=0;
	}

	for(int n = 0; n < ncells; n++)
	{

		for(int i=0; i < Envars; i++)
		{
			if(ERANDOMINIT) // perturb initial conditions
			{
				if(input[i] >= 0)
					init[i] = fmax(randn(input[i], input[i]*EISD),0.0);
				else
					init[i] = randn(input[i], fabs(input[i]*EISD));
			}
			else
				init[i] = input[i];
		}

		vec2Estate(init,E,n);
	}
	
	printf("\nSuccessfully set %d E initial conditions.\n", count);
			
	fclose(infile);

	return 0;
}

int Einitialize(Estate *E, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Envars];

        printf("Reading in all E initial conditions from file %s\n", name);
	infile = open_file(name, "r");
	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Envars; i++)
		{
			if(fscanf(infile, "%lf,", &buffer) != EOF)
				count++;
			input[i] = buffer;
		}

/*		if(ERANDOMINIT) // perturb initial conditions
		{
			for(int i=0; i < Envars; i++)
				input[i] = randn(input[i], fabs(input[i]*EISD));
		}
*/
		vec2Estate(input,E,n);
//		printf("%d\t%lf\n",n,E->V[n]);
	}
	
	if (count != Envars*ncells)
	{
		printf("\nEinit file did not contain the correct number of records (%d variables, %d cells)\n",Envars,ncells);
		return 1;
	}
	else 
		printf("\nSuccessfully set E initial conditions.\n");

	fclose(infile);

	return 0;
}

int Epinitialize_repeat(Eparameters *p, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Enparams], init[Enparams];

        printf("Reading in E parameters to repeat from file %s\n", name);
	infile = open_file(name, "r");
	while ( (fscanf(infile, "%lf,", &buffer) != EOF) && (count < Enparams) )
	{
		input[count] = buffer;
//		printf("%lf\t",input[count]);
		count++;
	}

	if (count < Enparams)
	{
		printf("\nWARNING: E parameter file contained only %d values but there are %d parameters. Remaining parameters have been set to zero.\n",count,Enparams);
		for (int i=count+1; i<Enparams ;i++)
			input[i]=0;
	}

	for(int n = 0; n < ncells; n++)
	{

		for(int i=0; i < Enparams; i++)
		{
			if( ERANDOMPARAMS & ( (i==10) | (i==12) ) ) // perturb Eca and clk
				init[i] = randn(input[i], input[i]*EPSD);
			else if( (i==13) & PERCENTEXCITE > 0)
			{
				double tmp = ((double)n)/ncells;
				if(EXCITEDIST == 1)
				{
					if( tmp*100.0 < (PERCENTEXCITE) )
						input[i] = -32;
					else
						input[i] = -80;
				}
				else if(EXCITEDIST == 2)
				{
					if( tmp*100.0 > (100.0-PERCENTEXCITE) )
						input[i] = -32;
					else
						input[i] = -80;
				}
				else
				{
					double rnum  = randu(0,1);
					if( rnum < PERCENTEXCITE/100.0 )
						input[i] = -32;
					else
						input[i] = -80;
				}	
			}
			else
				init[i] = input[i];
//			printf("%lf\t",init[0]);
		}

		vec2Eparams(init,p,n);
	}
	
	printf("\nSuccessfully set %d E parameters.\n", count);
			
	fclose(infile);

	return 0;
}

int Epinitialize(Eparameters *p, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Enparams];

        printf("Reading in all E parameters from file %s\n", name);
	infile = open_file(name, "r");

	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Enparams; i++)
		{
			if(fscanf(infile, "%lf,", &buffer) != EOF)
				count++;
//			if(i == 1) // gna KO (TTX)
			if(i == -1) // no KO
				input[i] = 0;
			else if( ERANDOMPARAMS & ( (i==10) | (i==12) ) ) // perturb Eca and clk
				input[i] = randn(input[i], input[i]*EPSD);
			else if( (i==13) & PERCENTEXCITE > 0)
			{
				double tmp = ((double)n)/ncells;
				if(EXCITEDIST == 1)
				{
					if( tmp*100.0 < (PERCENTEXCITE) )
						input[i] = -32;
					else
						input[i] = -80;
				}
				else if(EXCITEDIST == 2)
				{
					if( tmp*100.0 > (100.0-PERCENTEXCITE) )
						input[i] = -32;
					else
						input[i] = -80;
				}
				else
				{
					double rnum  = randu(0,1);
					if( rnum < PERCENTEXCITE/100.0 )
						input[i] = -32;
					else
						input[i] = -80;
				}	
			}
			else
				input[i] = buffer;
		}

		vec2Eparams(input,p,n);

//		printf("%d\t%lf\n",n,E->Eca[n]);
	}
	
	if (count != Enparams*ncells)
	{
		printf("\nEparameters file did not contain the correct number of records (%d variables, %d cells)\n",Enparams,ncells);
		return 1;
	}
	else 
		printf("\nSuccessfully set E parameters.\n");

	fclose(infile);

	return 0;
}

int write_Efinal(Estate *E, FILE *outfile)
{
	double *output;
	output = (double*)malloc(Envars*sizeof(double));

	for(int n = 0; n<ncells; n++)
	{
		Estate2vec(E, n, output);

		for(int i = 0; i<Envars; i++)
		{
			fprintf(outfile, "%.12lf,",output[i]);
		}
		fprintf(outfile, "\n");
	}

	return 0;
}

int write_Eparams(Eparameters *p, const char *name)
{
	FILE *pfile;
	int i, j;
	char pfilename[50];
  	sprintf(pfilename,"./Eparameters/%s",name);

	pfile=fopen(pfilename, "w");

	if (pfile == NULL)
  	{
    		perror ("The following error occurred");
		return 1;
 	}

	double output[Enparams];

	for (int n=0; n < ncells; n++)
	{
		/* write the parameters from the nth cell to a vector */
		Eparams2vec(p,n,output);

		/* write the vector of parameters to the file */
		for(i = 0; i < Enparams; i++)
		{
			fprintf(pfile, "%lf ", output[i]);
		}
		fprintf(pfile,"\n");

	}
	fclose(pfile);
	return 0;
}

int write_Eresult(double *output, FILE *outfile, int record, int Mt_step, int summary)
{
	printf("Writing ephys record %d, mt_step = %d, summary = %d\n",record,Mt_step, summary);
	double curr_t = Mt_step*Mdt;
	int maxt = Enstep/Erecord;
	int Eres_len=ncells*(maxt+1);

	if (summary)
	{
		int i = maxt;
		fprintf(outfile, "%.12lf", curr_t+i*Edt*Erecord/3600000.0);
		for (int j=0; j<ncells; j++) {
			fprintf(outfile, "\t%.12lf", output[record*Eres_len+j+i*ncells]);
		}
		fprintf(outfile, "\n");
	}
	else
	{
		for (int i=0; i<(maxt+1); i++) {
			fprintf(outfile, "%.12lf", curr_t+i*Edt*Erecord/3600000.0);
			for (int j=0; j<ncells; j++) {
				fprintf(outfile, "\t%.12lf", output[record*Eres_len+j+i*ncells]);
			}
			fprintf(outfile, "\n");
		}
	}
	return 0;
}

