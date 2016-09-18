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
#include <iostream>
#include <limits>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "scn.h"
#include "df.h"
#include "kim.h"
#include "parameters.h"
#include "utils.h"
#include <algorithm>

void vec2Estate(ephys_t *vec, Estate *E, int n)
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

void Estate2vec(Estate *E, int n, ephys_t *vec)
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

void vec2Eparams(ephys_t *vec, Eparameters *p, int n)
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

void Eparams2vec(Eparameters *p, int n, ephys_t *vec)
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
	std::ifstream ifs(name);
	int count = 0;
	ephys_t buffer, input[Envars], init[Envars];

        std::cout << "Reading in E initial conditions to repeat from file " << name << "\n";
	while ( (ifs >> buffer) && (count < Envars) )
	{
		input[count] = buffer;
		count++;
	}

	if (count < Envars)
	{
		std::cout << "\nWARNING: default E initialization file contained only " << count << " values but there are " << Envars << " states. Remaining states have been initialized to zero.\n";
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
	
	std::cout << "\nSuccessfully set " << count << " E initial conditions.\n";
			
	ifs.close();

	return 0;
}

int Einitialize(Estate *E, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	ephys_t buffer, input[Envars];

        std::cout << "Reading in all E initial conditions from file " << name << "%s\n";
	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Envars; i++)
		{
			if(ifs >> buffer)
				count++;
			input[i] = buffer;
		}

		if(ERANDOMINIT) // perturb initial conditions
		{
			for(int i=0; i < Envars; i++)
				input[i] = randn(input[i], fabs(input[i]*EISD));
		}

		vec2Estate(input,E,n);
	}
	
	if (count != Envars*ncells)
	{
		std::cout << "\nEinit file did not contain the correct number of records (" << Envars << " variables, " << ncells << " cells)\n";
		return 1;
	}
	else 
		std::cout << "\nSuccessfully set E initial conditions.\n";

	ifs.close();

	return 0;
}

int Epinitialize_repeat(Eparameters *p, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	ephys_t buffer, input[Enparams], init[Enparams];

	std::cout << "Reading in E parameters to repeat from file " << name << "\n";
	while ( (ifs >> buffer) && (count < Enparams) )
	{
		input[count] = buffer;
		count++;
	}

	if (count < Enparams)
	{
		std::cout << "\nWARNING: E parameter file contained only " << count << " values but there are " << Enparams << " parameters. Remaining parameters have been set to zero.\n";
		for (int i=count+1; i<Enparams ;i++)
			input[i]=0;
	}

	srand (time(NULL));
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
						init[i] = -32;
					else
						init[i] = -80;
				}
				else if(EXCITEDIST == 2)
				{
					if( tmp*100.0 > (100.0-PERCENTEXCITE) )
						init[i] = -32;
					else
						init[i] = -80;
				}
				else
				{
					double rnum  = randu(0,1);
					if( rnum < PERCENTEXCITE/100.0 )
						init[i] = -32;
					else
						init[i] = -80;
				}	
			}
			else
				init[i] = input[i];
		}

		vec2Eparams(init,p,n);
	}
	
	std::cout << "\nSuccessfully set " << count << " E parameters.\n";
			
	ifs.close();

	return 0;
}

int Epinitialize(Eparameters *p, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	ephys_t buffer, input[Enparams];

	std::cout << "Reading in all E parameters from file " << name << "\n";

	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Enparams; i++)
		{
			if(ifs >> buffer)
				count++;
			if( ERANDOMPARAMS & ( (i==10) | (i==12) ) ) // perturb Eca and clk
				input[i] = randn(input[i], input[i]*EPSD);
			else if( (i==13) & PERCENTEXCITE == 0)
				input[i] = -80;
			else if( (i==13) & PERCENTEXCITE > 0)
			{
				ephys_t tmp = ((ephys_t)n)/ncells;
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
	}
	
	if (count != Enparams*ncells)
	{
		std::cout << "\nEparameters file did not contain the correct number of records (" << Enparams << " variables, " << ncells << " cells)\n";
		return 1;
	}
	else 
		std::cout << "\nSuccessfully set E parameters.\n";

	ifs.close();

	return 0;
}

int write_Efinal(Estate *E, std::ofstream& outfile)
{
	ephys_t *output;
	output = (ephys_t*)malloc(Envars*sizeof(ephys_t));

	for(int n = 0; n<ncells; n++)
	{
		Estate2vec(E, n, output);

		for(int i = 0; i<Envars; i++)
		{
			outfile << output[i] << " ";
		}
		outfile << "\n";
	}

	return 0;
}

int write_Eparams(Eparameters *p, const char *name)
{
	int i, j;
	char pfilename[50];
  	sprintf(pfilename,"./Eparameters/%s",name);

	std::ofstream ofs(pfilename);

	if (ofs.fail())
		return 1;

	ephys_t output[Enparams];

	for (int n=0; n < ncells; n++)
	{
		/* write the parameters from the nth cell to a vector */
		Eparams2vec(p,n,output);

		/* write the vector of parameters to the file */
		for(i = 0; i < Enparams; i++)
		{
			ofs << output[i] << " ";
		}
		ofs << "\n";

	}
	ofs.close();
	return 0;
}

int write_Eresult(ephys_t *output, std::ofstream& outfile, int record, int Mt_step, int summary)
{
	std::cout << "Writing ephys record " << record << ", mt_step = " << Mt_step << ", summary = " << summary << "\n";
	double curr_t = Mt_step*Mdt;
	int maxt = Enstep/Erecord;
	int Eres_len=ncells*(maxt+1);

        outfile.precision(std::numeric_limits<double>::digits10 + 2);

	if (summary)
	{
		int i = maxt;
		outfile << curr_t+(double)i*Edt*Erecord/3600000.0;
		for (int j=0; j<ncells; j++) {
			outfile << " " << output[record*Eres_len+j+i*ncells];
		}
		outfile << "\n";
	}
	else
	{
		for (int i=0; i<(maxt+1); i++) {
			outfile << curr_t+(double)i*Edt*Erecord/3600000.0;
			for (int j=0; j<ncells; j++) {
				outfile << " " << output[record*Eres_len+j+i*ncells];
			}
			outfile << "\n";
		}
	}
	return 0;
}

