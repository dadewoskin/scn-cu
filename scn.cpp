/* simulate ncells cells using the detailed model from Kim & Forger, MSB 2012 */
/* coupled with the electrophysiology model from Diekman et al. */
/* equations are solved in PARALLEL on GPUs */
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
//#include "/usr/local/cuda/include/cuda_runtime.h"

#include "utils.h"
#include "scn.h"
#include "scn_aux.h"
#include "kim.h"
#include "kim_aux.h"
#include "kim_kernels.h"
#include "df.h"
#include "df_aux.h"
#include "df_kernels.h"
#include "parameters.h"
#include "hstTimer.hh"

void usage(int argc, char *argv[])
{
	printf("Usage: %s output_filename\n",argv[0]);
	printf("Flags:");
	printf(" \t-MC Mconnect_filename\n");
	printf(" \t-Mi one_cell_Mstate_initial_condition_filename\n");
	printf(" \t-MI all_cells_Mstate_initial_condition_filename\n");
	printf(" \t-Mp one_cell_Mparameter_filename\n");
	printf(" \t-MP all_cells_Mparameter_filename\n");
	printf(" \t-EC Econnect_filename\n");
	printf(" \t-Ei one_cell_Estate_initial_condition_filename\n");
	printf(" \t-EI all_cells_Estate_initial_condition_filename\n");
	printf(" \t-Ep one_cell_Eparameter_filename\n");
	printf(" \t-EP all_cells_Eparameter_filename\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
	///////////////////////////////////////////////////////////////////////
	/* Read in output filename from cmd line */
	if (!(argc == 2 | argc == 4 | argc == 6 | argc == 8 | argc == 10 | argc == 12 | argc == 14))
		usage(argc, argv);

	char date[50];
	char out_filename[50];
	char Mconnect_filename[50];
	char Minit_filename[70];
	char Mparam_filename[70];
	char Econnect_filename[50];
	char Einit_filename[70];
	char Eparam_filename[70];

	time_t rawtime;
	struct tm * timeinfo;

	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	sprintf(date,"%d%02d%02d", 1900+timeinfo->tm_year, 1+timeinfo->tm_mon, timeinfo->tm_mday);

	int MREADCONN = 0;
	int MREADINIT = 0;
	int MREADPARAM = 0;
	int EREADCONN = 0;
	int EREADINIT = 0;
	int EREADPARAM = 0;
	mclock_t idx = 1000000;

	sprintf(out_filename,"%s_%s",date, argv[1]);

	for (int idx = 1; idx < argc;  idx++) {
		if (strcmp(argv[idx], "-MC") == 0) {
       			if (argc < idx + 2)
				usage(argc, argv);
 			printf("Flag -MC passed: M connectivity file = %s\n", argv[idx+1]);
		 	sprintf(Mconnect_filename, argv[idx+1]);
			MREADCONN = 1;
    		}
		else if (strcmp(argv[idx], "-Mi") == 0) {
       			if (argc < idx + 2)
				usage(argc, argv);
 			printf("Flag -Mi passed: initial conditions file = %s\n", argv[idx+1]);
		 	sprintf(Minit_filename, argv[idx+1]);
			MREADINIT = 1;
    		}
		else if (strcmp(argv[idx], "-MI") == 0) {
       			if (argc < idx + 2)
				usage(argc, argv);
 			printf("Flag -MI passed: initial conditions file = %s\n", argv[idx+1]);
		 	sprintf(Minit_filename, argv[idx+1]);
			MREADINIT = 2;
    		}
		else if (strcmp(argv[idx], "-Mp") == 0) {
			if (argc < idx + 2)
				usage(argc, argv);
       			printf("Flag -Mp passed: parameter file = %s\n", argv[idx+1]);
		 	sprintf(Mparam_filename, argv[idx+1]);
			MREADPARAM = 1;
    		}
		else if (strcmp(argv[idx], "-MP") == 0) {
			if (argc < idx + 2)
				usage(argc, argv);
       			printf("Flag -MP passed: parameter file = %s\n", argv[idx+1]);
		 	sprintf(Mparam_filename, argv[idx+1]);
			MREADPARAM = 2;
    		}
		else if (strcmp(argv[idx], "-EC") == 0) {
       			if (argc < idx + 2)
				usage(argc, argv);
 			printf("Flag -EC passed: E connectivity file = %s\n", argv[idx+1]);
		 	sprintf(Econnect_filename, argv[idx+1]);
			EREADCONN = 1;
    		}
		else if (strcmp(argv[idx], "-Ei") == 0) {
       			if (argc < idx + 2)
				usage(argc, argv);
 			printf("Flag -Ei passed: initial conditions file = %s\n", argv[idx+1]);
		 	sprintf(Einit_filename, argv[idx+1]);
			EREADINIT = 1;
    		}
		else if (strcmp(argv[idx], "-EI") == 0) {
       			if (argc < idx + 2)
				usage(argc, argv);
 			printf("Flag -EI passed: initial conditions file = %s\n", argv[idx+1]);
		 	sprintf(Einit_filename, argv[idx+1]);
			EREADINIT = 2;
    		}
		else if (strcmp(argv[idx], "-Ep") == 0) {
			if (argc < idx + 2)
				usage(argc, argv);
       			printf("Flag -Ep passed: parameter file = %s\n", argv[idx+1]);
		 	sprintf(Eparam_filename, argv[idx+1]);
			EREADPARAM = 1;
    		}
		else if (strcmp(argv[idx], "-EP") == 0) {
			if (argc < idx + 2)
				usage(argc, argv);
       			printf("Flag -EP passed: parameter file = %s\n", argv[idx+1]);
		 	sprintf(Eparam_filename, argv[idx+1]);
			EREADPARAM = 2;
    		}
/*    		else {
        		printf("Error: unknown flag\n");
        		exit(-1);
    		}*/
	}

	///////////////////////////////////////////////////////////////////////
	//// Declare variables, allocate memory on devices, and initialize ////
	///////////////////////////////////////////////////////////////////////
	Mstate *h_Mx, *Mx; // state variables for molec clock
	Mparameters *h_Mp, *Mp; // parameters for molec clock
	Mresult *h_Mr, *Mr; // time series used for fitting for molec clock

	Estate *h_Ex, *Exi, *Exf; // state variables for electrophys
	Eparameters *h_Ep, *Ep; // parameters for electrophys
	ephys_t *h_Eresult, *Eresult; // array for holding ephys results
	
	mclock_t *h_MC, *MC; // array holding M connectivity matrix in column-major format
	ephys_t *h_EC, *EC; // array holding E connectivity matrix in column-major format
	mclock_t *h_Minput, *Minput; // array holding input to each cell from upstream cells at each time point
	ephys_t *h_Einput, *Einput; // array holding input to each cell from upstream cells at each time point
	int *h_Mupstream, *Mupstream; // array holding number of cells upstream of each cell n for VIP connections
	int *h_Eupstream, *Eupstream; // array holding number of cells upstream of each cell n for ephys connections

	Mstate *k1, *k2, *k3, *k4, *kb; // test-steps for runge kutta 4 (RK4) method and a buffer kb for molec clock

	int* h_Vmax = (int*) malloc (sizeof(int));
	int* h_Vmin = (int*) malloc (sizeof(int));
	int Mres_len=ncells*((int)(Mnstep/Mrecord)+1);
	int Eres_len=ncells*((int)(Enstep/Erecord)+1);

	//Allocate space on host for output, state, parameters, and connectivity data
	h_Mx = (Mstate*)malloc(sizeof(Mstate));
	h_Mp = (Mparameters*)malloc(sizeof(Mparameters));
	h_Mr = (Mresult*)malloc(sizeof(Mresult));
	h_Ex = (Estate*)malloc(sizeof(Estate));
	h_Ep = (Eparameters*)malloc(sizeof(Eparameters));
	h_Eresult = (ephys_t*)malloc(ENRES*Eres_len*sizeof(ephys_t));

	h_MC = (mclock_t*)malloc(ncells*ncells*sizeof(mclock_t));
	h_EC = (ephys_t*)malloc(ncells*ncells*sizeof(ephys_t));
	h_Minput = (mclock_t*)malloc(ncells*sizeof(mclock_t));
	h_Einput = (mclock_t*)malloc(ncells*sizeof(ephys_t));
	h_Mupstream = (int*)malloc(ncells*sizeof(int));
	h_Eupstream = (int*)malloc(ncells*sizeof(int));

	for (int i=0;i<ENRES*Eres_len; i++){
		h_Eresult[i]=0;
	}

	CUDA_SAFE_CALL (cudaSetDevice(1));
	CUDA_SAFE_CALL (cudaMalloc((void **) &Eresult, ENRES*Eres_len*sizeof(ephys_t)));
	CUDA_SAFE_CALL (cudaMemcpy(Eresult, h_Eresult, ENRES*Eres_len*sizeof(ephys_t), cudaMemcpyHostToDevice));
	//Allocate Mresult place holder on device
	CUDA_SAFE_CALL (cudaMalloc((void **) &Mr, sizeof(Mresult)));

	////////////////////////////////////////////////////////////////////////////////
	//Read in initial values to host
	if(MREADINIT==1) { //read M initial conditions for 1 cell from filename passed to main
		if(Minitialize_repeat(h_Mx, Minit_filename))
			exit(EXIT_FAILURE);
	}
	else if(MREADINIT==2) { //read M initial conditions for all cells from filename passed to main
		if(Minitialize(h_Mx, Minit_filename))
			exit(EXIT_FAILURE);
	}
	else {	//use default M initial conditions
		sprintf(Minit_filename, "./Minit/Mf20140724_wt_r2g2e80_2_ctd2.txt");
		if(Minitialize_repeat(h_Mx, Minit_filename))
			exit(EXIT_FAILURE);
	}

	if(EREADINIT==1) { //read E initial conditions for 1 cell from filename passed to main
		if(Einitialize_repeat(h_Ex, Einit_filename))
			exit(EXIT_FAILURE);
	}
	else if(EREADINIT==2) { //read E initial conditions for all cells from filename passed to main
		if(Einitialize(h_Ex, Einit_filename))
			exit(EXIT_FAILURE);
	}
	else {	//use default initial conditions
		sprintf(Einit_filename, "./Einit/Ef20140724_wt_r2g2e80_2_ctd2.txt");
		if(Einitialize(h_Ex, Einit_filename))
			exit(EXIT_FAILURE);
	}
	
	////////////////////////////////////////////////////////////////////////////////
	//Read in parameters to host
	if(MREADPARAM==1) { //read M parameters for 1 cell from filename passed to main
		if(Mpinitialize_repeat(h_Mp, Mparam_filename))
			exit(EXIT_FAILURE);
	}
	else if(MREADPARAM==2) { //read M parameters for all cells from filename passed to main
		if(Mpinitialize(h_Mp, Mparam_filename))
			exit(EXIT_FAILURE);
	}
	else {	//use default M parameters
		sprintf(Mparam_filename, "./Mparameters/20140224_1celldefault.txt");
		if(Mpinitialize_repeat(h_Mp, Mparam_filename))
			exit(EXIT_FAILURE);
	}

	if(EREADPARAM==1) { //read E parameters for 1 cell from filename passed to main
		if(Epinitialize_repeat(h_Ep, Eparam_filename))
			exit(EXIT_FAILURE);
	}
	else if(EREADPARAM==2) { //read E parameters for all cells from filename passed to main
		if(Epinitialize(h_Ep, Eparam_filename))
			exit(EXIT_FAILURE);
	}
	else {	//use default E parameters
                sprintf(Eparam_filename, "./Eparameters/20140429_1cell.txt");
		if(Epinitialize_repeat(h_Ep, Eparam_filename))
			exit(EXIT_FAILURE);
	}
	////////////////////////////////////////////////////////////////////////////////

	Mcheck_init(h_Mx, h_Mp);
	//Pass initial conditions and parameters to device
	CUDA_SAFE_CALL (cudaMalloc((void **) &Mx, sizeof(Mstate)));
	CUDA_SAFE_CALL (cudaMemcpy(Mx, h_Mx, sizeof(Mstate), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMalloc((void **) &Exi, sizeof(Estate)));
	CUDA_SAFE_CALL (cudaMemcpy(Exi, h_Ex, sizeof(Estate), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMalloc((void **) &Exf, sizeof(Estate)));
	CUDA_SAFE_CALL (cudaMemcpy(Exf, h_Ex, sizeof(Estate), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMalloc((void **) &Mp, sizeof(Mparameters)));
	CUDA_SAFE_CALL (cudaMemcpy(Mp, h_Mp, sizeof(Mparameters), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMalloc((void **) &Ep, sizeof(Eparameters)));
	CUDA_SAFE_CALL (cudaMemcpy(Ep, h_Ep, sizeof(Eparameters), cudaMemcpyHostToDevice));
	CHECK_LAUNCH_ERROR();

	////////////////////////////////////////////////////////////////////////////////

	//Allocate connectivity matrix on device
	srand (time(NULL));
	if (MAKECONNECT == 0) {
		if (MREADCONN == 0) // use default Mconnectivity matrix
		{
			sprintf(Mconnect_filename, "./connectivity/connectivity_An_vip.txt");
			printf("Using default VIP connectivity matrix: %s\n", Mconnect_filename);
		}
		if(read_connect(Mconnect_filename, h_MC))
				exit(EXIT_FAILURE);
		if (EREADCONN == 0) // use default Econnectivity matrix
		{
			sprintf(Econnect_filename, "./connectivity/32x32-percents/connectivity_10.txt");
			printf("Using default ephys connectivity matrix: %s\n", Econnect_filename);
		}
		if(read_connect(Econnect_filename, h_EC))
				exit(EXIT_FAILURE);
	}
	else {
		make_rconnect(h_MC,0.10);
		printf("MC made\n");
		make_rconnect(h_EC,0.10);
		printf("EC made\n");
		sprintf(Mconnect_filename,"M%s",out_filename);
		sprintf(Econnect_filename,"E%s",out_filename);
		if(MCPL)
			write_connect(h_MC, Mconnect_filename);
		if(ECPL)
			write_connect(h_EC, Econnect_filename);
	}
	CUDA_SAFE_CALL (cudaMalloc((void **) &MC, ncells*ncells*sizeof(mclock_t)));
	CUDA_SAFE_CALL (cudaMemcpy(MC, h_MC, ncells*ncells*sizeof(mclock_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMalloc((void **) &EC, ncells*ncells*sizeof(ephys_t)));
	CUDA_SAFE_CALL (cudaMemcpy(EC, h_EC, ncells*ncells*sizeof(ephys_t), cudaMemcpyHostToDevice));

	//Allocate and initialize upstream input vector
	for (int i=0; i<ncells; i++){
		h_Minput[i]=0;
		h_Einput[i]=0;
		h_Mupstream[i]=0;
		h_Eupstream[i]=0;
	}
	CUDA_SAFE_CALL (cudaMalloc((void **) &Minput, ncells*sizeof(mclock_t)));
	CUDA_SAFE_CALL (cudaMemcpy(Minput, h_Minput, ncells*sizeof(mclock_t), cudaMemcpyHostToDevice));
       	CUDA_SAFE_CALL (cudaMalloc((void **) &Einput, ncells*sizeof(ephys_t)));
	CUDA_SAFE_CALL (cudaMemcpy(Einput, h_Einput, ncells*sizeof(ephys_t), cudaMemcpyHostToDevice));
       		
	// Count number of upstream cells (assuming h_MC is in column-major format)
	for (int i=0; i<ncells; i++){
		for (int j=0; j<ncells; j++){
			if(h_MC[j+i*ncells]==1){
				h_Mupstream[j]++;
			}
			if(h_EC[j+i*ncells]==1){
				h_Eupstream[j]++;
			}
		}
	}
	for (int i = 0; i < ncells; i++){
		if(h_Mupstream[i]==0){
			h_Mupstream[i]=1;
		}
		if(h_Eupstream[i]==0){
			h_Eupstream[i]=1;
		}
	}

	CUDA_SAFE_CALL (cudaMalloc((void **) &Mupstream, ncells*sizeof(int)));
	CUDA_SAFE_CALL (cudaMemcpy(Mupstream, h_Mupstream, ncells*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMalloc((void **) &Eupstream, ncells*sizeof(int)));
	CUDA_SAFE_CALL (cudaMemcpy(Eupstream, h_Eupstream, ncells*sizeof(int), cudaMemcpyHostToDevice));

	//Initialize CUBLAS
	cublasHandle_t handle;
        cublasStatus_t ret;

        ret = cublasCreate(&handle);

        if (ret != CUBLAS_STATUS_SUCCESS)
        {
        	printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
        	exit(EXIT_FAILURE);
        }

	ephys_t alpha;
	if(EPHYS==1)
	        alpha = 1.0f;
	else
		alpha =  0.00016667f;
        const ephys_t beta  = 0.0f;

	//Allocate and initialize test-step state variables for RK4 method on device for molec clock model
	CUDA_SAFE_CALL (cudaMalloc((void **) &k1, sizeof(Mstate)));
	CUDA_SAFE_CALL (cudaMemcpy(k1, h_Mx, sizeof(Mstate), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL (cudaMalloc((void **) &k2, sizeof(Mstate)));
	CUDA_SAFE_CALL (cudaMemcpy(k2, h_Mx, sizeof(Mstate), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL (cudaMalloc((void **) &k3, sizeof(Mstate)));
	CUDA_SAFE_CALL (cudaMemcpy(k3, h_Mx, sizeof(Mstate), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL (cudaMalloc((void **) &k4, sizeof(Mstate)));
	CUDA_SAFE_CALL (cudaMemcpy(k4, h_Mx, sizeof(Mstate), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL (cudaMalloc((void **) &kb, sizeof(Mstate)));

	CUDA_SAFE_CALL (cudaMemcpy(kb, h_Mx, sizeof(Mstate), cudaMemcpyHostToDevice));

	printf("All memory allocated and initialized successfully.\n");
	//////////////////////////////////////////////////////////////////////
	//////////////////////// For Data Output /////////////////////////////
	//////////////////////////////////////////////////////////////////////
	char path[50];
	sprintf(path,"./output/M%s",out_filename);
	std::ofstream Moutfi(path);

	sprintf(path,"./Minit/Mf%s",out_filename);
	std::ofstream Mfoutfi(path);

	sprintf(path,"./output/EV%s",out_filename);
	std::ofstream EVoutfi(path);

	sprintf(path,"./output/EVmm%s",out_filename);
	std::ofstream EVmmoutfi(path);

//	sprintf(path,"./output/EC%s",out_filename);
//	std::ofstream ECoutfi(path);

	sprintf(path,"./output/ECsummary%s",out_filename);
	std::ofstream ECsummaryoutfi(path);

//	sprintf(path,"./output/EO%s",out_filename);
//	std::ofstream EOoutfi(path);

	sprintf(path,"./Einit/Ef%s",out_filename);
	std::ofstream Efoutfi(path);

//	sprintf(path,"./output/Egaba%s",out_filename);
//	std::ofstream EXoutfi(path);

	std::ofstream performance;// opens an output file for performance info
	sprintf(path,"./performance.txt");
	performance.open(path, std::ofstream::app);
	if (performance.fail())
		std::cout << "can not open performance file";

	printf("Files opened for output sucessfully.\n");

	write_Mparams(h_Mp, out_filename);
	write_Eparams(h_Ep, out_filename);
	//////////////////////////////////////////////////////////////////////
	/////////////////////// Run computation //////////////////////////////
	//////////////////////////////////////////////////////////////////////
	printf("Beginning computation.\n");
	hstTimer nvclk, innerclk;
	nvclk.tic();
	double innertime;

	int LIGHT = 0;

	int iM=0;
	int iE=0;
	double curr_t;

	//Run Ephys for a while to find correct initial conditions
	if (GETEICS && EPHYS) {
		for (int Et_step=0; Et_step < 1000001; Et_step++) {
			curr_t=Et_step*Edt/3600000.0;
			if(ECPL)
	 	      		ret = cublasmv(handle, CUBLAS_OP_N, ncells, ncells, &alpha, EC, ncells, Exf->gaba, 1, &beta, Einput, 1);
			leapfrog_wrapper(Exi, Exf, Ep, Mx, Einput, Eupstream, curr_t);
			leapfrog_copy_wrapper(iE, Eres_len, Eresult, Exi, Exf, Einput, Eupstream);
		}
	}

	record_result_wrapper(0, Mr, Mx, Mp); // record molec clock initial conditions
	for (int Mt_step=0; Mt_step < Mnstep; Mt_step++) {
		if (EPHYS==1) {
			iE=0;

			for (int Et_step=0; Et_step < Enstep; Et_step++) {
				curr_t=Mt_step*Mdt+Et_step*Edt/3600000.0;

				/* for GABA 10 Hz stim */
/*				double stim_len = 5.0;
		                if (fmod(curr_t,24.0) < stim_len && mod(Et_step,1000)==1)
						curr_t = 1.0;
				else	
						curr_t = 0.0;
*/				//////////////////////////////////////////////
				if(ECPL)
	        			ret = cublasmv(handle, CUBLAS_OP_N, ncells, ncells, &alpha, EC, ncells, Exf->gaba, 1, &beta, Einput, 1);
				leapfrog_wrapper(Exi, Exf, Ep, Mx, Einput, Eupstream, curr_t);
				leapfrog_copy_wrapper(iE, Eres_len, Eresult, Exi, Exf, Einput, Eupstream);
				if (Et_step%Erecord==0) {iE+=ncells;}
			}
			if( Mt_step%20==0 ) {
				CUDA_SAFE_CALL (cudaMemcpy(h_Eresult, Eresult, ENRES*Eres_len*sizeof(ephys_t), cudaMemcpyDeviceToHost));
				write_Eresult(h_Eresult, ECsummaryoutfi, 1, Mt_step, 1);

				if(VMAXMIN) {
					EVmmoutfi << Mt_step*Mdt;
					for (int z=0; z <ncells; z++) {
						ret = cublasmax(handle, Eres_len/ncells-1, Eresult+z, ncells, h_Vmax);
						ret = cublasmin(handle, Eres_len/ncells-1, Eresult+z, ncells, h_Vmin);
						EVmmoutfi << " " << h_Eresult[(*h_Vmax-1)*ncells+z] << " " << h_Eresult[(*h_Vmin-1)*ncells+z];
					}
					EVmmoutfi << "\n";
				}

				if( Mt_step%200==0 && Mt_step > (float)Mnstep*6.0/10.0) {
					write_Eresult(h_Eresult, EVoutfi, 0, Mt_step, 0);
//					write_Eresult(h_Eresult, ECoutfi, 1, Mt_step, 0);
//					write_Eresult(h_Eresult, EOoutfi, 2, Mt_step, 0);
				}
			}
//				CUDA_SAFE_CALL (cudaMemcpy(h_Ep->Egaba, Ep->Egaba, ncells*sizeof(ephys_t), cudaMemcpyDeviceToHost));
//				write_array(h_Ep->Egaba, ncells, EXoutfi);
		}
	
		curr_t=Mt_step*Mdt;
		if(MCPL)
		        ret = cublasmv(handle, CUBLAS_OP_N, ncells, ncells, &alpha, MC, ncells, Exf->cac, 1, &beta, Minput, 1);

		rhs_wrapper(curr_t, Mx, k1, Mp, Exf, Minput, Mupstream, LIGHT);
		lincomb_wrapper(.5*Mdt,kb,Mx,k1);
		rhs_wrapper(curr_t+.5*Mdt, kb, k2, Mp, Exf, Minput, Mupstream, LIGHT);
		lincomb_wrapper(.5*Mdt,kb,Mx,k2);
		rhs_wrapper(curr_t+.5*Mdt, kb, k3, Mp, Exf, Minput, Mupstream, LIGHT);
		lincomb_wrapper(Mdt,kb,Mx,k3);
		rhs_wrapper(curr_t+Mdt, kb, k4, Mp, Exf, Minput, Mupstream, LIGHT);
		rk4_wrapper(Mx, k1, k2, k3, k4, curr_t, idx);

		if ((Mt_step+1)%Mrecord==0) {
			printf("recording %d\n", Mt_step+1);
			iM+=ncells;
			record_result_wrapper(iM, Mr, Mx, Mp);
			CUDA_SAFE_CALL (cudaMemcpy(h_Mr, Mr, sizeof(Mresult), cudaMemcpyDeviceToHost));
			if (isnan(h_Mr->ptt[iM])) {
		                CUDA_SAFE_CALL (cudaMemcpy(h_Mx, k1, sizeof(Mstate), cudaMemcpyDeviceToHost));
        		        write_Mfinal(h_Mx,Mfoutfi);
				break;
			}
		}
	}

	CHECK_LAUNCH_ERROR();

	double telapsed = nvclk.toc();
	printf("Runtime: %lf\n",telapsed/(60*1000));

	//////////////////////////////////////////////////////////////////////
	/////////// Read results back to host and print to file //////////////
	//////////////////////////////////////////////////////////////////////
	printf("ready to copy mem back\n");

	CUDA_SAFE_CALL (cudaMemcpy(h_Mr, Mr, sizeof(Mresult), cudaMemcpyDeviceToHost));

//	write_Mresult(h_Mr->pom, Moutfi); // 1
//	write_Mresult(h_Mr->ptm, Moutfi); // 2
//	write_Mresult(h_Mr->rom, Moutfi); // 3
//	write_Mresult(h_Mr->rtm, Moutfi); // 4
//	write_Mresult(h_Mr->bmm, Moutfi); // 5
//	write_Mresult(h_Mr->rvm, Moutfi); // 6
//	write_Mresult(h_Mr->npm, Moutfi); // 7
	write_Mresult(h_Mr->pot, Moutfi); // 8		1
	write_Mresult(h_Mr->ptt, Moutfi); // 9		2
	write_Mresult(h_Mr->rot, Moutfi); // 10		
	write_Mresult(h_Mr->rtt, Moutfi); // 11		
	write_Mresult(h_Mr->bmt, Moutfi); // 12		
//	write_Mresult(h_Mr->clt, Moutfi); // 13
//	write_Mresult(h_Mr->clct, Moutfi); // 14
//	write_Mresult(h_Mr->clnt, Moutfi); // 15
//	write_Mresult(h_Mr->revt, Moutfi); // 16 13
	write_Mresult(h_Mr->cre, Moutfi); // 17 14	3
	write_Mresult(h_Mr->vip, Moutfi); // 18 15	4
	write_Mresult(h_Mr->G, Moutfi); // 19 16	5
//	write_Mresult(h_Mr->gsk, Moutfi); // 20 17	
	write_Mresult(h_Mr->xtra, Moutfi); // 21 18	6

	performance << out_filename << "\t" << "\t" << NTHREADS << "\t" << NBLOCKS << "\t" << ncells << "\t";
	performance << telapsed/(60*1000) << "\t" << telapsed/ncells*(CLKSPD*1e6) << "\t";
	performance << MISD << "\t" << MPSD << "\t" << EISD << "\t" << EPSD << "\t";
	performance << gsyn << "\t" << KO << "\t";
	performance << Mconnect_filename << "\t" << Minit_filename << "\t" << Mparam_filename << "\t";
	performance << Econnect_filename << "\t" << Einit_filename << "\t" << Eparam_filename << "\n";

	if (MFINAL) { //write final state of molecular clock
		CUDA_SAFE_CALL (cudaMemcpy(h_Mx, Mx, sizeof(Mstate), cudaMemcpyDeviceToHost));
		write_Mfinal(h_Mx,Mfoutfi);
	}
	if (EFINAL) { //write final state of ephys
		CUDA_SAFE_CALL (cudaMemcpy(h_Ex, Exf, sizeof(Estate), cudaMemcpyDeviceToHost));
		write_Efinal(h_Ex,Efoutfi);
	}

	///////////////////////////// Close files and free memory ////////////////////////////////
	EVoutfi.close();
	EVmmoutfi.close();
//	ECoutfi.close();
	ECsummaryoutfi.close();
//	EOoutfi.close();
//	EXoutfi.close();
	Efoutfi.close();
	Moutfi.close();
	Mfoutfi.close();
	performance.close();
	if (!EPHYS) {	// remove ephys files if no ephys recorded
		sprintf(path,"./output/EV%s",out_filename);
		remove(path);
//		sprintf(path,"./output/EC%s",out_filename);
//		remove(path);
		sprintf(path,"./output/ECsummary%s",out_filename);
		remove(path);
//		sprintf(path,"./output/EO%s",out_filename);
//		remove(path);
		sprintf(path,"./output/Egaba%s",out_filename);
		remove(path);
	}
	if (!MFINAL) {	// remove mfinal file if not recorded
		sprintf(path,"./output/Mf%s",out_filename);
		remove(path);
	}
	if (!EFINAL) {	// remove efinal file if not recorded
		sprintf(path,"./output/Ef%s",out_filename);
		remove(path);
	}
	if (!VMAXMIN) {	// remove EVmm file if not recorded
		sprintf(path,"./output/EVmm%s",out_filename);
		remove(path);
	}

	free(h_Mr);
	free(h_Eresult);
	free(h_Mx);
	free(h_Ex);
	free(h_Mp);
	free(h_Ep);
	free(h_MC);
	free(h_EC);
	CUDA_SAFE_CALL (cudaFree(Mr));
	CUDA_SAFE_CALL (cudaFree(Eresult));
	CUDA_SAFE_CALL (cudaFree(Mx));
	CUDA_SAFE_CALL (cudaFree(k1));
	CUDA_SAFE_CALL (cudaFree(k2));
	CUDA_SAFE_CALL (cudaFree(k3));
	CUDA_SAFE_CALL (cudaFree(k4));
	CUDA_SAFE_CALL (cudaFree(kb));
	CUDA_SAFE_CALL (cudaFree(Exi));
	CUDA_SAFE_CALL (cudaFree(Exf));
	CUDA_SAFE_CALL (cudaFree(Mp));
	CUDA_SAFE_CALL (cudaFree(Ep));
	CUDA_SAFE_CALL (cudaFree(MC));
	CUDA_SAFE_CALL (cudaFree(EC));
	cublasDestroy(handle);
	return 0;
}
