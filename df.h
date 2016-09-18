#ifndef DF_H_
#define DF_H_

#define Edt .1 /* step size (in milliseconds) */
#define Envars 13 //number of variables
#define Enparams 14 //number of parameters
#define Erecord 20 // record data to file every record time steps
#define Enstep 10001
#define ENRES 3 // number of variables to record in results vector

/* constants for gating equations of SCN neuron model in Diekman & Forger 2013 */ 
// Applied current
#define Iapp 0

// L-type inactivation
#define K1 3.93e-5	// mM
#define K2 6.55e-4	// mM

// calcium handling
#define ks 1.65e-4	// mM/fC
#define ts 0.1		// ms
#define bs 0.0

#define kc 8.59e-9	// mM/fC
#define tc 1.75e3	// ms
#define bc 3.1e-8

// time constants
#define taurl 3.1 
#define taurnl 3.1

// synaptic coupling constants
#define gsyn 0.5
#define PERCENTEXCITE -1 //Set 101 for 100% or -1 to use values from read in parameter file
#define EXCITEDIST 2 	//options for how to distribute excitatory cells:
			//0: Random
			//1: first PERCENTEXCITE
			//2: last PERCENTEXCITE

#endif
