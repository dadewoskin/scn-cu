#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#define MCPL 1 //turn molecular clock coupling on (1) or off (0)
#define ECPL 1 //turn electrophysiology coupling on (1) or off (0)

#define EPHYS 1 //electrophysiology model on (1) or off (0)
#define	GETEICS 1 //run ephys for a while before starting computation to get ephys initial conditions

#define CPL 1.0 // coupling strength

#define MAKECONNECT 0 //generate coupling matrix (1) or read it in from file (0)
#define MRANDOMINIT 0 //randomally perturb molec clock initial conditions
#define MRANDOMPARAMS 0 //randomally perturb molec clock parameters?

#define ERANDOMINIT 0 // randomally perturb ephys initial conditions?
#define ERANDOMPARAMS 0 //randomally perturb ephys parameters

#define MISD (MRANDOMINIT*0.02) //standard deviation = SD*mean for random perturbations of molec clock initial conditions
#define MPSD (MRANDOMPARAMS*0.02) //standard deviation = SD*mean for random perturbations of molec clock parameters
#define EISD (ERANDOMINIT*0.02) //standard deviation = SD*mean for random perturbations of ephys initial conditions
#define EPSD (ERANDOMPARAMS*0.02) //standard deviation = SD*mean for random perturbations of ephys parameters

/////////////////////
#define KO 0 
//// KNOCKOUTS ////
//0: No knockout
//1: Cry1
//2: Cry2
//3: Per1
//4: VIP
//5: cAMP
//6: Reverb

#define MFINAL 1 //write final state of molecular clock
#define EFINAL 1 //write final state of ephys

#endif
