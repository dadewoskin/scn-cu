#ifndef KIM_H_
#define KIM_H_

#define Mdt .004 /* step size (in hours) */
#define Mnvars 188 //number of variables
#define Mnparams 85 //number of parameters

#define Mrecord 50 // record data to file every record time steps
#define Mnstep 50000
#define Mrecsteps ((int)(Mnstep/Mrecord)+1) // number of result steps to record

struct Mresult{
	mclock_t pom[ncells*Mrecsteps], ptm[ncells*Mrecsteps], rom[ncells*Mrecsteps], rtm[ncells*Mrecsteps], bmm[ncells*Mrecsteps], rvm[ncells*Mrecsteps], npm[ncells*Mrecsteps], pot[ncells*Mrecsteps], ptt[ncells*Mrecsteps], rot[ncells*Mrecsteps], rtt[ncells*Mrecsteps], bmt[ncells*Mrecsteps], clt[ncells*Mrecsteps], clct[ncells*Mrecsteps], clnt[ncells*Mrecsteps], revt[ncells*Mrecsteps], cre[ncells*Mrecsteps], vip[ncells*Mrecsteps], G[ncells*Mrecsteps], gsk[ncells*Mrecsteps], xtra[ncells*Mrecsteps];
};

#endif
