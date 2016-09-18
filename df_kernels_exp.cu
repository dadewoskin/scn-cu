/* simulate ncells cells using the model from Diekman et al. */
/* equations are solved in PARALLEL on GPUs */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "scn.h"
#include "df.h"
#include "parameters.h"

__global__ void leapfrog(Estate *xi, Estate *xf, Eparameters *p, Mstate *M, ephys_t *input, int *upstream, double t)
{
	int j;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;
	
	ephys_t a, b, c, R, G, E, gkca, gkleak, gnaleak;
	a=1.0/Edt+1.0/(2.0*ts);
	ephys_t minf, hinf, ninf, rlinf, rnlinf, fnlinf, sinfi, sinff;
	ephys_t taum, tauh, taun, taufnl, tausi, tausf;
	ephys_t alpham, alphah, alphan, alpharl, alpharnl, alphafnl;
	ephys_t betam, betah, betan, betarl, betarnl, betafnl;
	ephys_t ya, yb;
	ephys_t Isyn;

	for (j=tid; j<ncells; j+=stride) {
		Isyn = 0.0;

		R = p->clk[j]*11.36*(M->G[j]-0.25); //changed in ~2/2014? = R2

/////////////////////////////////////////////////////////////////////////////////////////////////////

		gkca = 198.0/(1.0+exp(R))+2.0;
		gkleak = 0.2/(1.0+exp(R));

		gnaleak = p->gnaleak[j];

		//Time t+1/2 update:
		//Calculate time constants and eq values
		minf = 1.0/(1.0+exp(-(xi->V[j]+35.2)/8.1));
		taum = exp(-(xi->V[j]+286.0)/160.0);
		
		hinf = 1.0/(1.0+exp((xi->V[j]+62.0)/2.0));
		tauh = 0.51+exp(-(xi->V[j]+26.6)/7.1);

		ninf = 1.0/pow(1.0+exp((xi->V[j]-14.0)/(-17.0)),.25);
		taun = exp(-(xi->V[j]-67.0)/68.0);

		rlinf = 1.0/(1.0+exp(-(xi->V[j]+36.0)/5.1));

		rnlinf = 1.0/(1.0+exp(-(xi->V[j]+21.6)/6.7));
		
		fnlinf = 1.0/(1.0+exp((xi->V[j]+260.0)/65.0));
		taufnl = exp(-(xi->V[j]-444.0)/220.0);

		sinfi = 1e7*pow(xi->cas[j],2)/(1e7*pow(xi->cas[j],2)+5.6);
		sinff = 1e7*pow(xf->cas[j],2)/(1e7*pow(xf->cas[j],2)+5.6);
		tausi = 500.0/(1e7*pow(xi->cas[j],2)+5.6);
		tausf = 500.0/(1e7*pow(xf->cas[j],2)+5.6);

//		ya = 5.0*(input[j]/upstream[j]);
//		yb = 5.0*(input[j]/upstream[j])+0.18; 
		ya = 5.0*(input[j]/100.0);
		yb = 5.0*(input[j]/100.0)+0.18; 

		//Update gating variables
	//	xf->m[j] = 2.0*Edt/(2.0*taum+Edt)*minf+(2.0*taum-Edt)/(2.0*taum+Edt)*xi->m[j];
	//	xf->h[j] = 2.0*Edt/(2.0*tauh+Edt)*hinf+(2.0*tauh-Edt)/(2.0*tauh+Edt)*xi->h[j];
	//	xf->n[j] = 2.0*Edt/(2.0*taun+Edt)*ninf+(2.0*taun-Edt)/(2.0*taun+Edt)*xi->n[j];
	//	xf->rl[j] = 2.0*Edt/(2.0*taurl+Edt)*rlinf+(2.0*taurl-Edt)/(2.0*taurl+Edt)*xi->rl[j];
	//	xf->rnl[j] = 2.0*Edt/(2.0*taurnl+Edt)*rnlinf+(2.0*taurnl-Edt)/(2.0*taurnl+Edt)*xi->rnl[j];
	//	xf->fnl[j] = 2.0*Edt/(2.0*taufnl+Edt)*fnlinf+(2.0*taufnl-Edt)/(2.0*taufnl+Edt)*xi->fnl[j];
		alpham=minf/taum;
		betam=1.0/taum;
		alphah=hinf/tauh;
		betah=1.0/tauh;
		alphan=ninf/taun;
		betan=1.0/taun;
		alpharl=rlinf/taurl;
		betarl=1.0/taurl;
		alpharnl=rnlinf/taurnl;
		betarnl=1.0/taurnl;
		alphafnl=fnlinf/taufnl;
		betafnl=1.0/taufnl;
		xf->m[j] = alpham/betam+(xi->m[j]-alpham/betam)*exp(-betam*Edt);
		xf->h[j] = alphah/betah+(xi->h[j]-alphah/betah)*exp(-betah*Edt);
		xf->n[j] = alphan/betan+(xi->n[j]-alphan/betan)*exp(-betan*Edt);
		xf->rl[j] = alpharl/betarl+(xi->rl[j]-alpharl/betarl)*exp(-betarl*Edt);
		xf->rnl[j] = alpharnl/betarnl+(xi->rnl[j]-alpharnl/betarnl)*exp(-betarnl*Edt);
		xf->fnl[j] = alphafnl/betafnl+(xi->fnl[j]-alphafnl/betafnl)*exp(-betafnl*Edt);
	        
		//solve quadratic equation for cas (a is constant)
		b=(K2-xi->cas[j])/Edt+ks/2.0*p->gcanl[j]*xf->rnl[j]*xf->fnl[j]*(xi->V[j]-p->Eca[j])+(K2+xi->cas[j])/(2.0*ts)-bs+ks/2.0*(p->gcal[j]*xi->rl[j]*K1/(K2+xi->cas[j])*(xi->V[j]-p->Eca[j])+p->gcanl[j]*xi->rnl[j]*xi->fnl[j]*(xi->V[j]-p->Eca[j]));
		c=-K2/Edt*xi->cas[j]+ks/2.0*p->gcal[j]*xf->rl[j]*K1*(xi->V[j]-p->Eca[j])+ks*K2/2.0*p->gcanl[j]*xf->rnl[j]*xf->fnl[j]*(xi->V[j]-p->Eca[j])+K2/(2.0*ts)*xi->cas[j]-bs*K2+K2*ks/2*(p->gcal[j]*xi->rl[j]*K1/(K2+xi->cas[j])*(xi->V[j]-p->Eca[j])+p->gcanl[j]*xi->rnl[j]*xi->fnl[j]*(xi->V[j]-p->Eca[j]));
        
		//Update cas before s and cac (same time step, but s and cac depend on current cas)
		xf->cas[j]=(-b+sqrt(pow(b,2)-4.0*a*c))/(2.0*a);
		xf->s[j]=1.0/(1.0+Edt/(2.0*tausf))*(xi->s[j]*(1.0-Edt/(2.0*tausi))+Edt/2.0*(sinfi/tausi+sinff/tausf));
        
		xf->cac[j]=1.0/(1.0+Edt/(2.0*tc))*(xi->cac[j]*(1.0-Edt/(2.0*tc))+bc*Edt-Edt*kc/2.0*( p->gcal[j]*xf->rl[j]*(K1/(K2+xf->cas[j]))*(xi->V[j]-p->Eca[j])+p->gcanl[j]*xf->rnl[j]*xf->fnl[j]*(xi->V[j]-p->Eca[j])+p->gcal[j]*xi->rl[j]*(K1/(K2+xi->cas[j]))*(xi->V[j]-p->Eca[j])+p->gcanl[j]*xi->rnl[j]*xi->fnl[j]*(xi->V[j]-p->Eca[j])));

		//update synaptic gating variable
		xf->y[j] = (2.0*Edt*ya+(2.0-yb*Edt)*xi->y[j])/(2.0+yb*Edt);

		//Time t+1 update:
		G = p->gna[j]*pow(xf->m[j],3)*xf->h[j]+p->gk[j]*pow(xf->n[j],4)+p->gcal[j]*xf->rl[j]*(K1/(K2+xf->cas[j]))+p->gcanl[j]*xf->rnl[j]*xf->fnl[j]+gkca*pow(xf->s[j],2)+gkleak+gnaleak+gsyn*xf->y[j];
		E = p->Ena[j]*(p->gna[j]*pow(xf->m[j],3)*xf->h[j]+gnaleak)+p->Ek[j]*(p->gk[j]*pow(xf->n[j],4)+gkca*pow(xf->s[j],2)+gkleak)+p->Eca[j]*(p->gcal[j]*xf->rl[j]*(K1/(K2+xf->cas[j]))+p->gcanl[j]*xf->rnl[j]*xf->fnl[j])+p->Egaba[j]*gsyn*xf->y[j];

		//Update voltage
		xf->V[j] = 1.0/(p->c[j]+Edt/2.0*G)*(Edt*(Isyn+Iapp+E)+(p->c[j]-Edt/2.0*G)*xi->V[j]);

		//Update post-synaptic current out of this cell
		xf->out[j] = gsyn*xf->y[j]*(p->Egaba[j]-xf->V[j]); // not really ouput; this is the gaba current the cell experiences
		xf->gaba[j] = 1.0/(1.0+exp(-(xf->V[j]+20.0)/3.0)); // g2 changed 20140406

	}
} 
	
__global__ void leapfrog_copy(int i, int res_len, ephys_t *result, Estate *xi, Estate *xf, ephys_t *input, int *upstream )
{
	// Copy values from end of time step to be initial values of next
	int j;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;
	for (j=tid; j<ncells; j+=stride) {
		xi->V[j] = xf->V[j];
		xi->m[j] = xf->m[j];
		xi->h[j] = xf->h[j];
		xi->n[j] = xf->n[j];
		xi->rl[j] = xf->rl[j];
		xi->rnl[j] = xf->rnl[j];
		xi->fnl[j] = xf->fnl[j];
		xi->s[j] = xf->s[j];
		xi->cas[j] = xf->cas[j];
		xi->cac[j] = xf->cac[j];
		xi->out[j] = xf->out[j];
		xi->gaba[j] = xf->gaba[j];
		xi->y[j] = xf->y[j];
		result[i+j] = xf->V[j]+200*VMAXMIN;
		result[res_len+i+j] = xf->cac[j]; // cytosolic calcium
//		result[2*res_len+i+j] = xf->gaba[j]; // gaba output from each cell
		result[2*res_len+i+j] = xf->out[j]; // output current from each cell
	}
}

void leapfrog_wrapper(Estate *xi, Estate *xf, Eparameters *p, Mstate *M, ephys_t *input, int *upstream, double t) {
	leapfrog <<< NBLOCKS, NTHREADS >>> (xi, xf, p, M, input, upstream, t);
}

void leapfrog_copy_wrapper(int i, int res_len, ephys_t *result, Estate *xi, Estate *xf, ephys_t *input, int *upstream) {
	leapfrog_copy <<< NBLOCKS, NTHREADS >>> (i, res_len, result, xi, xf, input, upstream);
}
