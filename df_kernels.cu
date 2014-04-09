/* simulate ncells cells using the model from Diekman et al. */
/* equations are solved in PARALLEL on GPUs */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "scn.h"
#include "df.h"

__global__ void leapfrog(Estate *xi, Estate *xf, Eparameters *p, Mstate *M, double *input, int *upstream, double t)
{
	int j;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;
	
	double a, b, c, R, G, E, gkca, gkleak;
	a=1.0/Edt+1.0/(2.0*ts);
	double minf, hinf, ninf, rlinf, rnlinf, fnlinf, sinfi, sinff;
	double taum, tauh, taun, taufnl, tausi, tausf;
	double ya, yb;
	double Isyn;

	for (j=tid; j<ncells; j+=stride) {
		Isyn = 0.0;

	// Use to vary Egaba in time
/*		if( PERCENTEXCITE > 0 )
		{
			double tmp = ((double)j)/ncells;
			if( tmp*100.0 > (100.0-PERCENTEXCITE) )
				p->Egaba[j] = -56 + 24*sin(PI/12.0*t);
		}
*/

//		Isyn = -gsyn*input[j];///(double)upstream[j];
//		Isyn = -gsyn*input[j]*(xi->V[j]-p->Egaba[j]);

//		R = M->G[j];
//		gkca = 500*(R-.03);
//		R = M->x30000[j]/8.0;
//		gkca = 200*R;
//		gkleak = max(0.01+.02*R,0.0);
//		R = .5*(sin(PI/12.0*(t-10.3529))+1.0);
//		R = p->clk[j]*M->G[j];
//		R = p->clk[j]*(.17*sin(PI/12.0*t)+.27);


//		Original: .1 < R/p->clk < .45
//		R = p->clk[j]*(max(M->BC[j]-4.0,0.0))/5.5;
//		gkca = 200*R+.01;
//		gkleak = .2*R+0.01;

//		From PLOS CB paper (-5 < R < 5)
//		R = p->clk[j]*11.36*(M->G[j]-0.3); //original - reverted around 3/20 something = R1
		R = p->clk[j]*11.36*(M->G[j]-0.25); //changed in ~2/2014? = R2
//		R = p->clk[j]*11.36*((17.9416*M->G[j]+25.9390*M->CRE[j]*M->G[j])/40.0 - 0.3); // begin testing on 4/2 = R3

		gkca = 198.0/(1.0+exp(R))+2.0;
		gkleak = 0.2/(1.0+exp(R));

//		gkca = 200;
//		gkleak = 0.0333;

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

		ya = 5.0*input[j]/upstream[j];
		yb = 5.0*input[j]/upstream[j]+0.18;

		//Update gating variables
		xf->m[j] = 2.0*Edt/(2.0*taum+Edt)*minf+(2.0*taum-Edt)/(2.0*taum+Edt)*xi->m[j];
		xf->h[j] = 2.0*Edt/(2.0*tauh+Edt)*hinf+(2.0*tauh-Edt)/(2.0*tauh+Edt)*xi->h[j];
		xf->n[j] = 2.0*Edt/(2.0*taun+Edt)*ninf+(2.0*taun-Edt)/(2.0*taun+Edt)*xi->n[j];
		xf->rl[j] = 2.0*Edt/(2.0*taurl+Edt)*rlinf+(2.0*taurl-Edt)/(2.0*taurl+Edt)*xi->rl[j];
		xf->rnl[j] = 2.0*Edt/(2.0*taurnl+Edt)*rnlinf+(2.0*taurnl-Edt)/(2.0*taurnl+Edt)*xi->rnl[j];
		xf->fnl[j] = 2.0*Edt/(2.0*taufnl+Edt)*fnlinf+(2.0*taufnl-Edt)/(2.0*taufnl+Edt)*xi->fnl[j];
        
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
		G = p->gna[j]*pow(xf->m[j],3)*xf->h[j]+p->gk[j]*pow(xf->n[j],4)+p->gcal[j]*xf->rl[j]*(K1/(K2+xf->cas[j]))+p->gcanl[j]*xf->rnl[j]*xf->fnl[j]+gkca*pow(xf->s[j],2)+gkleak+p->gnaleak[j]+gsyn*xf->y[j];
		E = p->Ena[j]*(p->gna[j]*pow(xf->m[j],3)*xf->h[j]+p->gnaleak[j])+p->Ek[j]*(p->gk[j]*pow(xf->n[j],4)+gkca*pow(xf->s[j],2)+gkleak)+p->Eca[j]*(p->gcal[j]*xf->rl[j]*(K1/(K2+xf->cas[j]))+p->gcanl[j]*xf->rnl[j]*xf->fnl[j])+p->Egaba[j]*gsyn*xf->y[j];

		//Update voltage
		xf->V[j] = 1.0/(p->c[j]+Edt/2.0*G)*(Edt*(Isyn+Iapp+E)+(p->c[j]-Edt/2.0*G)*xi->V[j]);

		//Update post-synaptic current out of this cell
//		xf->out[j] = (1.0-p->lambda[j]*Edt/2.0)/(1.0+p->lambda[j]*Edt/2.0)*xi->out[j];
		xf->out[j] = gsyn*xf->y[j]*(p->Egaba[j]-xf->V[j]); // not really ouput; this is the gaba current
//		xf->gaba[j] = 1.0/(1.0+exp(-(xf->V[j]-2.0)/3.0)); // g1 original from math found neurosci
		xf->gaba[j] = 1.0/(1.0+exp(-(xf->V[j]+20.0)/3.0)); // g2 changed 20140406
//		xf->gaba[j] = 0.5*(1.0+tanh(0.4*xf->V[j]));

		//Check if a spike occured, and if it did increase the post-synaptic current out of this cell
//		if (xf->V[j]<THRESH && xi->V[j]>THRESH)
//			xf->out[j] += PSC;
	}
} 
	
__global__ void leapfrog_copy(int i, int res_len, double *result, Estate *xi, Estate *xf, double *input, int *upstream )
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
		result[i+j] = xf->V[j];
		result[res_len+i+j] = xf->cac[j]; // cytosolic calcium
//		result[2*res_len+i+j] = xf->gaba[j]; // gaba output from each cell
		result[2*res_len+i+j] = xf->out[j]; // output current from each cell
	}
}

void leapfrog_wrapper(Estate *xi, Estate *xf, Eparameters *p, Mstate *M, double *input, int *upstream, double t) {
	leapfrog <<< NBLOCKS, NTHREADS >>> (xi, xf, p, M, input, upstream, t);
}

void leapfrog_copy_wrapper(int i, int res_len, double *result, Estate *xi, Estate *xf, double *input, int *upstream) {
	leapfrog_copy <<< NBLOCKS, NTHREADS >>> (i, res_len, result, xi, xf, input, upstream);
}
