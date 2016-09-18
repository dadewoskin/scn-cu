/* simulate ncells cells using the detailed model from Kim & Forger, MSB 2012 */
/* equations are solved in PARALLEL on GPUs */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "scn.h"
#include "kim.h"
#include "parameters.h"

__global__ void rhs(double t, Mstate *s, Mstate *s_dot, Mparameters *p, Estate *x, mclock_t *input, int *upstream, int LIGHT)
{
	// Calculates right hand side
	int n;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;

	for (n=tid; n<ncells; n+=stride) {
	/* Used for creating PRC */
	// single cell PRC
	// add stim to rhs of whichever state variable you want to add stimulus to //
/*		mclock_t stim = 0;
		mclock_t offset = (mclock_t)(n-1.0)/(ncells-2.0)*24.0;
		mclock_t stim_t = 100+offset;
		mclock_t stim_len	= 2;
		if (n==0)
			stim=0;
		else if (t >= stim_t && t < stim_t+stim_len)
			stim = 10;
*/
	// network PRC
	// add stim to rhs of whichever state variable you want to add stimulus to //
/*		mclock_t stim = 0;
		mclock_t offset = 18.0;
		mclock_t stim_t = 44.0+offset;
		mclock_t stim_len = 6.0;
		if (t >= stim_t && t < stim_t+stim_len)
			stim = 5.0;
*/
	/* Calculate conserved quantities */
		//Free receptor
		mclock_t V00 = p->Vt[n]-s->V10[n]-s->V11[n]-s->V12[n]-s->V01[n]-s->V02[n]; // free receptor
		//Complexes with both CK1 and GSK3B
		mclock_t Kinases = s->x30300[n]+s->x40300[n]+s->x40310[n]+s->x40311[n]+s->x41300[n]+s->x41310[n]+s->x41311[n]+s->x42300[n]+s->x42310[n]+s->x42311[n]+s->x50300[n]+s->x50310[n]+s->x50311[n]+s->x51300[n]+s->x51310[n]+s->x51311[n]+s->x52300[n]+s->x52310[n]+s->x52311[n]+s->x60300[n]+s->x60310[n]+s->x60311[n]+s->x61300[n]+s->x61310[n]+s->x61311[n]+s->x62300[n]+s->x62310[n]+s->x62311[n];
		//Unbound CK1 in cytoplasm (x00100)
		mclock_t x00100 = p->Ct[n] - (Kinases + s->x00110[n]+s->x10100[n]+s->x20100[n]+s->x20110[n]+s->x20111[n]+s->x21100[n]+s->x21110[n]+s->x21111[n]+s->x22100[n]+s->x22110[n]+s->x22111[n]+s->x30100[n]+s->x40100[n]+s->x40110[n]+s->x40111[n]+s->x41100[n]+s->x41110[n]+s->x41111[n]+s->x42100[n]+s->x42110[n]+s->x42111[n]+s->x50100[n]+s->x50110[n]+s->x50111[n]+s->x51100[n]+s->x51110[n]+s->x51111[n]+s->x52100[n]+s->x52110[n]+s->x52111[n]+s->x60100[n]+s->x60110[n]+s->x60111[n]+s->x61100[n]+s->x61110[n]+s->x61111[n]+s->x62100[n]+s->x62110[n]+s->x62111[n]);
		//Unbound GSK3B in cytoplasm (x00200)
		mclock_t x00200 = p->Gt[n] - (Kinases + s->cyrevg[n]+s->revng[n]+s->cyrevgp[n]+s->revngp[n]+s->x00210[n]+s->x30200[n]+s->x40200[n]+s->x40210[n]+s->x40211[n]+s->x41200[n]+s->x41210[n]+s->x41211[n]+s->x42200[n]+s->x42210[n]+s->x42211[n]+s->x50200[n]+s->x50210[n]+s->x50211[n]+s->x51200[n]+s->x51210[n]+s->x51211[n]+s->x52200[n]+s->x52210[n]+s->x52211[n]+s->x60200[n]+s->x60210[n]+s->x60211[n]+s->x61200[n]+s->x61210[n]+s->x61211[n]+s->x62200[n]+s->x62210[n]+s->x62211[n]);

	/* Calculate derivatives */
		s_dot->GR[n] = -(p->unbin[n]*s->GR[n])+p->bin[n]*(1-s->G[n]-s->GR[n])*(s->x01011[n]+s->x02011[n]);
		s_dot->G[n] = -(p->unbin[n]*s->G[n])+p->bin[n]*(1-s->G[n]-s->GR[n])*s->x00011[n];
		s_dot->GrR[n] = -(p->unbinr[n]*s->GrR[n])+p->binr[n]*(1-s->Gr[n]-s->GrR[n])*(s->x01011[n]+s->x02011[n]);
		s_dot->Gr[n] = -(p->unbinr[n]*s->Gr[n])+p->binr[n]*(1-s->Gr[n]-s->GrR[n])*s->x00011[n];
		s_dot->GcR[n] = -(p->unbinc[n]*s->GcR[n])+p->binc[n]*(1-s->Gc[n]-s->GcR[n])*(s->x01011[n]+s->x02011[n]);
		s_dot->Gc[n] = -(p->unbinc[n]*s->Gc[n])+p->binc[n]*(1-s->Gc[n]-s->GcR[n])*s->x00011[n];

//		s_dot->GBR[n] = -(p->unbinrev[n]*s->GBR[n])+p->binrev[n]*s->GB[n]*(s->revn[n]+s->revng[n]+s->revngp[n]+s->revnp[n]);
//		s_dot->GB[n] = p->unbinrev[n]*s->GBR[n]-p->binrev[n]*s->GB[n]*(s->revn[n]+s->revng[n]+s->revngp[n]+s->revnp[n]);
//		s_dot->GBRb[n] = -(p->unbinrevb[n]*s->GBRb[n])+p->binrevb[n]*s->GBb[n]*(s->revn[n]+s->revng[n]+s->revngp[n]+s->revnp[n]);
//		s_dot->GBb[n] = p->unbinrevb[n]*s->GBRb[n]-p->binrevb[n]*s->GBb[n]*(s->revn[n]+s->revng[n]+s->revngp[n]+s->revnp[n]);

//		s_dot->GBR[n] = 0;
		s_dot->GB[n] = p->unbinrev[n]*(1.0-s->GB[n])-p->binrev[n]*s->GB[n]*(s->revn[n]+s->revng[n]+s->revngp[n]+s->revnp[n]);
//		s_dot->GBRb[n] = 0;
		s_dot->GBb[n] = p->unbinrevb[n]*(1.0-s->GBb[n])-p->binrevb[n]*s->GBb[n]*(s->revn[n]+s->revng[n]+s->revngp[n]+s->revnp[n]);

		s_dot->MnPo[n] = p->trPo[n]*(s->G[n]+LIGHT*p->lono[n]*19.9*p->lta[n]*(1.0-s->ltn[n]))+p->CtrPo[n]*s->CRE[n]*s->G[n]-p->tmc[n]*s->MnPo[n]-p->umPo[n]*s->MnPo[n];
		s_dot->McPo[n] = -(p->umPo[n]*s->McPo[n])+p->tmc[n]*s->MnPo[n];
		s_dot->MnPt[n] = p->trPt[n]*(s->G[n]+LIGHT*p->lont[n]*19.9*p->lta[n]*(1.0-s->ltn[n]))+p->CtrPt[n]*s->CRE[n]*s->G[n]-p->tmc[n]*s->MnPt[n]-p->umPt[n]*s->MnPt[n];
		s_dot->McPt[n] = -(p->umPt[n]*s->McPt[n])+p->tmc[n]*s->MnPt[n];

		s_dot->MnRt[n] = p->trRt[n]*s->Gc[n]-p->tmc[n]*s->MnRt[n]-p->umRt[n]*s->MnRt[n];
		s_dot->McRt[n] = -(p->umRt[n]*s->McRt[n])+p->tmc[n]*s->MnRt[n];
		s_dot->MnRev[n] = -(p->tmcrev[n]*s->MnRev[n])-p->umRev[n]*s->MnRev[n]+p->trRev[n]*s->Gr[n]*s->x00011[n];
		s_dot->McRev[n] = -(p->umRev[n]*s->McRev[n])+p->tmcrev[n]*s->MnRev[n];
		s_dot->MnRo[n] = p->trRo[n]*s->G[n]*s->GB[n]-p->tmc[n]*s->MnRo[n]-p->umRo[n]*s->MnRo[n];
		s_dot->McRo[n] = -(p->umRo[n]*s->McRo[n])+p->tmc[n]*s->MnRo[n];
		s_dot->MnB[n] = p->trB[n]*s->GBb[n]-p->tmc[n]*s->MnB[n]-p->umB[n]*s->MnB[n];
		s_dot->McB[n] = -(p->umB[n]*s->McB[n])+p->tmc[n]*s->MnB[n];
		s_dot->MnNp[n] = p->trNp[n]*s->GB[n]-p->tmc[n]*s->MnNp[n]-p->umNp[n]*s->MnNp[n];
		s_dot->McNp[n] = -(p->umNp[n]*s->McNp[n])+p->tmc[n]*s->MnNp[n];
		s_dot->B[n] = -(p->ub[n]*s->B[n])+p->uncbin[n]*s->BC[n]-p->cbin[n]*s->B[n]*s->Cl[n]+p->tlb[n]*s->McB[n];
		s_dot->Cl[n] = p->tlc[n]+p->uncbin[n]*s->BC[n]-p->uc[n]*s->Cl[n]-p->cbin[n]*s->B[n]*s->Cl[n]+p->tlnp[n]*s->McNp[n];
		s_dot->BC[n] = -(p->phos[n]*s->BC[n])-p->ubc[n]*s->BC[n]-p->uncbin[n]*s->BC[n]+p->cbin[n]*s->B[n]*s->Cl[n];
		s_dot->cyrev[n] = -((p->nlrev[n]+p->urev[n])*s->cyrev[n])+p->dg[n]*s->cyrevg[n]+p->tlrev[n]*s->McRev[n]+p->nerev[n]*s->revn[n]-p->ag[n]*s->cyrev[n]*x00200;
		s_dot->revn[n] = p->nlrev[n]*s->cyrev[n]+(-p->nerev[n]-p->urev[n])*s->revn[n]+p->dg[n]*s->revng[n]-p->ag[n]*p->Nf[n]*s->revn[n]*s->x00210[n];
		s_dot->cyrevg[n] = -(s->cyrevg[n]*(p->dg[n]+p->nlrev[n]+p->urev[n]+s->gto[n]))+p->nerev[n]*s->revng[n]+p->ag[n]*s->cyrev[n]*x00200;
		s_dot->revng[n] = p->nlrev[n]*s->cyrevg[n]-(p->dg[n]+p->nerev[n]+p->urev[n]+s->gto[n])*s->revng[n]+p->ag[n]*p->Nf[n]*s->revn[n]*s->x00210[n];
		s_dot->cyrevgp[n] = -((p->dg[n]+p->nlrev[n]+p->uprev[n])*s->cyrevgp[n])+s->cyrevg[n]*s->gto[n]+p->nerev[n]*s->revngp[n];
		s_dot->revngp[n] = p->nlrev[n]*s->cyrevgp[n]+s->gto[n]*s->revng[n]-(p->dg[n]+p->nerev[n]+p->uprev[n])*s->revngp[n];
		s_dot->cyrevp[n] = p->dg[n]*s->cyrevgp[n]-(p->nlrev[n]+p->uprev[n])*s->cyrevp[n]+p->nerev[n]*s->revnp[n];
		s_dot->revnp[n] = p->nlrev[n]*s->cyrevp[n]+p->dg[n]*s->revngp[n]-(p->nerev[n]+p->uprev[n])*s->revnp[n];
		s_dot->gto[n] = p->trgto[n]*s->G[n]*s->GB[n]-p->ugto[n]*s->gto[n];
		s_dot->x00001[n] = p->phos[n]*s->BC[n]-p->nlbc[n]*s->x00001[n]-p->ubc[n]*s->x00001[n];
		s_dot->x00011[n] = p->nlbc[n]*s->x00001[n]-p->ubc[n]*s->x00011[n]+p->uro[n]*s->x01011[n]-p->cbbin[n]*p->Nf[n]*s->x00011[n]*(s->x01010[n]+s->x02010[n])+p->urt[n]*s->x02011[n]+p->uncbbin[n]*(s->x01011[n]+s->x02011[n])+p->upu[n]*(s->x50011[n]+s->x50111[n]+s->x50211[n]+s->x50311[n])+p->up[n]*(s->x20011[n]+s->x20111[n]+s->x40011[n]+s->x40111[n]+s->x40211[n]+s->x40311[n]+s->x60011[n]+s->x60111[n]+s->x60211[n]+s->x60311[n])-p->bbin[n]*p->Nf[n]*s->x00011[n]*(s->x20010[n]+s->x20110[n]+s->x21010[n]+s->x21110[n]+s->x22010[n]+s->x22110[n]+s->x40010[n]+s->x40110[n]+s->x40210[n]+s->x40310[n]+s->x41010[n]+s->x41110[n]+s->x41210[n]+s->x41310[n]+s->x42010[n]+s->x42110[n]+s->x42210[n]+s->x42310[n]+s->x50010[n]+s->x50110[n]+s->x50210[n]+s->x50310[n]+s->x51010[n]+s->x51110[n]+s->x51210[n]+s->x51310[n]+s->x52010[n]+s->x52110[n]+s->x52210[n]+s->x52310[n]+s->x60010[n]+s->x60110[n]+s->x60210[n]+s->x60310[n]+s->x61010[n]+s->x61110[n]+s->x61210[n]+s->x61310[n]+s->x62010[n]+s->x62110[n]+s->x62210[n]+s->x62310[n])+p->unbbin[n]*(s->x20011[n]+s->x20111[n]+s->x21011[n]+s->x21111[n]+s->x22011[n]+s->x22111[n]+s->x40011[n]+s->x40111[n]+s->x40211[n]+s->x40311[n]+s->x41011[n]+s->x41111[n]+s->x41211[n]+s->x41311[n]+s->x42011[n]+s->x42111[n]+s->x42211[n]+s->x42311[n]+s->x50011[n]+s->x50111[n]+s->x50211[n]+s->x50311[n]+s->x51011[n]+s->x51111[n]+s->x51211[n]+s->x51311[n]+s->x52011[n]+s->x52111[n]+s->x52211[n]+s->x52311[n]+s->x60011[n]+s->x60111[n]+s->x60211[n]+s->x60311[n]+s->x61011[n]+s->x61111[n]+s->x61211[n]+s->x61311[n]+s->x62011[n]+s->x62111[n]+s->x62211[n]+s->x62311[n]);
//s_dot->x00100[n] = p->lne[n]*s->x00110[n]+p->upu[n]*(s->x10100[n]+s->x30100[n]+s->x30300[n]+s->x50100[n]+s->x50300[n])+p->up[n]*(s->x20100[n]+s->x40100[n]+s->x40300[n]+s->x60100[n]+s->x60300[n])-p->ac[n]*x00100*(s->x10000[n]+s->x20000[n]+s->x21000[n]+s->x22000[n]+s->x30000[n]+s->x40000[n]+s->x41000[n]+s->x42000[n]+s->x50000[n]+s->x51000[n]+s->x52000[n]+s->x60000[n]+s->x61000[n]+s->x62000[n])+p->dc[n]*(s->x10100[n]+s->x20100[n]+s->x21100[n]+s->x22100[n]+s->x30100[n]+s->x40100[n]+s->x41100[n]+s->x42100[n]+s->x50100[n]+s->x51100[n]+s->x52100[n]+s->x60100[n]+s->x61100[n]+s->x62100[n])-p->ac[n]*x00100*(s->x30200[n]+s->x40200[n]+s->x41200[n]+s->x42200[n]+s->x50200[n]+s->x51200[n]+s->x52200[n]+s->x60200[n]+s->x61200[n]+s->x62200[n])+p->dc[n]*(s->x30300[n]+s->x40300[n]+s->x41300[n]+s->x42300[n]+s->x50300[n]+s->x51300[n]+s->x52300[n]+s->x60300[n]+s->x61300[n]+s->x62300[n]);
		s_dot->x00110[n] = -(p->lne[n]*s->x00110[n])+p->upu[n]*(s->x50110[n]+s->x50111[n]+s->x50310[n]+s->x50311[n])+p->up[n]*(s->x20110[n]+s->x20111[n]+s->x40110[n]+s->x40111[n]+s->x40310[n]+s->x40311[n]+s->x60110[n]+s->x60111[n]+s->x60310[n]+s->x60311[n])-p->ac[n]*p->Nf[n]*s->x00110[n]*(s->x20010[n]+s->x21010[n]+s->x22010[n]+s->x40010[n]+s->x41010[n]+s->x42010[n]+s->x50010[n]+s->x51010[n]+s->x52010[n]+s->x60010[n]+s->x61010[n]+s->x62010[n])-p->ac[n]*p->Nf[n]*s->x00110[n]*(s->x20011[n]+s->x21011[n]+s->x22011[n]+s->x40011[n]+s->x41011[n]+s->x42011[n]+s->x50011[n]+s->x51011[n]+s->x52011[n]+s->x60011[n]+s->x61011[n]+s->x62011[n])+p->dc[n]*(s->x20110[n]+s->x21110[n]+s->x22110[n]+s->x40110[n]+s->x41110[n]+s->x42110[n]+s->x50110[n]+s->x51110[n]+s->x52110[n]+s->x60110[n]+s->x61110[n]+s->x62110[n])+p->dc[n]*(s->x20111[n]+s->x21111[n]+s->x22111[n]+s->x40111[n]+s->x41111[n]+s->x42111[n]+s->x50111[n]+s->x51111[n]+s->x52111[n]+s->x60111[n]+s->x61111[n]+s->x62111[n])-p->ac[n]*p->Nf[n]*s->x00110[n]*(s->x40210[n]+s->x41210[n]+s->x42210[n]+s->x50210[n]+s->x51210[n]+s->x52210[n]+s->x60210[n]+s->x61210[n]+s->x62210[n])-p->ac[n]*p->Nf[n]*s->x00110[n]*(s->x40211[n]+s->x41211[n]+s->x42211[n]+s->x50211[n]+s->x51211[n]+s->x52211[n]+s->x60211[n]+s->x61211[n]+s->x62211[n])+p->dc[n]*(s->x40310[n]+s->x41310[n]+s->x42310[n]+s->x50310[n]+s->x51310[n]+s->x52310[n]+s->x60310[n]+s->x61310[n]+s->x62310[n])+p->dc[n]*(s->x40311[n]+s->x41311[n]+s->x42311[n]+s->x50311[n]+s->x51311[n]+s->x52311[n]+s->x60311[n]+s->x61311[n]+s->x62311[n]);
//s_dot->x00200[n] = p->dg[n]*s->cyrevg[n]+p->urev[n]*s->cyrevg[n]+p->dg[n]*s->cyrevgp[n]+p->uprev[n]*s->cyrevgp[n]-p->ag[n]*s->cyrev[n]*x00200+p->lne[n]*s->x00210[n]+p->upu[n]*(s->x30200[n]+s->x30300[n]+s->x50200[n]+s->x50300[n])+p->up[n]*(s->x40200[n]+s->x40300[n]+s->x60200[n]+s->x60300[n])-p->agp[n]*x00200*(s->x30000[n]+s->x30100[n]+s->x40000[n]+s->x40100[n]+s->x41000[n]+s->x41100[n]+s->x42000[n]+s->x42100[n]+s->x50000[n]+s->x50100[n]+s->x51000[n]+s->x51100[n]+s->x52000[n]+s->x52100[n]+s->x60000[n]+s->x60100[n]+s->x61000[n]+s->x61100[n]+s->x62000[n]+s->x62100[n])+p->dg[n]*(s->x30200[n]+s->x30300[n]+s->x40200[n]+s->x40300[n]+s->x41200[n]+s->x41300[n]+s->x42200[n]+s->x42300[n]+s->x50200[n]+s->x50300[n]+s->x51200[n]+s->x51300[n]+s->x52200[n]+s->x52300[n]+s->x60200[n]+s->x60300[n]+s->x61200[n]+s->x61300[n]+s->x62200[n]+s->x62300[n]);
		s_dot->x00210[n] = p->dg[n]*s->revng[n]+p->urev[n]*s->revng[n]+p->dg[n]*s->revngp[n]+p->uprev[n]*s->revngp[n]-p->lne[n]*s->x00210[n]-p->ag[n]*p->Nf[n]*s->revn[n]*s->x00210[n]+p->upu[n]*(s->x50210[n]+s->x50211[n]+s->x50310[n]+s->x50311[n])+p->up[n]*(s->x40210[n]+s->x40211[n]+s->x40310[n]+s->x40311[n]+s->x60210[n]+s->x60211[n]+s->x60310[n]+s->x60311[n])-p->agp[n]*p->Nf[n]*s->x00210[n]*(s->x40010[n]+s->x40011[n]+s->x40110[n]+s->x40111[n]+s->x41010[n]+s->x41011[n]+s->x41110[n]+s->x41111[n]+s->x42010[n]+s->x42011[n]+s->x42110[n]+s->x42111[n]+s->x50010[n]+s->x50011[n]+s->x50110[n]+s->x50111[n]+s->x51010[n]+s->x51011[n]+s->x51110[n]+s->x51111[n]+s->x52010[n]+s->x52011[n]+s->x52110[n]+s->x52111[n]+s->x60010[n]+s->x60011[n]+s->x60110[n]+s->x60111[n]+s->x61010[n]+s->x61011[n]+s->x61110[n]+s->x61111[n]+s->x62010[n]+s->x62011[n]+s->x62110[n]+s->x62111[n])+p->dg[n]*(s->x40210[n]+s->x40211[n]+s->x40310[n]+s->x40311[n]+s->x41210[n]+s->x41211[n]+s->x41310[n]+s->x41311[n]+s->x42210[n]+s->x42211[n]+s->x42310[n]+s->x42311[n]+s->x50210[n]+s->x50211[n]+s->x50310[n]+s->x50311[n]+s->x51210[n]+s->x51211[n]+s->x51310[n]+s->x51311[n]+s->x52210[n]+s->x52211[n]+s->x52310[n]+s->x52311[n]+s->x60210[n]+s->x60211[n]+s->x60310[n]+s->x60311[n]+s->x61210[n]+s->x61211[n]+s->x61310[n]+s->x61311[n]+s->x62210[n]+s->x62211[n]+s->x62310[n]+s->x62311[n]);
		s_dot->x01000[n] = p->tlr[n]*s->McRo[n]-p->uro[n]*s->x01000[n]-p->ar[n]*s->x01000[n]*(s->x20000[n]+s->x20100[n]+s->x40000[n]+s->x40100[n]+s->x40200[n]+s->x40300[n]+s->x50000[n]+s->x50100[n]+s->x50200[n]+s->x50300[n]+s->x60000[n]+s->x60100[n]+s->x60200[n]+s->x60300[n])+p->dr[n]*(s->x21000[n]+s->x21100[n]+s->x41000[n]+s->x41100[n]+s->x41200[n]+s->x41300[n]+s->x51000[n]+s->x51100[n]+s->x51200[n]+s->x51300[n]+s->x61000[n]+s->x61100[n]+s->x61200[n]+s->x61300[n])+(-p->cvbin[n]*(V00+s->V10[n])*s->x01000[n]+p->uncvbin[n]*(s->V01[n]+s->V11[n]));
		s_dot->x01010[n] = -(p->uro[n]*s->x01010[n])-p->cbbin[n]*p->Nf[n]*s->x00011[n]*s->x01010[n]+p->uncbbin[n]*s->x01011[n]-p->ar[n]*p->Nf[n]*s->x01010[n]*(s->x20010[n]+s->x20110[n]+s->x40010[n]+s->x40110[n]+s->x40210[n]+s->x40310[n]+s->x50010[n]+s->x50110[n]+s->x50210[n]+s->x50310[n]+s->x60010[n]+s->x60110[n]+s->x60210[n]+s->x60310[n])-p->ar[n]*p->Nf[n]*s->x01010[n]*(s->x20011[n]+s->x20111[n]+s->x40011[n]+s->x40111[n]+s->x40211[n]+s->x40311[n]+s->x50011[n]+s->x50111[n]+s->x50211[n]+s->x50311[n]+s->x60011[n]+s->x60111[n]+s->x60211[n]+s->x60311[n])+p->dr[n]*(s->x21010[n]+s->x21110[n]+s->x41010[n]+s->x41110[n]+s->x41210[n]+s->x41310[n]+s->x51010[n]+s->x51110[n]+s->x51210[n]+s->x51310[n]+s->x61010[n]+s->x61110[n]+s->x61210[n]+s->x61310[n])+p->dr[n]*(s->x21011[n]+s->x21111[n]+s->x41011[n]+s->x41111[n]+s->x41211[n]+s->x41311[n]+s->x51011[n]+s->x51111[n]+s->x51211[n]+s->x51311[n]+s->x61011[n]+s->x61111[n]+s->x61211[n]+s->x61311[n]);
		s_dot->x01011[n] = p->cbbin[n]*p->Nf[n]*s->x00011[n]*s->x01010[n]-p->uncbbin[n]*s->x01011[n]-p->uro[n]*s->x01011[n]-p->ar[n]*p->Nf[n]*s->x01011[n]*(s->x20010[n]+s->x20110[n]+s->x40010[n]+s->x40110[n]+s->x40210[n]+s->x40310[n]+s->x50010[n]+s->x50110[n]+s->x50210[n]+s->x50310[n]+s->x60010[n]+s->x60110[n]+s->x60210[n]+s->x60310[n])+p->dr[n]*(s->x21011[n]+s->x21111[n]+s->x41011[n]+s->x41111[n]+s->x41211[n]+s->x41311[n]+s->x51011[n]+s->x51111[n]+s->x51211[n]+s->x51311[n]+s->x61011[n]+s->x61111[n]+s->x61211[n]+s->x61311[n]);
		s_dot->x02000[n] = p->tlr[n]*s->McRt[n]-p->urt[n]*s->x02000[n]-p->ar[n]*s->x02000[n]*(s->x20000[n]+s->x20100[n]+s->x40000[n]+s->x40100[n]+s->x40200[n]+s->x40300[n]+s->x50000[n]+s->x50100[n]+s->x50200[n]+s->x50300[n]+s->x60000[n]+s->x60100[n]+s->x60200[n]+s->x60300[n])+p->dr[n]*(s->x22000[n]+s->x22100[n]+s->x42000[n]+s->x42100[n]+s->x42200[n]+s->x42300[n]+s->x52000[n]+s->x52100[n]+s->x52200[n]+s->x52300[n]+s->x62000[n]+s->x62100[n]+s->x62200[n]+s->x62300[n])+(-p->cvbin[n]*(V00+s->V10[n])*s->x02000[n]+p->uncvbin[n]*(s->V02[n]+s->V12[n]));
		s_dot->x02010[n] = -(p->urt[n]*s->x02010[n])-p->cbbin[n]*p->Nf[n]*s->x00011[n]*s->x02010[n]+p->uncbbin[n]*s->x02011[n]-p->ar[n]*p->Nf[n]*s->x02010[n]*(s->x20010[n]+s->x20110[n]+s->x40010[n]+s->x40110[n]+s->x40210[n]+s->x40310[n]+s->x50010[n]+s->x50110[n]+s->x50210[n]+s->x50310[n]+s->x60010[n]+s->x60110[n]+s->x60210[n]+s->x60310[n])-p->ar[n]*p->Nf[n]*s->x02010[n]*(s->x20011[n]+s->x20111[n]+s->x40011[n]+s->x40111[n]+s->x40211[n]+s->x40311[n]+s->x50011[n]+s->x50111[n]+s->x50211[n]+s->x50311[n]+s->x60011[n]+s->x60111[n]+s->x60211[n]+s->x60311[n])+p->dr[n]*(s->x22010[n]+s->x22110[n]+s->x42010[n]+s->x42110[n]+s->x42210[n]+s->x42310[n]+s->x52010[n]+s->x52110[n]+s->x52210[n]+s->x52310[n]+s->x62010[n]+s->x62110[n]+s->x62210[n]+s->x62310[n])+p->dr[n]*(s->x22011[n]+s->x22111[n]+s->x42011[n]+s->x42111[n]+s->x42211[n]+s->x42311[n]+s->x52011[n]+s->x52111[n]+s->x52211[n]+s->x52311[n]+s->x62011[n]+s->x62111[n]+s->x62211[n]+s->x62311[n]);
		s_dot->x02011[n] = p->cbbin[n]*p->Nf[n]*s->x00011[n]*s->x02010[n]-p->uncbbin[n]*s->x02011[n]-p->urt[n]*s->x02011[n]-p->ar[n]*p->Nf[n]*s->x02011[n]*(s->x20010[n]+s->x20110[n]+s->x40010[n]+s->x40110[n]+s->x40210[n]+s->x40310[n]+s->x50010[n]+s->x50110[n]+s->x50210[n]+s->x50310[n]+s->x60010[n]+s->x60110[n]+s->x60210[n]+s->x60310[n])+p->dr[n]*(s->x22011[n]+s->x22111[n]+s->x42011[n]+s->x42111[n]+s->x42211[n]+s->x42311[n]+s->x52011[n]+s->x52111[n]+s->x52211[n]+s->x52311[n]+s->x62011[n]+s->x62111[n]+s->x62211[n]+s->x62311[n]);
		s_dot->x10000[n] = p->tlp[n]*s->McPo[n]-p->upu[n]*s->x10000[n]-p->ac[n]*x00100*s->x10000[n]+p->dc[n]*s->x10100[n];
		s_dot->x10100[n] = p->ac[n]*x00100*s->x10000[n]-p->dc[n]*s->x10100[n]-p->hoo[n]*s->x10100[n]-p->upu[n]*s->x10100[n];
		s_dot->x20000[n] = -(p->nl[n]*s->x20000[n])-p->up[n]*s->x20000[n]-p->ac[n]*x00100*s->x20000[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x20000[n]+p->ne[n]*s->x20010[n]+p->dc[n]*s->x20100[n]+p->dr[n]*(s->x21000[n]+s->x22000[n]);
		s_dot->x20010[n] = p->nl[n]*s->x20000[n]-p->ne[n]*s->x20010[n]-p->up[n]*s->x20010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x20010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x20010[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x20010[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x20010[n]+p->ubc[n]*s->x20011[n]+p->unbbin[n]*s->x20011[n]+p->dc[n]*s->x20110[n]+p->dr[n]*(s->x21010[n]+s->x22010[n])+p->dr[n]*(s->x21011[n]+s->x22011[n]);
		s_dot->x20011[n] = p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x20010[n]-p->ubc[n]*s->x20011[n]-p->unbbin[n]*s->x20011[n]-p->up[n]*s->x20011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x20011[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x20011[n]+p->dc[n]*s->x20111[n]+p->dr[n]*(s->x21011[n]+s->x22011[n]);
		s_dot->x20100[n] = p->hoo[n]*s->x10100[n]+p->ac[n]*x00100*s->x20000[n]-p->dc[n]*s->x20100[n]-p->nl[n]*s->x20100[n]-p->up[n]*s->x20100[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x20100[n]+p->ne[n]*s->x20110[n]+p->dr[n]*(s->x21100[n]+s->x22100[n]);
		s_dot->x20110[n] = p->ac[n]*p->Nf[n]*s->x00110[n]*s->x20010[n]+p->nl[n]*s->x20100[n]-p->dc[n]*s->x20110[n]-p->ne[n]*s->x20110[n]-p->up[n]*s->x20110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x20110[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x20110[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x20110[n]+p->ubc[n]*s->x20111[n]+p->unbbin[n]*s->x20111[n]+p->dr[n]*(s->x21110[n]+s->x22110[n])+p->dr[n]*(s->x21111[n]+s->x22111[n]);
		s_dot->x20111[n] = p->ac[n]*p->Nf[n]*s->x00110[n]*s->x20011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x20110[n]-p->dc[n]*s->x20111[n]-p->ubc[n]*s->x20111[n]-p->unbbin[n]*s->x20111[n]-p->up[n]*s->x20111[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x20111[n]+p->dr[n]*(s->x21111[n]+s->x22111[n]);
		s_dot->x21000[n] = p->ar[n]*s->x01000[n]*s->x20000[n]-p->dr[n]*s->x21000[n]-p->nl[n]*s->x21000[n]-p->ac[n]*x00100*s->x21000[n]+p->ne[n]*s->x21010[n]+p->dc[n]*s->x21100[n];
		s_dot->x21010[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x20010[n]+p->nl[n]*s->x21000[n]-p->dr[n]*s->x21010[n]-p->ne[n]*s->x21010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x21010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x21010[n]+p->unbbin[n]*s->x21011[n]+p->dc[n]*s->x21110[n];
		s_dot->x21011[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x20010[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x20011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x21010[n]-2*p->dr[n]*s->x21011[n]-p->unbbin[n]*s->x21011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x21011[n]+p->dc[n]*s->x21111[n];
		s_dot->x21100[n] = p->ar[n]*s->x01000[n]*s->x20100[n]+p->ac[n]*x00100*s->x21000[n]-p->dc[n]*s->x21100[n]-p->dr[n]*s->x21100[n]-p->nl[n]*s->x21100[n]+p->ne[n]*s->x21110[n];
		s_dot->x21110[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x20110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x21010[n]+p->nl[n]*s->x21100[n]-p->dc[n]*s->x21110[n]-p->dr[n]*s->x21110[n]-p->ne[n]*s->x21110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x21110[n]+p->unbbin[n]*s->x21111[n];
		s_dot->x21111[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x20110[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x20111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x21011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x21110[n]-p->dc[n]*s->x21111[n]-2*p->dr[n]*s->x21111[n]-p->unbbin[n]*s->x21111[n];
		s_dot->x22000[n] = p->ar[n]*s->x02000[n]*s->x20000[n]-p->dr[n]*s->x22000[n]-p->nl[n]*s->x22000[n]-p->ac[n]*x00100*s->x22000[n]+p->ne[n]*s->x22010[n]+p->dc[n]*s->x22100[n];
		s_dot->x22010[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x20010[n]+p->nl[n]*s->x22000[n]-p->dr[n]*s->x22010[n]-p->ne[n]*s->x22010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x22010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x22010[n]+p->unbbin[n]*s->x22011[n]+p->dc[n]*s->x22110[n];
		s_dot->x22011[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x20010[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x20011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x22010[n]-2*p->dr[n]*s->x22011[n]-p->unbbin[n]*s->x22011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x22011[n]+p->dc[n]*s->x22111[n];
		s_dot->x22100[n] = p->ar[n]*s->x02000[n]*s->x20100[n]+p->ac[n]*x00100*s->x22000[n]-p->dc[n]*s->x22100[n]-p->dr[n]*s->x22100[n]-p->nl[n]*s->x22100[n]+p->ne[n]*s->x22110[n];
		s_dot->x22110[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x20110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x22010[n]+p->nl[n]*s->x22100[n]-p->dc[n]*s->x22110[n]-p->dr[n]*s->x22110[n]-p->ne[n]*s->x22110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x22110[n]+p->unbbin[n]*s->x22111[n];
		s_dot->x22111[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x20110[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x20111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x22011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x22110[n]-p->dc[n]*s->x22111[n]-2*p->dr[n]*s->x22111[n]-p->unbbin[n]*s->x22111[n];
		s_dot->x30000[n] = p->tlp[n]*s->McPt[n]-p->upu[n]*s->x30000[n]-p->ac[n]*x00100*s->x30000[n]-p->agp[n]*x00200*s->x30000[n]+p->dc[n]*s->x30100[n]+p->dg[n]*s->x30200[n];
		s_dot->x30100[n] = p->ac[n]*x00100*s->x30000[n]-p->dc[n]*s->x30100[n]-p->hto[n]*s->x30100[n]-p->upu[n]*s->x30100[n]-p->agp[n]*x00200*s->x30100[n]+p->dg[n]*s->x30300[n];
		s_dot->x30200[n] = p->agp[n]*x00200*s->x30000[n]-p->dg[n]*s->x30200[n]-p->upu[n]*s->x30200[n]-s->gto[n]*s->x30200[n]-p->ac[n]*x00100*s->x30200[n]+p->dc[n]*s->x30300[n];
		s_dot->x30300[n] = p->agp[n]*x00200*s->x30100[n]+p->ac[n]*x00100*s->x30200[n]-p->dc[n]*s->x30300[n]-p->dg[n]*s->x30300[n]-p->hto[n]*s->x30300[n]-p->upu[n]*s->x30300[n]-s->gto[n]*s->x30300[n];
		s_dot->x40000[n] = -(p->nl[n]*s->x40000[n])-p->up[n]*s->x40000[n]-p->ac[n]*x00100*s->x40000[n]-p->agp[n]*x00200*s->x40000[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x40000[n]+p->ne[n]*s->x40010[n]+p->dc[n]*s->x40100[n]+p->dg[n]*s->x40200[n]+p->dr[n]*(s->x41000[n]+s->x42000[n]);
		s_dot->x40010[n] = p->nl[n]*s->x40000[n]-p->ne[n]*s->x40010[n]-p->up[n]*s->x40010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40010[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40010[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x40010[n]+p->ubc[n]*s->x40011[n]+p->unbbin[n]*s->x40011[n]+p->dc[n]*s->x40110[n]+p->dg[n]*s->x40210[n]+p->dr[n]*(s->x41010[n]+s->x42010[n])+p->dr[n]*(s->x41011[n]+s->x42011[n]);
		s_dot->x40011[n] = p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40010[n]-p->ubc[n]*s->x40011[n]-p->unbbin[n]*s->x40011[n]-p->up[n]*s->x40011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40011[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40011[n]+p->dc[n]*s->x40111[n]+p->dg[n]*s->x40211[n]+p->dr[n]*(s->x41011[n]+s->x42011[n]);
		s_dot->x40100[n] = p->hto[n]*s->x30100[n]+p->ac[n]*x00100*s->x40000[n]-p->dc[n]*s->x40100[n]-p->nl[n]*s->x40100[n]-p->up[n]*s->x40100[n]-p->agp[n]*x00200*s->x40100[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x40100[n]+p->ne[n]*s->x40110[n]+p->dg[n]*s->x40300[n]+p->dr[n]*(s->x41100[n]+s->x42100[n]);
		s_dot->x40110[n] = p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40010[n]+p->nl[n]*s->x40100[n]-p->dc[n]*s->x40110[n]-p->ne[n]*s->x40110[n]-p->up[n]*s->x40110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40110[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40110[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x40110[n]+p->ubc[n]*s->x40111[n]+p->unbbin[n]*s->x40111[n]+p->dg[n]*s->x40310[n]+p->dr[n]*(s->x41110[n]+s->x42110[n])+p->dr[n]*(s->x41111[n]+s->x42111[n]);
		s_dot->x40111[n] = p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40110[n]-p->dc[n]*s->x40111[n]-p->ubc[n]*s->x40111[n]-p->unbbin[n]*s->x40111[n]-p->up[n]*s->x40111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40111[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40111[n]+p->dg[n]*s->x40311[n]+p->dr[n]*(s->x41111[n]+s->x42111[n]);
		s_dot->x40200[n] = p->agp[n]*x00200*s->x40000[n]-p->dg[n]*s->x40200[n]-p->nl[n]*s->x40200[n]-p->up[n]*s->x40200[n]-s->gto[n]*s->x40200[n]-p->ac[n]*x00100*s->x40200[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x40200[n]+p->ne[n]*s->x40210[n]+p->dc[n]*s->x40300[n]+p->dr[n]*(s->x41200[n]+s->x42200[n]);
		s_dot->x40210[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40010[n]+p->nl[n]*s->x40200[n]-p->dg[n]*s->x40210[n]-p->ne[n]*s->x40210[n]-p->up[n]*s->x40210[n]-s->gto[n]*s->x40210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40210[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40210[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x40210[n]+p->ubc[n]*s->x40211[n]+p->unbbin[n]*s->x40211[n]+p->dc[n]*s->x40310[n]+p->dr[n]*(s->x41210[n]+s->x42210[n])+p->dr[n]*(s->x41211[n]+s->x42211[n]);
		s_dot->x40211[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40210[n]-p->dg[n]*s->x40211[n]-p->ubc[n]*s->x40211[n]-p->unbbin[n]*s->x40211[n]-p->up[n]*s->x40211[n]-s->gto[n]*s->x40211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40211[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40211[n]+p->dc[n]*s->x40311[n]+p->dr[n]*(s->x41211[n]+s->x42211[n]);
		s_dot->x40300[n] = p->hto[n]*s->x30300[n]+p->agp[n]*x00200*s->x40100[n]+p->ac[n]*x00100*s->x40200[n]-p->dc[n]*s->x40300[n]-p->dg[n]*s->x40300[n]-p->nl[n]*s->x40300[n]-p->up[n]*s->x40300[n]-s->gto[n]*s->x40300[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x40300[n]+p->ne[n]*s->x40310[n]+p->dr[n]*(s->x41300[n]+s->x42300[n]);
		s_dot->x40310[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40210[n]+p->nl[n]*s->x40300[n]-p->dc[n]*s->x40310[n]-p->dg[n]*s->x40310[n]-p->ne[n]*s->x40310[n]-p->up[n]*s->x40310[n]-s->gto[n]*s->x40310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40310[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40310[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x40310[n]+p->ubc[n]*s->x40311[n]+p->unbbin[n]*s->x40311[n]+p->dr[n]*(s->x41310[n]+s->x42310[n])+p->dr[n]*(s->x41311[n]+s->x42311[n]);
		s_dot->x40311[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x40111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x40211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x40310[n]-p->dc[n]*s->x40311[n]-p->dg[n]*s->x40311[n]-p->ubc[n]*s->x40311[n]-p->unbbin[n]*s->x40311[n]-p->up[n]*s->x40311[n]-s->gto[n]*s->x40311[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x40311[n]+p->dr[n]*(s->x41311[n]+s->x42311[n]);
		s_dot->x41000[n] = p->ar[n]*s->x01000[n]*s->x40000[n]-p->dr[n]*s->x41000[n]-p->nl[n]*s->x41000[n]-p->ac[n]*x00100*s->x41000[n]-p->agp[n]*x00200*s->x41000[n]+p->ne[n]*s->x41010[n]+p->dc[n]*s->x41100[n]+p->dg[n]*s->x41200[n];
		s_dot->x41010[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40010[n]+p->nl[n]*s->x41000[n]-p->dr[n]*s->x41010[n]-p->ne[n]*s->x41010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41010[n]+p->unbbin[n]*s->x41011[n]+p->dc[n]*s->x41110[n]+p->dg[n]*s->x41210[n];
		s_dot->x41011[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x40010[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41010[n]-2*p->dr[n]*s->x41011[n]-p->unbbin[n]*s->x41011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41011[n]+p->dc[n]*s->x41111[n]+p->dg[n]*s->x41211[n];
		s_dot->x41100[n] = p->ar[n]*s->x01000[n]*s->x40100[n]+p->ac[n]*x00100*s->x41000[n]-p->dc[n]*s->x41100[n]-p->dr[n]*s->x41100[n]-p->nl[n]*s->x41100[n]-p->agp[n]*x00200*s->x41100[n]+p->ne[n]*s->x41110[n]+p->dg[n]*s->x41300[n];
		s_dot->x41110[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41010[n]+p->nl[n]*s->x41100[n]-p->dc[n]*s->x41110[n]-p->dr[n]*s->x41110[n]-p->ne[n]*s->x41110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41110[n]+p->unbbin[n]*s->x41111[n]+p->dg[n]*s->x41310[n];
		s_dot->x41111[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x40110[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41110[n]-p->dc[n]*s->x41111[n]-2*p->dr[n]*s->x41111[n]-p->unbbin[n]*s->x41111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41111[n]+p->dg[n]*s->x41311[n];
		s_dot->x41200[n] = p->ar[n]*s->x01000[n]*s->x40200[n]+p->agp[n]*x00200*s->x41000[n]-p->dg[n]*s->x41200[n]-p->dr[n]*s->x41200[n]-p->nl[n]*s->x41200[n]-s->gto[n]*s->x41200[n]-p->ac[n]*x00100*s->x41200[n]+p->ne[n]*s->x41210[n]+p->dc[n]*s->x41300[n];
		s_dot->x41210[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41010[n]+p->nl[n]*s->x41200[n]-p->dg[n]*s->x41210[n]-p->dr[n]*s->x41210[n]-p->ne[n]*s->x41210[n]-s->gto[n]*s->x41210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41210[n]+p->unbbin[n]*s->x41211[n]+p->dc[n]*s->x41310[n];
		s_dot->x41211[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x40210[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41210[n]-p->dg[n]*s->x41211[n]-2*p->dr[n]*s->x41211[n]-p->unbbin[n]*s->x41211[n]-s->gto[n]*s->x41211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41211[n]+p->dc[n]*s->x41311[n];
		s_dot->x41300[n] = p->ar[n]*s->x01000[n]*s->x40300[n]+p->agp[n]*x00200*s->x41100[n]+p->ac[n]*x00100*s->x41200[n]-p->dc[n]*s->x41300[n]-p->dg[n]*s->x41300[n]-p->dr[n]*s->x41300[n]-p->nl[n]*s->x41300[n]-s->gto[n]*s->x41300[n]+p->ne[n]*s->x41310[n];
		s_dot->x41310[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41210[n]+p->nl[n]*s->x41300[n]-p->dc[n]*s->x41310[n]-p->dg[n]*s->x41310[n]-p->dr[n]*s->x41310[n]-p->ne[n]*s->x41310[n]-s->gto[n]*s->x41310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41310[n]+p->unbbin[n]*s->x41311[n];
		s_dot->x41311[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x40310[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x40311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x41111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x41211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x41310[n]-p->dc[n]*s->x41311[n]-p->dg[n]*s->x41311[n]-2*p->dr[n]*s->x41311[n]-p->unbbin[n]*s->x41311[n]-s->gto[n]*s->x41311[n];
		s_dot->x42000[n] = p->ar[n]*s->x02000[n]*s->x40000[n]-p->dr[n]*s->x42000[n]-p->nl[n]*s->x42000[n]-p->ac[n]*x00100*s->x42000[n]-p->agp[n]*x00200*s->x42000[n]+p->ne[n]*s->x42010[n]+p->dc[n]*s->x42100[n]+p->dg[n]*s->x42200[n];
		s_dot->x42010[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40010[n]+p->nl[n]*s->x42000[n]-p->dr[n]*s->x42010[n]-p->ne[n]*s->x42010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42010[n]+p->unbbin[n]*s->x42011[n]+p->dc[n]*s->x42110[n]+p->dg[n]*s->x42210[n];
		s_dot->x42011[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x40010[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42010[n]-2*p->dr[n]*s->x42011[n]-p->unbbin[n]*s->x42011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42011[n]+p->dc[n]*s->x42111[n]+p->dg[n]*s->x42211[n];
		s_dot->x42100[n] = p->ar[n]*s->x02000[n]*s->x40100[n]+p->ac[n]*x00100*s->x42000[n]-p->dc[n]*s->x42100[n]-p->dr[n]*s->x42100[n]-p->nl[n]*s->x42100[n]-p->agp[n]*x00200*s->x42100[n]+p->ne[n]*s->x42110[n]+p->dg[n]*s->x42300[n];
		s_dot->x42110[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42010[n]+p->nl[n]*s->x42100[n]-p->dc[n]*s->x42110[n]-p->dr[n]*s->x42110[n]-p->ne[n]*s->x42110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42110[n]+p->unbbin[n]*s->x42111[n]+p->dg[n]*s->x42310[n];
		s_dot->x42111[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x40110[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42110[n]-p->dc[n]*s->x42111[n]-2*p->dr[n]*s->x42111[n]-p->unbbin[n]*s->x42111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42111[n]+p->dg[n]*s->x42311[n];
		s_dot->x42200[n] = p->ar[n]*s->x02000[n]*s->x40200[n]+p->agp[n]*x00200*s->x42000[n]-p->dg[n]*s->x42200[n]-p->dr[n]*s->x42200[n]-p->nl[n]*s->x42200[n]-s->gto[n]*s->x42200[n]-p->ac[n]*x00100*s->x42200[n]+p->ne[n]*s->x42210[n]+p->dc[n]*s->x42300[n];
		s_dot->x42210[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42010[n]+p->nl[n]*s->x42200[n]-p->dg[n]*s->x42210[n]-p->dr[n]*s->x42210[n]-p->ne[n]*s->x42210[n]-s->gto[n]*s->x42210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42210[n]+p->unbbin[n]*s->x42211[n]+p->dc[n]*s->x42310[n];
		s_dot->x42211[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x40210[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42210[n]-p->dg[n]*s->x42211[n]-2*p->dr[n]*s->x42211[n]-p->unbbin[n]*s->x42211[n]-s->gto[n]*s->x42211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42211[n]+p->dc[n]*s->x42311[n];
		s_dot->x42300[n] = p->ar[n]*s->x02000[n]*s->x40300[n]+p->agp[n]*x00200*s->x42100[n]+p->ac[n]*x00100*s->x42200[n]-p->dc[n]*s->x42300[n]-p->dg[n]*s->x42300[n]-p->dr[n]*s->x42300[n]-p->nl[n]*s->x42300[n]-s->gto[n]*s->x42300[n]+p->ne[n]*s->x42310[n];
		s_dot->x42310[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42210[n]+p->nl[n]*s->x42300[n]-p->dc[n]*s->x42310[n]-p->dg[n]*s->x42310[n]-p->dr[n]*s->x42310[n]-p->ne[n]*s->x42310[n]-s->gto[n]*s->x42310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42310[n]+p->unbbin[n]*s->x42311[n];
		s_dot->x42311[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x40310[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x40311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x42111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x42211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x42310[n]-p->dc[n]*s->x42311[n]-p->dg[n]*s->x42311[n]-2*p->dr[n]*s->x42311[n]-p->unbbin[n]*s->x42311[n]-s->gto[n]*s->x42311[n];
		s_dot->x50000[n] = -(p->nl[n]*s->x50000[n])-p->upu[n]*s->x50000[n]-p->ac[n]*x00100*s->x50000[n]-p->agp[n]*x00200*s->x50000[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x50000[n]+p->ne[n]*s->x50010[n]+p->dc[n]*s->x50100[n]+p->dg[n]*s->x50200[n]+p->dr[n]*(s->x51000[n]+s->x52000[n]);
		s_dot->x50010[n] = p->nl[n]*s->x50000[n]-p->ne[n]*s->x50010[n]-p->upu[n]*s->x50010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50010[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50010[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x50010[n]+p->ubc[n]*s->x50011[n]+p->unbbin[n]*s->x50011[n]+p->dc[n]*s->x50110[n]+p->dg[n]*s->x50210[n]+p->dr[n]*(s->x51010[n]+s->x52010[n])+p->dr[n]*(s->x51011[n]+s->x52011[n]);
		s_dot->x50011[n] = p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50010[n]-p->ubc[n]*s->x50011[n]-p->unbbin[n]*s->x50011[n]-p->upu[n]*s->x50011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50011[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50011[n]+p->dc[n]*s->x50111[n]+p->dg[n]*s->x50211[n]+p->dr[n]*(s->x51011[n]+s->x52011[n]);
		s_dot->x50100[n] = p->ac[n]*x00100*s->x50000[n]-p->dc[n]*s->x50100[n]-p->hto[n]*s->x50100[n]-p->nl[n]*s->x50100[n]-p->upu[n]*s->x50100[n]-p->agp[n]*x00200*s->x50100[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x50100[n]+p->ne[n]*s->x50110[n]+p->dg[n]*s->x50300[n]+p->dr[n]*(s->x51100[n]+s->x52100[n]);
		s_dot->x50110[n] = p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50010[n]+p->nl[n]*s->x50100[n]-p->dc[n]*s->x50110[n]-p->hto[n]*s->x50110[n]-p->ne[n]*s->x50110[n]-p->upu[n]*s->x50110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50110[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50110[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x50110[n]+p->ubc[n]*s->x50111[n]+p->unbbin[n]*s->x50111[n]+p->dg[n]*s->x50310[n]+p->dr[n]*(s->x51110[n]+s->x52110[n])+p->dr[n]*(s->x51111[n]+s->x52111[n]);
		s_dot->x50111[n] = p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50110[n]-p->dc[n]*s->x50111[n]-p->hto[n]*s->x50111[n]-p->ubc[n]*s->x50111[n]-p->unbbin[n]*s->x50111[n]-p->upu[n]*s->x50111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50111[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50111[n]+p->dg[n]*s->x50311[n]+p->dr[n]*(s->x51111[n]+s->x52111[n]);
		s_dot->x50200[n] = s->gto[n]*s->x30200[n]+p->agp[n]*x00200*s->x50000[n]-p->dg[n]*s->x50200[n]-p->nl[n]*s->x50200[n]-p->upu[n]*s->x50200[n]-p->ac[n]*x00100*s->x50200[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x50200[n]+p->ne[n]*s->x50210[n]+p->dc[n]*s->x50300[n]+p->dr[n]*(s->x51200[n]+s->x52200[n]);
		s_dot->x50210[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50010[n]+p->nl[n]*s->x50200[n]-p->dg[n]*s->x50210[n]-p->ne[n]*s->x50210[n]-p->upu[n]*s->x50210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50210[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50210[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x50210[n]+p->ubc[n]*s->x50211[n]+p->unbbin[n]*s->x50211[n]+p->dc[n]*s->x50310[n]+p->dr[n]*(s->x51210[n]+s->x52210[n])+p->dr[n]*(s->x51211[n]+s->x52211[n]);
		s_dot->x50211[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50210[n]-p->dg[n]*s->x50211[n]-p->ubc[n]*s->x50211[n]-p->unbbin[n]*s->x50211[n]-p->upu[n]*s->x50211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50211[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50211[n]+p->dc[n]*s->x50311[n]+p->dr[n]*(s->x51211[n]+s->x52211[n]);
		s_dot->x50300[n] = s->gto[n]*s->x30300[n]+p->agp[n]*x00200*s->x50100[n]+p->ac[n]*x00100*s->x50200[n]-p->dc[n]*s->x50300[n]-p->dg[n]*s->x50300[n]-p->hto[n]*s->x50300[n]-p->nl[n]*s->x50300[n]-p->upu[n]*s->x50300[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x50300[n]+p->ne[n]*s->x50310[n]+p->dr[n]*(s->x51300[n]+s->x52300[n]);
		s_dot->x50310[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50210[n]+p->nl[n]*s->x50300[n]-p->dc[n]*s->x50310[n]-p->dg[n]*s->x50310[n]-p->hto[n]*s->x50310[n]-p->ne[n]*s->x50310[n]-p->upu[n]*s->x50310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50310[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50310[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x50310[n]+p->ubc[n]*s->x50311[n]+p->unbbin[n]*s->x50311[n]+p->dr[n]*(s->x51310[n]+s->x52310[n])+p->dr[n]*(s->x51311[n]+s->x52311[n]);
		s_dot->x50311[n] = p->agp[n]*p->Nf[n]*s->x00210[n]*s->x50111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x50211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x50310[n]-p->dc[n]*s->x50311[n]-p->dg[n]*s->x50311[n]-p->hto[n]*s->x50311[n]-p->ubc[n]*s->x50311[n]-p->unbbin[n]*s->x50311[n]-p->upu[n]*s->x50311[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x50311[n]+p->dr[n]*(s->x51311[n]+s->x52311[n]);
		s_dot->x51000[n] = p->ar[n]*s->x01000[n]*s->x50000[n]-p->dr[n]*s->x51000[n]-p->nl[n]*s->x51000[n]-p->ac[n]*x00100*s->x51000[n]-p->agp[n]*x00200*s->x51000[n]+p->ne[n]*s->x51010[n]+p->dc[n]*s->x51100[n]+p->dg[n]*s->x51200[n];
		s_dot->x51010[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50010[n]+p->nl[n]*s->x51000[n]-p->dr[n]*s->x51010[n]-p->ne[n]*s->x51010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51010[n]+p->unbbin[n]*s->x51011[n]+p->dc[n]*s->x51110[n]+p->dg[n]*s->x51210[n];
		s_dot->x51011[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x50010[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51010[n]-2*p->dr[n]*s->x51011[n]-p->unbbin[n]*s->x51011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51011[n]+p->dc[n]*s->x51111[n]+p->dg[n]*s->x51211[n];
		s_dot->x51100[n] = p->ar[n]*s->x01000[n]*s->x50100[n]+p->ac[n]*x00100*s->x51000[n]-p->dc[n]*s->x51100[n]-p->dr[n]*s->x51100[n]-p->nl[n]*s->x51100[n]-p->agp[n]*x00200*s->x51100[n]+p->ne[n]*s->x51110[n]+p->dg[n]*s->x51300[n];
		s_dot->x51110[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51010[n]+p->nl[n]*s->x51100[n]-p->dc[n]*s->x51110[n]-p->dr[n]*s->x51110[n]-p->ne[n]*s->x51110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51110[n]+p->unbbin[n]*s->x51111[n]+p->dg[n]*s->x51310[n];
		s_dot->x51111[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x50110[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51110[n]-p->dc[n]*s->x51111[n]-2*p->dr[n]*s->x51111[n]-p->unbbin[n]*s->x51111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51111[n]+p->dg[n]*s->x51311[n];
		s_dot->x51200[n] = p->ar[n]*s->x01000[n]*s->x50200[n]+p->agp[n]*x00200*s->x51000[n]-p->dg[n]*s->x51200[n]-p->dr[n]*s->x51200[n]-p->nl[n]*s->x51200[n]-p->ac[n]*x00100*s->x51200[n]+p->ne[n]*s->x51210[n]+p->dc[n]*s->x51300[n];
		s_dot->x51210[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51010[n]+p->nl[n]*s->x51200[n]-p->dg[n]*s->x51210[n]-p->dr[n]*s->x51210[n]-p->ne[n]*s->x51210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51210[n]+p->unbbin[n]*s->x51211[n]+p->dc[n]*s->x51310[n];
		s_dot->x51211[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x50210[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51210[n]-p->dg[n]*s->x51211[n]-2*p->dr[n]*s->x51211[n]-p->unbbin[n]*s->x51211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51211[n]+p->dc[n]*s->x51311[n];
		s_dot->x51300[n] = p->ar[n]*s->x01000[n]*s->x50300[n]+p->agp[n]*x00200*s->x51100[n]+p->ac[n]*x00100*s->x51200[n]-p->dc[n]*s->x51300[n]-p->dg[n]*s->x51300[n]-p->dr[n]*s->x51300[n]-p->nl[n]*s->x51300[n]+p->ne[n]*s->x51310[n];
		s_dot->x51310[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51210[n]+p->nl[n]*s->x51300[n]-p->dc[n]*s->x51310[n]-p->dg[n]*s->x51310[n]-p->dr[n]*s->x51310[n]-p->ne[n]*s->x51310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51310[n]+p->unbbin[n]*s->x51311[n];
		s_dot->x51311[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x50310[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x50311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x51111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x51211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x51310[n]-p->dc[n]*s->x51311[n]-p->dg[n]*s->x51311[n]-2*p->dr[n]*s->x51311[n]-p->unbbin[n]*s->x51311[n];
		s_dot->x52000[n] = p->ar[n]*s->x02000[n]*s->x50000[n]-p->dr[n]*s->x52000[n]-p->nl[n]*s->x52000[n]-p->ac[n]*x00100*s->x52000[n]-p->agp[n]*x00200*s->x52000[n]+p->ne[n]*s->x52010[n]+p->dc[n]*s->x52100[n]+p->dg[n]*s->x52200[n];
		s_dot->x52010[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50010[n]+p->nl[n]*s->x52000[n]-p->dr[n]*s->x52010[n]-p->ne[n]*s->x52010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52010[n]+p->unbbin[n]*s->x52011[n]+p->dc[n]*s->x52110[n]+p->dg[n]*s->x52210[n];
		s_dot->x52011[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x50010[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52010[n]-2*p->dr[n]*s->x52011[n]-p->unbbin[n]*s->x52011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52011[n]+p->dc[n]*s->x52111[n]+p->dg[n]*s->x52211[n];
		s_dot->x52100[n] = p->ar[n]*s->x02000[n]*s->x50100[n]+p->ac[n]*x00100*s->x52000[n]-p->dc[n]*s->x52100[n]-p->dr[n]*s->x52100[n]-p->nl[n]*s->x52100[n]-p->agp[n]*x00200*s->x52100[n]+p->ne[n]*s->x52110[n]+p->dg[n]*s->x52300[n];
		s_dot->x52110[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52010[n]+p->nl[n]*s->x52100[n]-p->dc[n]*s->x52110[n]-p->dr[n]*s->x52110[n]-p->ne[n]*s->x52110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52110[n]+p->unbbin[n]*s->x52111[n]+p->dg[n]*s->x52310[n];
		s_dot->x52111[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x50110[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52110[n]-p->dc[n]*s->x52111[n]-2*p->dr[n]*s->x52111[n]-p->unbbin[n]*s->x52111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52111[n]+p->dg[n]*s->x52311[n];
		s_dot->x52200[n] = p->ar[n]*s->x02000[n]*s->x50200[n]+p->agp[n]*x00200*s->x52000[n]-p->dg[n]*s->x52200[n]-p->dr[n]*s->x52200[n]-p->nl[n]*s->x52200[n]-p->ac[n]*x00100*s->x52200[n]+p->ne[n]*s->x52210[n]+p->dc[n]*s->x52300[n];
		s_dot->x52210[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52010[n]+p->nl[n]*s->x52200[n]-p->dg[n]*s->x52210[n]-p->dr[n]*s->x52210[n]-p->ne[n]*s->x52210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52210[n]+p->unbbin[n]*s->x52211[n]+p->dc[n]*s->x52310[n];
		s_dot->x52211[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x50210[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52210[n]-p->dg[n]*s->x52211[n]-2*p->dr[n]*s->x52211[n]-p->unbbin[n]*s->x52211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52211[n]+p->dc[n]*s->x52311[n];
		s_dot->x52300[n] = p->ar[n]*s->x02000[n]*s->x50300[n]+p->agp[n]*x00200*s->x52100[n]+p->ac[n]*x00100*s->x52200[n]-p->dc[n]*s->x52300[n]-p->dg[n]*s->x52300[n]-p->dr[n]*s->x52300[n]-p->nl[n]*s->x52300[n]+p->ne[n]*s->x52310[n];
		s_dot->x52310[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52210[n]+p->nl[n]*s->x52300[n]-p->dc[n]*s->x52310[n]-p->dg[n]*s->x52310[n]-p->dr[n]*s->x52310[n]-p->ne[n]*s->x52310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52310[n]+p->unbbin[n]*s->x52311[n];
		s_dot->x52311[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x50310[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x50311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x52111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x52211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x52310[n]-p->dc[n]*s->x52311[n]-p->dg[n]*s->x52311[n]-2*p->dr[n]*s->x52311[n]-p->unbbin[n]*s->x52311[n];
		s_dot->x60000[n] = -(p->nl[n]*s->x60000[n])-p->up[n]*s->x60000[n]-p->ac[n]*x00100*s->x60000[n]-p->agp[n]*x00200*s->x60000[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x60000[n]+p->ne[n]*s->x60010[n]+p->dc[n]*s->x60100[n]+p->dg[n]*s->x60200[n]+p->dr[n]*(s->x61000[n]+s->x62000[n]);
		s_dot->x60010[n] = p->nl[n]*s->x60000[n]-p->ne[n]*s->x60010[n]-p->up[n]*s->x60010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60010[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60010[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x60010[n]+p->ubc[n]*s->x60011[n]+p->unbbin[n]*s->x60011[n]+p->dc[n]*s->x60110[n]+p->dg[n]*s->x60210[n]+p->dr[n]*(s->x61010[n]+s->x62010[n])+p->dr[n]*(s->x61011[n]+s->x62011[n]);
		s_dot->x60011[n] = p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60010[n]-p->ubc[n]*s->x60011[n]-p->unbbin[n]*s->x60011[n]-p->up[n]*s->x60011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60011[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60011[n]+p->dc[n]*s->x60111[n]+p->dg[n]*s->x60211[n]+p->dr[n]*(s->x61011[n]+s->x62011[n]);
		s_dot->x60100[n] = p->hto[n]*s->x50100[n]+p->ac[n]*x00100*s->x60000[n]-p->dc[n]*s->x60100[n]-p->nl[n]*s->x60100[n]-p->up[n]*s->x60100[n]-p->agp[n]*x00200*s->x60100[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x60100[n]+p->ne[n]*s->x60110[n]+p->dg[n]*s->x60300[n]+p->dr[n]*(s->x61100[n]+s->x62100[n]);
		s_dot->x60110[n] = p->hto[n]*s->x50110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60010[n]+p->nl[n]*s->x60100[n]-p->dc[n]*s->x60110[n]-p->ne[n]*s->x60110[n]-p->up[n]*s->x60110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60110[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60110[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x60110[n]+p->ubc[n]*s->x60111[n]+p->unbbin[n]*s->x60111[n]+p->dg[n]*s->x60310[n]+p->dr[n]*(s->x61110[n]+s->x62110[n])+p->dr[n]*(s->x61111[n]+s->x62111[n]);
		s_dot->x60111[n] = p->hto[n]*s->x50111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60110[n]-p->dc[n]*s->x60111[n]-p->ubc[n]*s->x60111[n]-p->unbbin[n]*s->x60111[n]-p->up[n]*s->x60111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60111[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60111[n]+p->dg[n]*s->x60311[n]+p->dr[n]*(s->x61111[n]+s->x62111[n]);
		s_dot->x60200[n] = s->gto[n]*s->x40200[n]+p->agp[n]*x00200*s->x60000[n]-p->dg[n]*s->x60200[n]-p->nl[n]*s->x60200[n]-p->up[n]*s->x60200[n]-p->ac[n]*x00100*s->x60200[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x60200[n]+p->ne[n]*s->x60210[n]+p->dc[n]*s->x60300[n]+p->dr[n]*(s->x61200[n]+s->x62200[n]);
		s_dot->x60210[n] = s->gto[n]*s->x40210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60010[n]+p->nl[n]*s->x60200[n]-p->dg[n]*s->x60210[n]-p->ne[n]*s->x60210[n]-p->up[n]*s->x60210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60210[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60210[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x60210[n]+p->ubc[n]*s->x60211[n]+p->unbbin[n]*s->x60211[n]+p->dc[n]*s->x60310[n]+p->dr[n]*(s->x61210[n]+s->x62210[n])+p->dr[n]*(s->x61211[n]+s->x62211[n]);
		s_dot->x60211[n] = s->gto[n]*s->x40211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60210[n]-p->dg[n]*s->x60211[n]-p->ubc[n]*s->x60211[n]-p->unbbin[n]*s->x60211[n]-p->up[n]*s->x60211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60211[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60211[n]+p->dc[n]*s->x60311[n]+p->dr[n]*(s->x61211[n]+s->x62211[n]);
		s_dot->x60300[n] = s->gto[n]*s->x40300[n]+p->hto[n]*s->x50300[n]+p->agp[n]*x00200*s->x60100[n]+p->ac[n]*x00100*s->x60200[n]-p->dc[n]*s->x60300[n]-p->dg[n]*s->x60300[n]-p->nl[n]*s->x60300[n]-p->up[n]*s->x60300[n]-p->ar[n]*(s->x01000[n]+s->x02000[n])*s->x60300[n]+p->ne[n]*s->x60310[n]+p->dr[n]*(s->x61300[n]+s->x62300[n]);
		s_dot->x60310[n] = s->gto[n]*s->x40310[n]+p->hto[n]*s->x50310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60210[n]+p->nl[n]*s->x60300[n]-p->dc[n]*s->x60310[n]-p->dg[n]*s->x60310[n]-p->ne[n]*s->x60310[n]-p->up[n]*s->x60310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60310[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60310[n]-p->ar[n]*p->Nf[n]*(s->x01011[n]+s->x02011[n])*s->x60310[n]+p->ubc[n]*s->x60311[n]+p->unbbin[n]*s->x60311[n]+p->dr[n]*(s->x61310[n]+s->x62310[n])+p->dr[n]*(s->x61311[n]+s->x62311[n]);
		s_dot->x60311[n] = s->gto[n]*s->x40311[n]+p->hto[n]*s->x50311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x60111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x60211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x60310[n]-p->dc[n]*s->x60311[n]-p->dg[n]*s->x60311[n]-p->ubc[n]*s->x60311[n]-p->unbbin[n]*s->x60311[n]-p->up[n]*s->x60311[n]-p->ar[n]*p->Nf[n]*(s->x01010[n]+s->x02010[n])*s->x60311[n]+p->dr[n]*(s->x61311[n]+s->x62311[n]);
		s_dot->x61000[n] = p->ar[n]*s->x01000[n]*s->x60000[n]-p->dr[n]*s->x61000[n]-p->nl[n]*s->x61000[n]-p->ac[n]*x00100*s->x61000[n]-p->agp[n]*x00200*s->x61000[n]+p->ne[n]*s->x61010[n]+p->dc[n]*s->x61100[n]+p->dg[n]*s->x61200[n];
		s_dot->x61010[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60010[n]+p->nl[n]*s->x61000[n]-p->dr[n]*s->x61010[n]-p->ne[n]*s->x61010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61010[n]+p->unbbin[n]*s->x61011[n]+p->dc[n]*s->x61110[n]+p->dg[n]*s->x61210[n];
		s_dot->x61011[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x60010[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61010[n]-2*p->dr[n]*s->x61011[n]-p->unbbin[n]*s->x61011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61011[n]+p->dc[n]*s->x61111[n]+p->dg[n]*s->x61211[n];
		s_dot->x61100[n] = p->ar[n]*s->x01000[n]*s->x60100[n]+p->ac[n]*x00100*s->x61000[n]-p->dc[n]*s->x61100[n]-p->dr[n]*s->x61100[n]-p->nl[n]*s->x61100[n]-p->agp[n]*x00200*s->x61100[n]+p->ne[n]*s->x61110[n]+p->dg[n]*s->x61300[n];
		s_dot->x61110[n] = p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61010[n]+p->nl[n]*s->x61100[n]-p->dc[n]*s->x61110[n]-p->dr[n]*s->x61110[n]-p->ne[n]*s->x61110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61110[n]+p->unbbin[n]*s->x61111[n]+p->dg[n]*s->x61310[n];
		s_dot->x61111[n] = p->ar[n]*p->Nf[n]*s->x01011[n]*s->x60110[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61110[n]-p->dc[n]*s->x61111[n]-2*p->dr[n]*s->x61111[n]-p->unbbin[n]*s->x61111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61111[n]+p->dg[n]*s->x61311[n];
		s_dot->x61200[n] = s->gto[n]*s->x41200[n]+p->ar[n]*s->x01000[n]*s->x60200[n]+p->agp[n]*x00200*s->x61000[n]-p->dg[n]*s->x61200[n]-p->dr[n]*s->x61200[n]-p->nl[n]*s->x61200[n]-p->ac[n]*x00100*s->x61200[n]+p->ne[n]*s->x61210[n]+p->dc[n]*s->x61300[n];
		s_dot->x61210[n] = s->gto[n]*s->x41210[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61010[n]+p->nl[n]*s->x61200[n]-p->dg[n]*s->x61210[n]-p->dr[n]*s->x61210[n]-p->ne[n]*s->x61210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61210[n]+p->unbbin[n]*s->x61211[n]+p->dc[n]*s->x61310[n];
		s_dot->x61211[n] = s->gto[n]*s->x41211[n]+p->ar[n]*p->Nf[n]*s->x01011[n]*s->x60210[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61210[n]-p->dg[n]*s->x61211[n]-2*p->dr[n]*s->x61211[n]-p->unbbin[n]*s->x61211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61211[n]+p->dc[n]*s->x61311[n];
		s_dot->x61300[n] = s->gto[n]*s->x41300[n]+p->ar[n]*s->x01000[n]*s->x60300[n]+p->agp[n]*x00200*s->x61100[n]+p->ac[n]*x00100*s->x61200[n]-p->dc[n]*s->x61300[n]-p->dg[n]*s->x61300[n]-p->dr[n]*s->x61300[n]-p->nl[n]*s->x61300[n]+p->ne[n]*s->x61310[n];
		s_dot->x61310[n] = s->gto[n]*s->x41310[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61210[n]+p->nl[n]*s->x61300[n]-p->dc[n]*s->x61310[n]-p->dg[n]*s->x61310[n]-p->dr[n]*s->x61310[n]-p->ne[n]*s->x61310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61310[n]+p->unbbin[n]*s->x61311[n];
		s_dot->x61311[n] = s->gto[n]*s->x41311[n]+p->ar[n]*p->Nf[n]*s->x01011[n]*s->x60310[n]+p->ar[n]*p->Nf[n]*s->x01010[n]*s->x60311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x61111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x61211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x61310[n]-p->dc[n]*s->x61311[n]-p->dg[n]*s->x61311[n]-2*p->dr[n]*s->x61311[n]-p->unbbin[n]*s->x61311[n];
		s_dot->x62000[n] = p->ar[n]*s->x02000[n]*s->x60000[n]-p->dr[n]*s->x62000[n]-p->nl[n]*s->x62000[n]-p->ac[n]*x00100*s->x62000[n]-p->agp[n]*x00200*s->x62000[n]+p->ne[n]*s->x62010[n]+p->dc[n]*s->x62100[n]+p->dg[n]*s->x62200[n];
		s_dot->x62010[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60010[n]+p->nl[n]*s->x62000[n]-p->dr[n]*s->x62010[n]-p->ne[n]*s->x62010[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62010[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62010[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62010[n]+p->unbbin[n]*s->x62011[n]+p->dc[n]*s->x62110[n]+p->dg[n]*s->x62210[n];
		s_dot->x62011[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x60010[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62010[n]-2*p->dr[n]*s->x62011[n]-p->unbbin[n]*s->x62011[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62011[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62011[n]+p->dc[n]*s->x62111[n]+p->dg[n]*s->x62211[n];
		s_dot->x62100[n] = p->ar[n]*s->x02000[n]*s->x60100[n]+p->ac[n]*x00100*s->x62000[n]-p->dc[n]*s->x62100[n]-p->dr[n]*s->x62100[n]-p->nl[n]*s->x62100[n]-p->agp[n]*x00200*s->x62100[n]+p->ne[n]*s->x62110[n]+p->dg[n]*s->x62300[n];
		s_dot->x62110[n] = p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62010[n]+p->nl[n]*s->x62100[n]-p->dc[n]*s->x62110[n]-p->dr[n]*s->x62110[n]-p->ne[n]*s->x62110[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62110[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62110[n]+p->unbbin[n]*s->x62111[n]+p->dg[n]*s->x62310[n];
		s_dot->x62111[n] = p->ar[n]*p->Nf[n]*s->x02011[n]*s->x60110[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62110[n]-p->dc[n]*s->x62111[n]-2*p->dr[n]*s->x62111[n]-p->unbbin[n]*s->x62111[n]-p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62111[n]+p->dg[n]*s->x62311[n];
		s_dot->x62200[n] = s->gto[n]*s->x42200[n]+p->ar[n]*s->x02000[n]*s->x60200[n]+p->agp[n]*x00200*s->x62000[n]-p->dg[n]*s->x62200[n]-p->dr[n]*s->x62200[n]-p->nl[n]*s->x62200[n]-p->ac[n]*x00100*s->x62200[n]+p->ne[n]*s->x62210[n]+p->dc[n]*s->x62300[n];
		s_dot->x62210[n] = s->gto[n]*s->x42210[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60210[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62010[n]+p->nl[n]*s->x62200[n]-p->dg[n]*s->x62210[n]-p->dr[n]*s->x62210[n]-p->ne[n]*s->x62210[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62210[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62210[n]+p->unbbin[n]*s->x62211[n]+p->dc[n]*s->x62310[n];
		s_dot->x62211[n] = s->gto[n]*s->x42211[n]+p->ar[n]*p->Nf[n]*s->x02011[n]*s->x60210[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60211[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62011[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62210[n]-p->dg[n]*s->x62211[n]-2*p->dr[n]*s->x62211[n]-p->unbbin[n]*s->x62211[n]-p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62211[n]+p->dc[n]*s->x62311[n];
		s_dot->x62300[n] = s->gto[n]*s->x42300[n]+p->ar[n]*s->x02000[n]*s->x60300[n]+p->agp[n]*x00200*s->x62100[n]+p->ac[n]*x00100*s->x62200[n]-p->dc[n]*s->x62300[n]-p->dg[n]*s->x62300[n]-p->dr[n]*s->x62300[n]-p->nl[n]*s->x62300[n]+p->ne[n]*s->x62310[n];
		s_dot->x62310[n] = s->gto[n]*s->x42310[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60310[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62110[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62210[n]+p->nl[n]*s->x62300[n]-p->dc[n]*s->x62310[n]-p->dg[n]*s->x62310[n]-p->dr[n]*s->x62310[n]-p->ne[n]*s->x62310[n]-p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62310[n]+p->unbbin[n]*s->x62311[n];
		s_dot->x62311[n] = s->gto[n]*s->x42311[n]+p->ar[n]*p->Nf[n]*s->x02011[n]*s->x60310[n]+p->ar[n]*p->Nf[n]*s->x02010[n]*s->x60311[n]+p->agp[n]*p->Nf[n]*s->x00210[n]*s->x62111[n]+p->ac[n]*p->Nf[n]*s->x00110[n]*s->x62211[n]+p->bbin[n]*p->Nf[n]*s->x00011[n]*s->x62310[n]-p->dc[n]*s->x62311[n]-p->dg[n]*s->x62311[n]-2*p->dr[n]*s->x62311[n]-p->unbbin[n]*s->x62311[n];


		/////////////// Light equation /////////////////////
		s_dot->ltn[n] = 60.0*LIGHT*p->lta[n]*(1.0-s->ltn[n])-p->ltb[n]*s->ltn[n];
/*
		mclock_t frsk = 0.0;
		mclock_t fractpart, intpart;
		fractpart = modf (t/24.0, &intpart);
		mclock_t CT = 24.0*fractpart;
*/
//		if ( (n <= (float)ncells/3.0) && ( CT >=6.0 ) && (CT < 18.0) )
/*		if ( (n <= (float)ncells/3.0) && ( CT >=12.0 ) )
			frsk = 1.0;
		else
			frsk = 0.0;
*/
		mclock_t cscale = 1.0;
		mclock_t vscale = 1.0;
		/////////////// Signall*ing equations ///////////////
		//network coupled
		s_dot->vip[n] = (mclock_t)(CPL)*6000.0*p->vpr[n]*input[n]/upstream[n]-p->uv[n]*s->vip[n]-p->vbin[n]*(V00+s->V01[n]+s->V02[n])*s->vip[n]+p->unvbin[n]*(s->V10[n]+s->V11[n]+s->V12[n]);
		//self coupled
//		s_dot->vip[n] = 6000.0*p->vpr[n]*x->cac[n]-p->uv[n]*s->vip[n]-p->vbin[n]*(V00+s->V01[n]+s->V02[n])*s->vip[n]+p->unvbin[n]*(s->V10[n]+s->V11[n]+s->V12[n]);
		s_dot->V10[n] = p->vbin[n]*V00*s->vip[n]-p->unvbin[n]*s->V10[n]-p->cvbin[n]*s->V10[n]*(s->x01000[n]+s->x02000[n])+p->uncvbin[n]*(s->V11[n]+s->V12[n]);
		s_dot->V11[n] = p->cvbin[n]*s->V10[n]*s->x01000[n]-(p->unvbin[n]+p->uncvbin[n])*s->V11[n]+p->vbin[n]*s->V01[n]*s->vip[n];
		s_dot->V12[n] = p->cvbin[n]*s->V10[n]*s->x02000[n]-(p->unvbin[n]+p->uncvbin[n])*s->V12[n]+p->vbin[n]*s->V02[n]*s->vip[n];
		s_dot->V01[n] = p->cvbin[n]*V00*s->x01000[n]-p->uncvbin[n]*s->V01[n]-p->vbin[n]*s->V01[n]*s->vip[n]+p->unvbin[n]*s->V11[n];
		s_dot->V02[n] = p->cvbin[n]*V00*s->x02000[n]-p->uncvbin[n]*s->V02[n]-p->vbin[n]*s->V02[n]*s->vip[n]+p->unvbin[n]*s->V12[n];
		s_dot->cAMP[n] = vscale*p->vs[n]*s->V10[n]-p->us[n]*s->cAMP[n];
		s_dot->CREB[n] = cscale*(1.0+EPHYS*6499.0)*x->cac[n] + vscale*p->vs[n]*s->cAMP[n]-p->us[n]*s->CREB[n];
		s_dot->CRE[n] = p->sbin[n]*s->CREB[n]*(1.0-s->CRE[n])-p->unsbin[n]*s->CRE[n];
	}
}

__global__ void lincomb(mclock_t c, Mstate *s, Mstate *s1, Mstate *s2)
{
	// creates new struct: s = s1+c*s2
	int n;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;
	for (n=tid; n<ncells; n+=stride) {
		s->GR[n] = s1->GR[n]+c*s2->GR[n];
		s->G[n] = s1->G[n]+c*s2->G[n];
		s->GrR[n] = s1->GrR[n]+c*s2->GrR[n];
		s->Gr[n] = s1->Gr[n]+c*s2->Gr[n];
		s->GcR[n] = s1->GcR[n]+c*s2->GcR[n];
		s->Gc[n] = s1->Gc[n]+c*s2->Gc[n];
//		s->GBR[n] = s1->GBR[n]+c*s2->GBR[n];
		s->GB[n] = s1->GB[n]+c*s2->GB[n];
//		s->GBRb[n] = s1->GBRb[n]+c*s2->GBRb[n];
		s->GBb[n] = s1->GBb[n]+c*s2->GBb[n];
		s->MnPo[n] = s1->MnPo[n]+c*s2->MnPo[n];
		s->McPo[n] = s1->McPo[n]+c*s2->McPo[n];
		s->MnPt[n] = s1->MnPt[n]+c*s2->MnPt[n];
		s->McPt[n] = s1->McPt[n]+c*s2->McPt[n];
		s->MnRt[n] = s1->MnRt[n]+c*s2->MnRt[n];
		s->McRt[n] = s1->McRt[n]+c*s2->McRt[n];
		s->MnRev[n] = s1->MnRev[n]+c*s2->MnRev[n];
		s->McRev[n] = s1->McRev[n]+c*s2->McRev[n];
		s->MnRo[n] = s1->MnRo[n]+c*s2->MnRo[n];
		s->McRo[n] = s1->McRo[n]+c*s2->McRo[n];
		s->MnB[n] = s1->MnB[n]+c*s2->MnB[n];
		s->McB[n] = s1->McB[n]+c*s2->McB[n];
		s->MnNp[n] = s1->MnNp[n]+c*s2->MnNp[n];
		s->McNp[n] = s1->McNp[n]+c*s2->McNp[n];
		s->B[n] = s1->B[n]+c*s2->B[n];
		s->Cl[n] = s1->Cl[n]+c*s2->Cl[n];
		s->BC[n] = s1->BC[n]+c*s2->BC[n];
		s->cyrev[n] = s1->cyrev[n]+c*s2->cyrev[n];
		s->revn[n] = s1->revn[n]+c*s2->revn[n];
		s->cyrevg[n] = s1->cyrevg[n]+c*s2->cyrevg[n];
		s->revng[n] = s1->revng[n]+c*s2->revng[n];
		s->cyrevgp[n] = s1->cyrevgp[n]+c*s2->cyrevgp[n];
		s->revngp[n] = s1->revngp[n]+c*s2->revngp[n];
		s->cyrevp[n] = s1->cyrevp[n]+c*s2->cyrevp[n];
		s->revnp[n] = s1->revnp[n]+c*s2->revnp[n];
		s->gto[n] = s1->gto[n]+c*s2->gto[n];
		s->x00001[n] = s1->x00001[n]+c*s2->x00001[n];
		s->x00011[n] = s1->x00011[n]+c*s2->x00011[n];
//		s->x00100[n] = s1->x00100[n]+c*s2->x00100[n];
		s->x00110[n] = s1->x00110[n]+c*s2->x00110[n];
//		s->x00200[n] = s1->x00200[n]+c*s2->x00200[n];
		s->x00210[n] = s1->x00210[n]+c*s2->x00210[n];
		s->x01000[n] = s1->x01000[n]+c*s2->x01000[n];
		s->x01010[n] = s1->x01010[n]+c*s2->x01010[n];
		s->x01011[n] = s1->x01011[n]+c*s2->x01011[n];
		s->x02000[n] = s1->x02000[n]+c*s2->x02000[n];
		s->x02010[n] = s1->x02010[n]+c*s2->x02010[n];
		s->x02011[n] = s1->x02011[n]+c*s2->x02011[n];
		s->x10000[n] = s1->x10000[n]+c*s2->x10000[n];
		s->x10100[n] = s1->x10100[n]+c*s2->x10100[n];
		s->x20000[n] = s1->x20000[n]+c*s2->x20000[n];
		s->x20010[n] = s1->x20010[n]+c*s2->x20010[n];
		s->x20011[n] = s1->x20011[n]+c*s2->x20011[n];
		s->x20100[n] = s1->x20100[n]+c*s2->x20100[n];
		s->x20110[n] = s1->x20110[n]+c*s2->x20110[n];
		s->x20111[n] = s1->x20111[n]+c*s2->x20111[n];
		s->x21000[n] = s1->x21000[n]+c*s2->x21000[n];
		s->x21010[n] = s1->x21010[n]+c*s2->x21010[n];
		s->x21011[n] = s1->x21011[n]+c*s2->x21011[n];
		s->x21100[n] = s1->x21100[n]+c*s2->x21100[n];
		s->x21110[n] = s1->x21110[n]+c*s2->x21110[n];
		s->x21111[n] = s1->x21111[n]+c*s2->x21111[n];
		s->x22000[n] = s1->x22000[n]+c*s2->x22000[n];
		s->x22010[n] = s1->x22010[n]+c*s2->x22010[n];
		s->x22011[n] = s1->x22011[n]+c*s2->x22011[n];
		s->x22100[n] = s1->x22100[n]+c*s2->x22100[n];
		s->x22110[n] = s1->x22110[n]+c*s2->x22110[n];
		s->x22111[n] = s1->x22111[n]+c*s2->x22111[n];
		s->x30000[n] = s1->x30000[n]+c*s2->x30000[n];
		s->x30100[n] = s1->x30100[n]+c*s2->x30100[n];
		s->x30200[n] = s1->x30200[n]+c*s2->x30200[n];
		s->x30300[n] = s1->x30300[n]+c*s2->x30300[n];
		s->x40000[n] = s1->x40000[n]+c*s2->x40000[n];
		s->x40010[n] = s1->x40010[n]+c*s2->x40010[n];
		s->x40011[n] = s1->x40011[n]+c*s2->x40011[n];
		s->x40100[n] = s1->x40100[n]+c*s2->x40100[n];
		s->x40110[n] = s1->x40110[n]+c*s2->x40110[n];
		s->x40111[n] = s1->x40111[n]+c*s2->x40111[n];
		s->x40200[n] = s1->x40200[n]+c*s2->x40200[n];
		s->x40210[n] = s1->x40210[n]+c*s2->x40210[n];
		s->x40211[n] = s1->x40211[n]+c*s2->x40211[n];
		s->x40300[n] = s1->x40300[n]+c*s2->x40300[n];
		s->x40310[n] = s1->x40310[n]+c*s2->x40310[n];
		s->x40311[n] = s1->x40311[n]+c*s2->x40311[n];
		s->x41000[n] = s1->x41000[n]+c*s2->x41000[n];
		s->x41010[n] = s1->x41010[n]+c*s2->x41010[n];
		s->x41011[n] = s1->x41011[n]+c*s2->x41011[n];
		s->x41100[n] = s1->x41100[n]+c*s2->x41100[n];
		s->x41110[n] = s1->x41110[n]+c*s2->x41110[n];
		s->x41111[n] = s1->x41111[n]+c*s2->x41111[n];
		s->x41200[n] = s1->x41200[n]+c*s2->x41200[n];
		s->x41210[n] = s1->x41210[n]+c*s2->x41210[n];
		s->x41211[n] = s1->x41211[n]+c*s2->x41211[n];
		s->x41300[n] = s1->x41300[n]+c*s2->x41300[n];
		s->x41310[n] = s1->x41310[n]+c*s2->x41310[n];
		s->x41311[n] = s1->x41311[n]+c*s2->x41311[n];
		s->x42000[n] = s1->x42000[n]+c*s2->x42000[n];
		s->x42010[n] = s1->x42010[n]+c*s2->x42010[n];
		s->x42011[n] = s1->x42011[n]+c*s2->x42011[n];
		s->x42100[n] = s1->x42100[n]+c*s2->x42100[n];
		s->x42110[n] = s1->x42110[n]+c*s2->x42110[n];
		s->x42111[n] = s1->x42111[n]+c*s2->x42111[n];
		s->x42200[n] = s1->x42200[n]+c*s2->x42200[n];
		s->x42210[n] = s1->x42210[n]+c*s2->x42210[n];
		s->x42211[n] = s1->x42211[n]+c*s2->x42211[n];
		s->x42300[n] = s1->x42300[n]+c*s2->x42300[n];
		s->x42310[n] = s1->x42310[n]+c*s2->x42310[n];
		s->x42311[n] = s1->x42311[n]+c*s2->x42311[n];
		s->x50000[n] = s1->x50000[n]+c*s2->x50000[n];
		s->x50010[n] = s1->x50010[n]+c*s2->x50010[n];
		s->x50011[n] = s1->x50011[n]+c*s2->x50011[n];
		s->x50100[n] = s1->x50100[n]+c*s2->x50100[n];
		s->x50110[n] = s1->x50110[n]+c*s2->x50110[n];
		s->x50111[n] = s1->x50111[n]+c*s2->x50111[n];
		s->x50200[n] = s1->x50200[n]+c*s2->x50200[n];
		s->x50210[n] = s1->x50210[n]+c*s2->x50210[n];
		s->x50211[n] = s1->x50211[n]+c*s2->x50211[n];
		s->x50300[n] = s1->x50300[n]+c*s2->x50300[n];
		s->x50310[n] = s1->x50310[n]+c*s2->x50310[n];
		s->x50311[n] = s1->x50311[n]+c*s2->x50311[n];
		s->x51000[n] = s1->x51000[n]+c*s2->x51000[n];
		s->x51010[n] = s1->x51010[n]+c*s2->x51010[n];
		s->x51011[n] = s1->x51011[n]+c*s2->x51011[n];
		s->x51100[n] = s1->x51100[n]+c*s2->x51100[n];
		s->x51110[n] = s1->x51110[n]+c*s2->x51110[n];
		s->x51111[n] = s1->x51111[n]+c*s2->x51111[n];
		s->x51200[n] = s1->x51200[n]+c*s2->x51200[n];
		s->x51210[n] = s1->x51210[n]+c*s2->x51210[n];
		s->x51211[n] = s1->x51211[n]+c*s2->x51211[n];
		s->x51300[n] = s1->x51300[n]+c*s2->x51300[n];
		s->x51310[n] = s1->x51310[n]+c*s2->x51310[n];
		s->x51311[n] = s1->x51311[n]+c*s2->x51311[n];
		s->x52000[n] = s1->x52000[n]+c*s2->x52000[n];
		s->x52010[n] = s1->x52010[n]+c*s2->x52010[n];
		s->x52011[n] = s1->x52011[n]+c*s2->x52011[n];
		s->x52100[n] = s1->x52100[n]+c*s2->x52100[n];
		s->x52110[n] = s1->x52110[n]+c*s2->x52110[n];
		s->x52111[n] = s1->x52111[n]+c*s2->x52111[n];
		s->x52200[n] = s1->x52200[n]+c*s2->x52200[n];
		s->x52210[n] = s1->x52210[n]+c*s2->x52210[n];
		s->x52211[n] = s1->x52211[n]+c*s2->x52211[n];
		s->x52300[n] = s1->x52300[n]+c*s2->x52300[n];
		s->x52310[n] = s1->x52310[n]+c*s2->x52310[n];
		s->x52311[n] = s1->x52311[n]+c*s2->x52311[n];
		s->x60000[n] = s1->x60000[n]+c*s2->x60000[n];
		s->x60010[n] = s1->x60010[n]+c*s2->x60010[n];
		s->x60011[n] = s1->x60011[n]+c*s2->x60011[n];
		s->x60100[n] = s1->x60100[n]+c*s2->x60100[n];
		s->x60110[n] = s1->x60110[n]+c*s2->x60110[n];
		s->x60111[n] = s1->x60111[n]+c*s2->x60111[n];
		s->x60200[n] = s1->x60200[n]+c*s2->x60200[n];
		s->x60210[n] = s1->x60210[n]+c*s2->x60210[n];
		s->x60211[n] = s1->x60211[n]+c*s2->x60211[n];
		s->x60300[n] = s1->x60300[n]+c*s2->x60300[n];
		s->x60310[n] = s1->x60310[n]+c*s2->x60310[n];
		s->x60311[n] = s1->x60311[n]+c*s2->x60311[n];
		s->x61000[n] = s1->x61000[n]+c*s2->x61000[n];
		s->x61010[n] = s1->x61010[n]+c*s2->x61010[n];
		s->x61011[n] = s1->x61011[n]+c*s2->x61011[n];
		s->x61100[n] = s1->x61100[n]+c*s2->x61100[n];
		s->x61110[n] = s1->x61110[n]+c*s2->x61110[n];
		s->x61111[n] = s1->x61111[n]+c*s2->x61111[n];
		s->x61200[n] = s1->x61200[n]+c*s2->x61200[n];
		s->x61210[n] = s1->x61210[n]+c*s2->x61210[n];
		s->x61211[n] = s1->x61211[n]+c*s2->x61211[n];
		s->x61300[n] = s1->x61300[n]+c*s2->x61300[n];
		s->x61310[n] = s1->x61310[n]+c*s2->x61310[n];
		s->x61311[n] = s1->x61311[n]+c*s2->x61311[n];
		s->x62000[n] = s1->x62000[n]+c*s2->x62000[n];
		s->x62010[n] = s1->x62010[n]+c*s2->x62010[n];
		s->x62011[n] = s1->x62011[n]+c*s2->x62011[n];
		s->x62100[n] = s1->x62100[n]+c*s2->x62100[n];
		s->x62110[n] = s1->x62110[n]+c*s2->x62110[n];
		s->x62111[n] = s1->x62111[n]+c*s2->x62111[n];
		s->x62200[n] = s1->x62200[n]+c*s2->x62200[n];
		s->x62210[n] = s1->x62210[n]+c*s2->x62210[n];
		s->x62211[n] = s1->x62211[n]+c*s2->x62211[n];
		s->x62300[n] = s1->x62300[n]+c*s2->x62300[n];
		s->x62310[n] = s1->x62310[n]+c*s2->x62310[n];
		s->x62311[n] = s1->x62311[n]+c*s2->x62311[n];
		s->ltn[n] = s1->ltn[n]+c*s2->ltn[n];
		s->vip[n] = s1->vip[n]+c*s2->vip[n];
		s->V10[n] = s1->V10[n]+c*s2->V10[n];
		s->V11[n] = s1->V11[n]+c*s2->V11[n];
		s->V12[n] = s1->V12[n]+c*s2->V12[n];
		s->V01[n] = s1->V01[n]+c*s2->V01[n];
		s->V02[n] = s1->V02[n]+c*s2->V02[n];
		s->cAMP[n] = s1->cAMP[n]+c*s2->cAMP[n];
		s->CREB[n] = s1->CREB[n]+c*s2->CREB[n];
		s->CRE[n] = s1->CRE[n]+c*s2->CRE[n];
	}
}

__global__ void rk4(Mstate *s, Mstate *k1, Mstate *k2, Mstate *k3, Mstate *k4, double t, mclock_t idx) 
{
	// Calculate state at next time step using RK4
	int n;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;

	for (n=tid; n<ncells; n+=stride) {
	/* Used for creating PRC */
	// add stim to rhs of whichever state variable you want to add stimulus to //
	       /* mclock_t stim = 0;
//                mclock_t offset = idx*1.0;
                mclock_t stim_t = 200;
                if (t == stim_t)
                        stim = 20;
*/
		s->GR[n] = s->GR[n] + 0.16666666667*Mdt*(k1->GR[n]+2*k2->GR[n]+2*k3->GR[n]+k4->GR[n]);
		s->G[n] = s->G[n] + 0.16666666667*Mdt*(k1->G[n]+2*k2->G[n]+2*k3->G[n]+k4->G[n]);
		s->GrR[n] = s->GrR[n] + 0.16666666667*Mdt*(k1->GrR[n]+2*k2->GrR[n]+2*k3->GrR[n]+k4->GrR[n]);
		s->Gr[n] = s->Gr[n] + 0.16666666667*Mdt*(k1->Gr[n]+2*k2->Gr[n]+2*k3->Gr[n]+k4->Gr[n]);
		s->GcR[n] = s->GcR[n] + 0.16666666667*Mdt*(k1->GcR[n]+2*k2->GcR[n]+2*k3->GcR[n]+k4->GcR[n]);
		s->Gc[n] = s->Gc[n] + 0.16666666667*Mdt*(k1->Gc[n]+2*k2->Gc[n]+2*k3->Gc[n]+k4->Gc[n]);
//		s->GBR[n] = s->GBR[n] + 0.16666666667*Mdt*(k1->GBR[n]+2*k2->GBR[n]+2*k3->GBR[n]+k4->GBR[n]);
		s->GB[n] = s->GB[n] + 0.16666666667*Mdt*(k1->GB[n]+2*k2->GB[n]+2*k3->GB[n]+k4->GB[n]);
//		s->GBRb[n] = s->GBRb[n] + 0.16666666667*Mdt*(k1->GBRb[n]+2*k2->GBRb[n]+2*k3->GBRb[n]+k4->GBRb[n]);
		s->GBb[n] = s->GBb[n] + 0.16666666667*Mdt*(k1->GBb[n]+2*k2->GBb[n]+2*k3->GBb[n]+k4->GBb[n]);
		s->MnPo[n] = s->MnPo[n] + 0.16666666667*Mdt*(k1->MnPo[n]+2*k2->MnPo[n]+2*k3->MnPo[n]+k4->MnPo[n]);
		s->McPo[n] = s->McPo[n] + 0.16666666667*Mdt*(k1->McPo[n]+2*k2->McPo[n]+2*k3->McPo[n]+k4->McPo[n]);
		s->MnPt[n] = s->MnPt[n] + .166666666667*Mdt*(k1->MnPt[n]+2*k2->MnPt[n]+2*k3->MnPt[n]+k4->MnPt[n]);
		s->McPt[n] = s->McPt[n] + 0.16666666667*Mdt*(k1->McPt[n]+2*k2->McPt[n]+2*k3->McPt[n]+k4->McPt[n]);
		s->MnRt[n] = s->MnRt[n] + 0.16666666667*Mdt*(k1->MnRt[n]+2*k2->MnRt[n]+2*k3->MnRt[n]+k4->MnRt[n]);
		s->McRt[n] = s->McRt[n] + 0.16666666667*Mdt*(k1->McRt[n]+2*k2->McRt[n]+2*k3->McRt[n]+k4->McRt[n]);
		s->MnRev[n] = s->MnRev[n] + 0.16666666667*Mdt*(k1->MnRev[n]+2*k2->MnRev[n]+2*k3->MnRev[n]+k4->MnRev[n]);
		s->McRev[n] = s->McRev[n] + 0.16666666667*Mdt*(k1->McRev[n]+2*k2->McRev[n]+2*k3->McRev[n]+k4->McRev[n]);
		s->MnRo[n] = s->MnRo[n] + 0.16666666667*Mdt*(k1->MnRo[n]+2*k2->MnRo[n]+2*k3->MnRo[n]+k4->MnRo[n]);
		s->McRo[n] = s->McRo[n] + 0.16666666667*Mdt*(k1->McRo[n]+2*k2->McRo[n]+2*k3->McRo[n]+k4->McRo[n]);
		s->MnB[n] = s->MnB[n] + 0.16666666667*Mdt*(k1->MnB[n]+2*k2->MnB[n]+2*k3->MnB[n]+k4->MnB[n]);
		s->McB[n] = s->McB[n] + 0.16666666667*Mdt*(k1->McB[n]+2*k2->McB[n]+2*k3->McB[n]+k4->McB[n]);
		s->MnNp[n] = s->MnNp[n] + 0.16666666667*Mdt*(k1->MnNp[n]+2*k2->MnNp[n]+2*k3->MnNp[n]+k4->MnNp[n]);
		s->McNp[n] = s->McNp[n] + 0.16666666667*Mdt*(k1->McNp[n]+2*k2->McNp[n]+2*k3->McNp[n]+k4->McNp[n]);
		s->B[n] = s->B[n] + 0.16666666667*Mdt*(k1->B[n]+2*k2->B[n]+2*k3->B[n]+k4->B[n]);
		s->Cl[n] = s->Cl[n] + 0.16666666667*Mdt*(k1->Cl[n]+2*k2->Cl[n]+2*k3->Cl[n]+k4->Cl[n]);
		s->BC[n] = s->BC[n] + 0.16666666667*Mdt*(k1->BC[n]+2*k2->BC[n]+2*k3->BC[n]+k4->BC[n]);
		s->cyrev[n] = s->cyrev[n] + 0.16666666667*Mdt*(k1->cyrev[n]+2*k2->cyrev[n]+2*k3->cyrev[n]+k4->cyrev[n]);
		s->revn[n] = s->revn[n] + 0.16666666667*Mdt*(k1->revn[n]+2*k2->revn[n]+2*k3->revn[n]+k4->revn[n]);
		s->cyrevg[n] = s->cyrevg[n] + 0.16666666667*Mdt*(k1->cyrevg[n]+2*k2->cyrevg[n]+2*k3->cyrevg[n]+k4->cyrevg[n]);
		s->revng[n] = s->revng[n] + 0.16666666667*Mdt*(k1->revng[n]+2*k2->revng[n]+2*k3->revng[n]+k4->revng[n]);
		s->cyrevgp[n] = s->cyrevgp[n] + 0.16666666667*Mdt*(k1->cyrevgp[n]+2*k2->cyrevgp[n]+2*k3->cyrevgp[n]+k4->cyrevgp[n]);
		s->revngp[n] = s->revngp[n] + 0.16666666667*Mdt*(k1->revngp[n]+2*k2->revngp[n]+2*k3->revngp[n]+k4->revngp[n]);
		s->cyrevp[n] = s->cyrevp[n] + 0.16666666667*Mdt*(k1->cyrevp[n]+2*k2->cyrevp[n]+2*k3->cyrevp[n]+k4->cyrevp[n]);
		s->revnp[n] = s->revnp[n] + 0.16666666667*Mdt*(k1->revnp[n]+2*k2->revnp[n]+2*k3->revnp[n]+k4->revnp[n]);
		s->gto[n] = s->gto[n] + 0.16666666667*Mdt*(k1->gto[n]+2*k2->gto[n]+2*k3->gto[n]+k4->gto[n]);
		s->x00001[n] = s->x00001[n] + 0.16666666667*Mdt*(k1->x00001[n]+2*k2->x00001[n]+2*k3->x00001[n]+k4->x00001[n]);
		s->x00011[n] = s->x00011[n] + 0.16666666667*Mdt*(k1->x00011[n]+2*k2->x00011[n]+2*k3->x00011[n]+k4->x00011[n]);
//		s->x00100[n] = s->x00100[n] + 0.16666666667*Mdt*(k1->x00100[n]+2*k2->x00100[n]+2*k3->x00100[n]+k4->x00100[n]);
		s->x00110[n] = s->x00110[n] + 0.16666666667*Mdt*(k1->x00110[n]+2*k2->x00110[n]+2*k3->x00110[n]+k4->x00110[n]);
//		s->x00200[n] = s->x00200[n] + 0.16666666667*Mdt*(k1->x00200[n]+2*k2->x00200[n]+2*k3->x00200[n]+k4->x00200[n]);
		s->x00210[n] = s->x00210[n] + 0.16666666667*Mdt*(k1->x00210[n]+2*k2->x00210[n]+2*k3->x00210[n]+k4->x00210[n]);
		s->x01000[n] = s->x01000[n] + 0.16666666667*Mdt*(k1->x01000[n]+2*k2->x01000[n]+2*k3->x01000[n]+k4->x01000[n]);
		s->x01010[n] = s->x01010[n] + 0.16666666667*Mdt*(k1->x01010[n]+2*k2->x01010[n]+2*k3->x01010[n]+k4->x01010[n]);
		s->x01011[n] = s->x01011[n] + 0.16666666667*Mdt*(k1->x01011[n]+2*k2->x01011[n]+2*k3->x01011[n]+k4->x01011[n]);
		s->x02000[n] = s->x02000[n] + 0.16666666667*Mdt*(k1->x02000[n]+2*k2->x02000[n]+2*k3->x02000[n]+k4->x02000[n]);
		s->x02010[n] = s->x02010[n] + 0.16666666667*Mdt*(k1->x02010[n]+2*k2->x02010[n]+2*k3->x02010[n]+k4->x02010[n]);
		s->x02011[n] = s->x02011[n] + 0.16666666667*Mdt*(k1->x02011[n]+2*k2->x02011[n]+2*k3->x02011[n]+k4->x02011[n]);
		s->x10000[n] = s->x10000[n] + 0.16666666667*Mdt*(k1->x10000[n]+2*k2->x10000[n]+2*k3->x10000[n]+k4->x10000[n]);
		s->x10100[n] = s->x10100[n] + 0.16666666667*Mdt*(k1->x10100[n]+2*k2->x10100[n]+2*k3->x10100[n]+k4->x10100[n]);
		s->x20000[n] = s->x20000[n] + 0.16666666667*Mdt*(k1->x20000[n]+2*k2->x20000[n]+2*k3->x20000[n]+k4->x20000[n]);
		s->x20010[n] = s->x20010[n] + 0.16666666667*Mdt*(k1->x20010[n]+2*k2->x20010[n]+2*k3->x20010[n]+k4->x20010[n]);
		s->x20011[n] = s->x20011[n] + 0.16666666667*Mdt*(k1->x20011[n]+2*k2->x20011[n]+2*k3->x20011[n]+k4->x20011[n]);
		s->x20100[n] = s->x20100[n] + 0.16666666667*Mdt*(k1->x20100[n]+2*k2->x20100[n]+2*k3->x20100[n]+k4->x20100[n]);
		s->x20110[n] = s->x20110[n] + 0.16666666667*Mdt*(k1->x20110[n]+2*k2->x20110[n]+2*k3->x20110[n]+k4->x20110[n]);
		s->x20111[n] = s->x20111[n] + 0.16666666667*Mdt*(k1->x20111[n]+2*k2->x20111[n]+2*k3->x20111[n]+k4->x20111[n]);
		s->x21000[n] = s->x21000[n] + 0.16666666667*Mdt*(k1->x21000[n]+2*k2->x21000[n]+2*k3->x21000[n]+k4->x21000[n]);
		s->x21010[n] = s->x21010[n] + 0.16666666667*Mdt*(k1->x21010[n]+2*k2->x21010[n]+2*k3->x21010[n]+k4->x21010[n]);
		s->x21011[n] = s->x21011[n] + 0.16666666667*Mdt*(k1->x21011[n]+2*k2->x21011[n]+2*k3->x21011[n]+k4->x21011[n]);
		s->x21100[n] = s->x21100[n] + 0.16666666667*Mdt*(k1->x21100[n]+2*k2->x21100[n]+2*k3->x21100[n]+k4->x21100[n]);
		s->x21110[n] = s->x21110[n] + 0.16666666667*Mdt*(k1->x21110[n]+2*k2->x21110[n]+2*k3->x21110[n]+k4->x21110[n]);
		s->x21111[n] = s->x21111[n] + 0.16666666667*Mdt*(k1->x21111[n]+2*k2->x21111[n]+2*k3->x21111[n]+k4->x21111[n]);
		s->x22000[n] = s->x22000[n] + 0.16666666667*Mdt*(k1->x22000[n]+2*k2->x22000[n]+2*k3->x22000[n]+k4->x22000[n]);
		s->x22010[n] = s->x22010[n] + 0.16666666667*Mdt*(k1->x22010[n]+2*k2->x22010[n]+2*k3->x22010[n]+k4->x22010[n]);
		s->x22011[n] = s->x22011[n] + 0.16666666667*Mdt*(k1->x22011[n]+2*k2->x22011[n]+2*k3->x22011[n]+k4->x22011[n]);
		s->x22100[n] = s->x22100[n] + 0.16666666667*Mdt*(k1->x22100[n]+2*k2->x22100[n]+2*k3->x22100[n]+k4->x22100[n]);
		s->x22110[n] = s->x22110[n] + 0.16666666667*Mdt*(k1->x22110[n]+2*k2->x22110[n]+2*k3->x22110[n]+k4->x22110[n]);
		s->x22111[n] = s->x22111[n] + 0.16666666667*Mdt*(k1->x22111[n]+2*k2->x22111[n]+2*k3->x22111[n]+k4->x22111[n]);
		s->x30000[n] = s->x30000[n] + 0.16666666667*Mdt*(k1->x30000[n]+2*k2->x30000[n]+2*k3->x30000[n]+k4->x30000[n]);
		s->x30100[n] = s->x30100[n] + 0.16666666667*Mdt*(k1->x30100[n]+2*k2->x30100[n]+2*k3->x30100[n]+k4->x30100[n]);
		s->x30200[n] = s->x30200[n] + 0.16666666667*Mdt*(k1->x30200[n]+2*k2->x30200[n]+2*k3->x30200[n]+k4->x30200[n]);
		s->x30300[n] = s->x30300[n] + 0.16666666667*Mdt*(k1->x30300[n]+2*k2->x30300[n]+2*k3->x30300[n]+k4->x30300[n]);
		s->x40000[n] = s->x40000[n] + 0.16666666667*Mdt*(k1->x40000[n]+2*k2->x40000[n]+2*k3->x40000[n]+k4->x40000[n]);
		s->x40010[n] = s->x40010[n] + 0.16666666667*Mdt*(k1->x40010[n]+2*k2->x40010[n]+2*k3->x40010[n]+k4->x40010[n]);
		s->x40011[n] = s->x40011[n] + 0.16666666667*Mdt*(k1->x40011[n]+2*k2->x40011[n]+2*k3->x40011[n]+k4->x40011[n]);
		s->x40100[n] = s->x40100[n] + 0.16666666667*Mdt*(k1->x40100[n]+2*k2->x40100[n]+2*k3->x40100[n]+k4->x40100[n]);
		s->x40110[n] = s->x40110[n] + 0.16666666667*Mdt*(k1->x40110[n]+2*k2->x40110[n]+2*k3->x40110[n]+k4->x40110[n]);
		s->x40111[n] = s->x40111[n] + 0.16666666667*Mdt*(k1->x40111[n]+2*k2->x40111[n]+2*k3->x40111[n]+k4->x40111[n]);
		s->x40200[n] = s->x40200[n] + 0.16666666667*Mdt*(k1->x40200[n]+2*k2->x40200[n]+2*k3->x40200[n]+k4->x40200[n]);
		s->x40210[n] = s->x40210[n] + 0.16666666667*Mdt*(k1->x40210[n]+2*k2->x40210[n]+2*k3->x40210[n]+k4->x40210[n]);
		s->x40211[n] = s->x40211[n] + 0.16666666667*Mdt*(k1->x40211[n]+2*k2->x40211[n]+2*k3->x40211[n]+k4->x40211[n]);
		s->x40300[n] = s->x40300[n] + 0.16666666667*Mdt*(k1->x40300[n]+2*k2->x40300[n]+2*k3->x40300[n]+k4->x40300[n]);
		s->x40310[n] = s->x40310[n] + 0.16666666667*Mdt*(k1->x40310[n]+2*k2->x40310[n]+2*k3->x40310[n]+k4->x40310[n]);
		s->x40311[n] = s->x40311[n] + 0.16666666667*Mdt*(k1->x40311[n]+2*k2->x40311[n]+2*k3->x40311[n]+k4->x40311[n]);
		s->x41000[n] = s->x41000[n] + 0.16666666667*Mdt*(k1->x41000[n]+2*k2->x41000[n]+2*k3->x41000[n]+k4->x41000[n]);
		s->x41010[n] = s->x41010[n] + 0.16666666667*Mdt*(k1->x41010[n]+2*k2->x41010[n]+2*k3->x41010[n]+k4->x41010[n]);
		s->x41011[n] = s->x41011[n] + 0.16666666667*Mdt*(k1->x41011[n]+2*k2->x41011[n]+2*k3->x41011[n]+k4->x41011[n]);
		s->x41100[n] = s->x41100[n] + 0.16666666667*Mdt*(k1->x41100[n]+2*k2->x41100[n]+2*k3->x41100[n]+k4->x41100[n]);
		s->x41110[n] = s->x41110[n] + 0.16666666667*Mdt*(k1->x41110[n]+2*k2->x41110[n]+2*k3->x41110[n]+k4->x41110[n]);
		s->x41111[n] = s->x41111[n] + 0.16666666667*Mdt*(k1->x41111[n]+2*k2->x41111[n]+2*k3->x41111[n]+k4->x41111[n]);
		s->x41200[n] = s->x41200[n] + 0.16666666667*Mdt*(k1->x41200[n]+2*k2->x41200[n]+2*k3->x41200[n]+k4->x41200[n]);
		s->x41210[n] = s->x41210[n] + 0.16666666667*Mdt*(k1->x41210[n]+2*k2->x41210[n]+2*k3->x41210[n]+k4->x41210[n]);
		s->x41211[n] = s->x41211[n] + 0.16666666667*Mdt*(k1->x41211[n]+2*k2->x41211[n]+2*k3->x41211[n]+k4->x41211[n]);
		s->x41300[n] = s->x41300[n] + 0.16666666667*Mdt*(k1->x41300[n]+2*k2->x41300[n]+2*k3->x41300[n]+k4->x41300[n]);
		s->x41310[n] = s->x41310[n] + 0.16666666667*Mdt*(k1->x41310[n]+2*k2->x41310[n]+2*k3->x41310[n]+k4->x41310[n]);
		s->x41311[n] = s->x41311[n] + 0.16666666667*Mdt*(k1->x41311[n]+2*k2->x41311[n]+2*k3->x41311[n]+k4->x41311[n]);
		s->x42000[n] = s->x42000[n] + 0.16666666667*Mdt*(k1->x42000[n]+2*k2->x42000[n]+2*k3->x42000[n]+k4->x42000[n]);
		s->x42010[n] = s->x42010[n] + 0.16666666667*Mdt*(k1->x42010[n]+2*k2->x42010[n]+2*k3->x42010[n]+k4->x42010[n]);
		s->x42011[n] = s->x42011[n] + 0.16666666667*Mdt*(k1->x42011[n]+2*k2->x42011[n]+2*k3->x42011[n]+k4->x42011[n]);
		s->x42100[n] = s->x42100[n] + 0.16666666667*Mdt*(k1->x42100[n]+2*k2->x42100[n]+2*k3->x42100[n]+k4->x42100[n]);
		s->x42110[n] = s->x42110[n] + 0.16666666667*Mdt*(k1->x42110[n]+2*k2->x42110[n]+2*k3->x42110[n]+k4->x42110[n]);
		s->x42111[n] = s->x42111[n] + 0.16666666667*Mdt*(k1->x42111[n]+2*k2->x42111[n]+2*k3->x42111[n]+k4->x42111[n]);
		s->x42200[n] = s->x42200[n] + 0.16666666667*Mdt*(k1->x42200[n]+2*k2->x42200[n]+2*k3->x42200[n]+k4->x42200[n]);
		s->x42210[n] = s->x42210[n] + 0.16666666667*Mdt*(k1->x42210[n]+2*k2->x42210[n]+2*k3->x42210[n]+k4->x42210[n]);
		s->x42211[n] = s->x42211[n] + 0.16666666667*Mdt*(k1->x42211[n]+2*k2->x42211[n]+2*k3->x42211[n]+k4->x42211[n]);
		s->x42300[n] = s->x42300[n] + 0.16666666667*Mdt*(k1->x42300[n]+2*k2->x42300[n]+2*k3->x42300[n]+k4->x42300[n]);
		s->x42310[n] = s->x42310[n] + 0.16666666667*Mdt*(k1->x42310[n]+2*k2->x42310[n]+2*k3->x42310[n]+k4->x42310[n]);
		s->x42311[n] = s->x42311[n] + 0.16666666667*Mdt*(k1->x42311[n]+2*k2->x42311[n]+2*k3->x42311[n]+k4->x42311[n]);
		s->x50000[n] = s->x50000[n] + 0.16666666667*Mdt*(k1->x50000[n]+2*k2->x50000[n]+2*k3->x50000[n]+k4->x50000[n]);
		s->x50010[n] = s->x50010[n] + 0.16666666667*Mdt*(k1->x50010[n]+2*k2->x50010[n]+2*k3->x50010[n]+k4->x50010[n]);
		s->x50011[n] = s->x50011[n] + 0.16666666667*Mdt*(k1->x50011[n]+2*k2->x50011[n]+2*k3->x50011[n]+k4->x50011[n]);
		s->x50100[n] = s->x50100[n] + 0.16666666667*Mdt*(k1->x50100[n]+2*k2->x50100[n]+2*k3->x50100[n]+k4->x50100[n]);
		s->x50110[n] = s->x50110[n] + 0.16666666667*Mdt*(k1->x50110[n]+2*k2->x50110[n]+2*k3->x50110[n]+k4->x50110[n]);
		s->x50111[n] = s->x50111[n] + 0.16666666667*Mdt*(k1->x50111[n]+2*k2->x50111[n]+2*k3->x50111[n]+k4->x50111[n]);
		s->x50200[n] = s->x50200[n] + 0.16666666667*Mdt*(k1->x50200[n]+2*k2->x50200[n]+2*k3->x50200[n]+k4->x50200[n]);
		s->x50210[n] = s->x50210[n] + 0.16666666667*Mdt*(k1->x50210[n]+2*k2->x50210[n]+2*k3->x50210[n]+k4->x50210[n]);
		s->x50211[n] = s->x50211[n] + 0.16666666667*Mdt*(k1->x50211[n]+2*k2->x50211[n]+2*k3->x50211[n]+k4->x50211[n]);
		s->x50300[n] = s->x50300[n] + 0.16666666667*Mdt*(k1->x50300[n]+2*k2->x50300[n]+2*k3->x50300[n]+k4->x50300[n]);
		s->x50310[n] = s->x50310[n] + 0.16666666667*Mdt*(k1->x50310[n]+2*k2->x50310[n]+2*k3->x50310[n]+k4->x50310[n]);
		s->x50311[n] = s->x50311[n] + 0.16666666667*Mdt*(k1->x50311[n]+2*k2->x50311[n]+2*k3->x50311[n]+k4->x50311[n]);
		s->x51000[n] = s->x51000[n] + 0.16666666667*Mdt*(k1->x51000[n]+2*k2->x51000[n]+2*k3->x51000[n]+k4->x51000[n]);
		s->x51010[n] = s->x51010[n] + 0.16666666667*Mdt*(k1->x51010[n]+2*k2->x51010[n]+2*k3->x51010[n]+k4->x51010[n]);
		s->x51011[n] = s->x51011[n] + 0.16666666667*Mdt*(k1->x51011[n]+2*k2->x51011[n]+2*k3->x51011[n]+k4->x51011[n]);
		s->x51100[n] = s->x51100[n] + 0.16666666667*Mdt*(k1->x51100[n]+2*k2->x51100[n]+2*k3->x51100[n]+k4->x51100[n]);
		s->x51110[n] = s->x51110[n] + 0.16666666667*Mdt*(k1->x51110[n]+2*k2->x51110[n]+2*k3->x51110[n]+k4->x51110[n]);
		s->x51111[n] = s->x51111[n] + 0.16666666667*Mdt*(k1->x51111[n]+2*k2->x51111[n]+2*k3->x51111[n]+k4->x51111[n]);
		s->x51200[n] = s->x51200[n] + 0.16666666667*Mdt*(k1->x51200[n]+2*k2->x51200[n]+2*k3->x51200[n]+k4->x51200[n]);
		s->x51210[n] = s->x51210[n] + 0.16666666667*Mdt*(k1->x51210[n]+2*k2->x51210[n]+2*k3->x51210[n]+k4->x51210[n]);
		s->x51211[n] = s->x51211[n] + 0.16666666667*Mdt*(k1->x51211[n]+2*k2->x51211[n]+2*k3->x51211[n]+k4->x51211[n]);
		s->x51300[n] = s->x51300[n] + 0.16666666667*Mdt*(k1->x51300[n]+2*k2->x51300[n]+2*k3->x51300[n]+k4->x51300[n]);
		s->x51310[n] = s->x51310[n] + 0.16666666667*Mdt*(k1->x51310[n]+2*k2->x51310[n]+2*k3->x51310[n]+k4->x51310[n]);
		s->x51311[n] = s->x51311[n] + 0.16666666667*Mdt*(k1->x51311[n]+2*k2->x51311[n]+2*k3->x51311[n]+k4->x51311[n]);
		s->x52000[n] = s->x52000[n] + 0.16666666667*Mdt*(k1->x52000[n]+2*k2->x52000[n]+2*k3->x52000[n]+k4->x52000[n]);
		s->x52010[n] = s->x52010[n] + 0.16666666667*Mdt*(k1->x52010[n]+2*k2->x52010[n]+2*k3->x52010[n]+k4->x52010[n]);
		s->x52011[n] = s->x52011[n] + 0.16666666667*Mdt*(k1->x52011[n]+2*k2->x52011[n]+2*k3->x52011[n]+k4->x52011[n]);
		s->x52100[n] = s->x52100[n] + 0.16666666667*Mdt*(k1->x52100[n]+2*k2->x52100[n]+2*k3->x52100[n]+k4->x52100[n]);
		s->x52110[n] = s->x52110[n] + 0.16666666667*Mdt*(k1->x52110[n]+2*k2->x52110[n]+2*k3->x52110[n]+k4->x52110[n]);
		s->x52111[n] = s->x52111[n] + 0.16666666667*Mdt*(k1->x52111[n]+2*k2->x52111[n]+2*k3->x52111[n]+k4->x52111[n]);
		s->x52200[n] = s->x52200[n] + 0.16666666667*Mdt*(k1->x52200[n]+2*k2->x52200[n]+2*k3->x52200[n]+k4->x52200[n]);
		s->x52210[n] = s->x52210[n] + 0.16666666667*Mdt*(k1->x52210[n]+2*k2->x52210[n]+2*k3->x52210[n]+k4->x52210[n]);
		s->x52211[n] = s->x52211[n] + 0.16666666667*Mdt*(k1->x52211[n]+2*k2->x52211[n]+2*k3->x52211[n]+k4->x52211[n]);
		s->x52300[n] = s->x52300[n] + 0.16666666667*Mdt*(k1->x52300[n]+2*k2->x52300[n]+2*k3->x52300[n]+k4->x52300[n]);
		s->x52310[n] = s->x52310[n] + 0.16666666667*Mdt*(k1->x52310[n]+2*k2->x52310[n]+2*k3->x52310[n]+k4->x52310[n]);
		s->x52311[n] = s->x52311[n] + 0.16666666667*Mdt*(k1->x52311[n]+2*k2->x52311[n]+2*k3->x52311[n]+k4->x52311[n]);
		s->x60000[n] = s->x60000[n] + 0.16666666667*Mdt*(k1->x60000[n]+2*k2->x60000[n]+2*k3->x60000[n]+k4->x60000[n]);
		s->x60010[n] = s->x60010[n] + 0.16666666667*Mdt*(k1->x60010[n]+2*k2->x60010[n]+2*k3->x60010[n]+k4->x60010[n]);
		s->x60011[n] = s->x60011[n] + 0.16666666667*Mdt*(k1->x60011[n]+2*k2->x60011[n]+2*k3->x60011[n]+k4->x60011[n]);
		s->x60100[n] = s->x60100[n] + 0.16666666667*Mdt*(k1->x60100[n]+2*k2->x60100[n]+2*k3->x60100[n]+k4->x60100[n]);
		s->x60110[n] = s->x60110[n] + 0.16666666667*Mdt*(k1->x60110[n]+2*k2->x60110[n]+2*k3->x60110[n]+k4->x60110[n]);
		s->x60111[n] = s->x60111[n] + 0.16666666667*Mdt*(k1->x60111[n]+2*k2->x60111[n]+2*k3->x60111[n]+k4->x60111[n]);
		s->x60200[n] = s->x60200[n] + 0.16666666667*Mdt*(k1->x60200[n]+2*k2->x60200[n]+2*k3->x60200[n]+k4->x60200[n]);
		s->x60210[n] = s->x60210[n] + 0.16666666667*Mdt*(k1->x60210[n]+2*k2->x60210[n]+2*k3->x60210[n]+k4->x60210[n]);
		s->x60211[n] = s->x60211[n] + 0.16666666667*Mdt*(k1->x60211[n]+2*k2->x60211[n]+2*k3->x60211[n]+k4->x60211[n]);
		s->x60300[n] = s->x60300[n] + 0.16666666667*Mdt*(k1->x60300[n]+2*k2->x60300[n]+2*k3->x60300[n]+k4->x60300[n]);
		s->x60310[n] = s->x60310[n] + 0.16666666667*Mdt*(k1->x60310[n]+2*k2->x60310[n]+2*k3->x60310[n]+k4->x60310[n]);
		s->x60311[n] = s->x60311[n] + 0.16666666667*Mdt*(k1->x60311[n]+2*k2->x60311[n]+2*k3->x60311[n]+k4->x60311[n]);
		s->x61000[n] = s->x61000[n] + 0.16666666667*Mdt*(k1->x61000[n]+2*k2->x61000[n]+2*k3->x61000[n]+k4->x61000[n]);
		s->x61010[n] = s->x61010[n] + 0.16666666667*Mdt*(k1->x61010[n]+2*k2->x61010[n]+2*k3->x61010[n]+k4->x61010[n]);
		s->x61011[n] = s->x61011[n] + 0.16666666667*Mdt*(k1->x61011[n]+2*k2->x61011[n]+2*k3->x61011[n]+k4->x61011[n]);
		s->x61100[n] = s->x61100[n] + 0.16666666667*Mdt*(k1->x61100[n]+2*k2->x61100[n]+2*k3->x61100[n]+k4->x61100[n]);
		s->x61110[n] = s->x61110[n] + 0.16666666667*Mdt*(k1->x61110[n]+2*k2->x61110[n]+2*k3->x61110[n]+k4->x61110[n]);
		s->x61111[n] = s->x61111[n] + 0.16666666667*Mdt*(k1->x61111[n]+2*k2->x61111[n]+2*k3->x61111[n]+k4->x61111[n]);
		s->x61200[n] = s->x61200[n] + 0.16666666667*Mdt*(k1->x61200[n]+2*k2->x61200[n]+2*k3->x61200[n]+k4->x61200[n]);
		s->x61210[n] = s->x61210[n] + 0.16666666667*Mdt*(k1->x61210[n]+2*k2->x61210[n]+2*k3->x61210[n]+k4->x61210[n]);
		s->x61211[n] = s->x61211[n] + 0.16666666667*Mdt*(k1->x61211[n]+2*k2->x61211[n]+2*k3->x61211[n]+k4->x61211[n]);
		s->x61300[n] = s->x61300[n] + 0.16666666667*Mdt*(k1->x61300[n]+2*k2->x61300[n]+2*k3->x61300[n]+k4->x61300[n]);
		s->x61310[n] = s->x61310[n] + 0.16666666667*Mdt*(k1->x61310[n]+2*k2->x61310[n]+2*k3->x61310[n]+k4->x61310[n]);
		s->x61311[n] = s->x61311[n] + 0.16666666667*Mdt*(k1->x61311[n]+2*k2->x61311[n]+2*k3->x61311[n]+k4->x61311[n]);
		s->x62000[n] = s->x62000[n] + 0.16666666667*Mdt*(k1->x62000[n]+2*k2->x62000[n]+2*k3->x62000[n]+k4->x62000[n]);
		s->x62010[n] = s->x62010[n] + 0.16666666667*Mdt*(k1->x62010[n]+2*k2->x62010[n]+2*k3->x62010[n]+k4->x62010[n]);
		s->x62011[n] = s->x62011[n] + 0.16666666667*Mdt*(k1->x62011[n]+2*k2->x62011[n]+2*k3->x62011[n]+k4->x62011[n]);
		s->x62100[n] = s->x62100[n] + 0.16666666667*Mdt*(k1->x62100[n]+2*k2->x62100[n]+2*k3->x62100[n]+k4->x62100[n]);
		s->x62110[n] = s->x62110[n] + 0.16666666667*Mdt*(k1->x62110[n]+2*k2->x62110[n]+2*k3->x62110[n]+k4->x62110[n]);
		s->x62111[n] = s->x62111[n] + 0.16666666667*Mdt*(k1->x62111[n]+2*k2->x62111[n]+2*k3->x62111[n]+k4->x62111[n]);
		s->x62200[n] = s->x62200[n] + 0.16666666667*Mdt*(k1->x62200[n]+2*k2->x62200[n]+2*k3->x62200[n]+k4->x62200[n]);
		s->x62210[n] = s->x62210[n] + 0.16666666667*Mdt*(k1->x62210[n]+2*k2->x62210[n]+2*k3->x62210[n]+k4->x62210[n]);
		s->x62211[n] = s->x62211[n] + 0.16666666667*Mdt*(k1->x62211[n]+2*k2->x62211[n]+2*k3->x62211[n]+k4->x62211[n]);
		s->x62300[n] = s->x62300[n] + 0.16666666667*Mdt*(k1->x62300[n]+2*k2->x62300[n]+2*k3->x62300[n]+k4->x62300[n]);
		s->x62310[n] = s->x62310[n] + 0.16666666667*Mdt*(k1->x62310[n]+2*k2->x62310[n]+2*k3->x62310[n]+k4->x62310[n]);
		s->x62311[n] = s->x62311[n] + 0.16666666667*Mdt*(k1->x62311[n]+2*k2->x62311[n]+2*k3->x62311[n]+k4->x62311[n]);
		s->ltn[n] = s->ltn[n] + 0.16666666667*Mdt*(k1->ltn[n]+2*k2->ltn[n]+2*k3->ltn[n]+k4->ltn[n]);
		s->vip[n] = s->vip[n] + 0.16666666667*Mdt*(k1->vip[n]+2*k2->vip[n]+2*k3->vip[n]+k4->vip[n]);
		s->V10[n] = s->V10[n] + 0.16666666667*Mdt*(k1->V10[n]+2*k2->V10[n]+2*k3->V10[n]+k4->V10[n]);
		s->V11[n] = s->V11[n] + 0.16666666667*Mdt*(k1->V11[n]+2*k2->V11[n]+2*k3->V11[n]+k4->V11[n]);
		s->V12[n] = s->V12[n] + 0.16666666667*Mdt*(k1->V12[n]+2*k2->V12[n]+2*k3->V12[n]+k4->V12[n]);
		s->V01[n] = s->V01[n] + 0.16666666667*Mdt*(k1->V01[n]+2*k2->V01[n]+2*k3->V01[n]+k4->V01[n]);
		s->V02[n] = s->V02[n] + 0.16666666667*Mdt*(k1->V02[n]+2*k2->V02[n]+2*k3->V02[n]+k4->V02[n]);
		s->cAMP[n] = s->cAMP[n] + 0.16666666667*Mdt*(k1->cAMP[n]+2*k2->cAMP[n]+2*k3->cAMP[n]+k4->cAMP[n]);
		s->CREB[n] = s->CREB[n] + 0.16666666667*Mdt*(k1->CREB[n]+2*k2->CREB[n]+2*k3->CREB[n]+k4->CREB[n]);
		s->CRE[n] = s->CRE[n] + 0.16666666667*Mdt*(k1->CRE[n]+2*k2->CRE[n]+2*k3->CRE[n]+k4->CRE[n]);
	}
}

__global__ void record_result(int i, Mresult *r, Mstate *s, Mparameters *p) 
{
	// Record results
	int n;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = NTHREADS * NBLOCKS;
	for (n=tid; n<ncells; n+=stride) {

		r->pom[i+n] = s->MnPo[n] + s->McPo[n]; // per1 mRNA
		r->ptm[i+n] = s->MnPt[n] + s->McPt[n]; // per2 mRNA
		r->rom[i+n] = s->MnRo[n] + s->McRo[n]; // cry1 mRNA
		r->rtm[i+n] = s->MnRt[n] + s->McRt[n]; // cry2 mRNA
		r->bmm[i+n] = s->MnB[n] + s->McB[n]; // bmal mRNA
		r->rvm[i+n] = s->MnRev[n] + s->McRev[n]; // reverb mRNA
		r->npm[i+n] = s->MnNp[n] + s->McNp[n]; // bmal mRNA
		
		r->pot[i+n] = s->x10000[n] + s->x10100[n] + s->x20000[n] + s->x20010[n] + s->x20011[n] + s->x20100[n] + s->x20110[n] + s->x20111[n] + s->x21000[n] + s->x21010[n] + s->x21011[n] + s->x21100[n] + s->x21110[n] + s->x21111[n] + s->x22000[n] + s->x22010[n] + s->x22011[n] + s->x22100[n] + s->x22110[n] + s->x22111[n]; // per1 protien

		r->ptt[i+n] = s->x30000[n] + s->x30100[n] + s->x30200[n] + s->x30300[n] + s->x40000[n] + s->x40010[n] + s->x40011[n] + s->x40100[n] + s->x40110[n] + s->x40111[n] + s->x40200[n] + s->x40210[n] + s->x40211[n] + s->x40300[n] + s->x40310[n] + s->x40311[n] + s->x41000[n] + s->x41010[n] + s->x41011[n] + s->x41100[n] + s->x41110[n] + s->x41111[n] + s->x41200[n] + s->x41210[n] + s->x41211[n] + s->x41300[n] + s->x41310[n] + s->x41311[n] + s->x42000[n] + s->x42010[n] + s->x42011[n] + s->x42100[n] + s->x42110[n] + s->x42111[n] + s->x42200[n] + s->x42210[n] + s->x42211[n] + s->x42300[n] + s->x42310[n] + s->x42311[n] + s->x50000[n] + s->x50010[n] + s->x50011[n] + s->x50100[n] + s->x50110[n] + s->x50111[n] + s->x50200[n] + s->x50210[n] + s->x50211[n] + s->x50300[n] + s->x50310[n] + s->x50311[n] + s->x51000[n] + s->x51010[n] + s->x51011[n] + s->x51100[n] + s->x51110[n] + s->x51111[n] + s->x51200[n] + s->x51210[n] + s->x51211[n] + s->x51300[n] + s->x51310[n] + s->x51311[n] + s->x52000[n] + s->x52010[n] + s->x52011[n] + s->x52100[n] + s->x52110[n] + s->x52111[n] + s->x52200[n] + s->x52210[n] + s->x52211[n] + s->x52300[n] + s->x52310[n] + s->x52311[n] + s->x60000[n] + s->x60010[n] + s->x60011[n] + s->x60100[n] + s->x60110[n] + s->x60111[n] + s->x60200[n] + s->x60210[n] + s->x60211[n] + s->x60300[n] + s->x60310[n] + s->x60311[n] + s->x61000[n] + s->x61010[n] + s->x61011[n] + s->x61100[n] + s->x61110[n] + s->x61111[n] + s->x61200[n] + s->x61210[n] + s->x61211[n] + s->x61300[n] + s->x61310[n] + s->x61311[n] + s->x62000[n] + s->x62010[n] + s->x62011[n] + s->x62100[n] + s->x62110[n] + s->x62111[n] + s->x62200[n] + s->x62210[n] + s->x62211[n] + s->x62300[n] + s->x62310[n] + s->x62311[n]; // per2 protein

		r->rot[i+n] = s->x01000[n] + s->x01010[n] + s->x01011[n] + s->x21000[n] + s->x21010[n] + s->x21011[n] + s->x21100[n] + s->x21110[n] + s->x21111[n] + s->x41000[n] + s->x41010[n] + s->x41011[n] + s->x41100[n] + s->x41110[n] + s->x41111[n] + s->x41200[n] + s->x41210[n] + s->x41211[n] + s->x41300[n] + s->x41310[n] + s->x41311[n] + s->x51000[n] + s->x51010[n] + s->x51011[n] + s->x51100[n] + s->x51110[n] + s->x51111[n] + s->x51200[n] + s->x51210[n] + s->x51211[n] + s->x51300[n] + s->x51310[n] + s->x51311[n] + s->x61000[n] + s->x61010[n] + s->x61011[n] + s->x61100[n] + s->x61110[n] + s->x61111[n] + s->x61200[n] + s->x61210[n] + s->x61211[n] + s->x61300[n] + s->x61310[n] + s->x61311[n] + s->V01[n] + s->V11[n]; // cry1 protein

		r->rtt[i+n] = s->x02000[n] + s->x02010[n] + s->x02011[n] + s->x22000[n] + s->x22010[n] + s->x22011[n] + s->x22100[n] + s->x22110[n] + s->x22111[n] + s->x42000[n] + s->x42010[n] + s->x42011[n] + s->x42100[n] + s->x42110[n] + s->x42111[n] + s->x42200[n] + s->x42210[n] + s->x42211[n] + s->x42300[n] + s->x42310[n] + s->x42311[n] + s->x52000[n] + s->x52010[n] + s->x52011[n] + s->x52100[n] + s->x52110[n] + s->x52111[n] + s->x52200[n] + s->x52210[n] + s->x52211[n] + s->x52300[n] + s->x52310[n] + s->x52311[n] + s->x62000[n] + s->x62010[n] + s->x62011[n] + s->x62100[n] + s->x62110[n] + s->x62111[n] + s->x62200[n] + s->x62210[n] + s->x62211[n] + s->x62300[n] + s->x62310[n] + s->x62311[n] + s->V02[n] + s->V12[n]; // cry2 protein

		r->bmt[i+n] = s->B[n] + s->BC[n] + s->x00001[n] + s->x00011[n] + s->x01011[n] + s->x02011[n] + s->x20011[n] + s->x20111[n] + s->x21011[n] + s->x21111[n] + s->x22011[n] + s->x22111[n] + s->x40011[n] + s->x40111[n] + s->x40211[n] + s->x40311[n] + s->x41011[n] + s->x41111[n] + s->x41211[n] + s->x41311[n] + s->x42011[n] + s->x42111[n] + s->x42211[n] + s->x42311[n] + s->x50011[n] + s->x50111[n] + s->x50211[n] + s->x50311[n] + s->x51011[n] + s->x51111[n] + s->x51211[n] + s->x51311[n] + s->x52011[n] + s->x52111[n] + s->x52211[n] + s->x52311[n] + s->x60011[n] + s->x60111[n] + s->x60211[n] + s->x60311[n] + s->x61011[n] + s->x61111[n] + s->x61211[n] + s->x61311[n] + s->x62011[n] + s->x62111[n] + s->x62211[n] + s->x62311[n]; // bmal protein

//		r->clt[i+n] = s->Cl[n] + s->BC[n] + s->x00001[n] + s->x00011[n] + s->x01011[n] + s->x02011[n] + s->x20011[n] + s->x20111[n] + s->x21011[n] + s->x21111[n] + s->x22011[n] + s->x22111[n] + s->x40011[n] + s->x40111[n] + s->x40211[n] + s->x40311[n] + s->x41011[n] + s->x41111[n] + s->x41211[n] + s->x41311[n] + s->x42011[n] + s->x42111[n] + s->x42211[n] + s->x42311[n] + s->x50011[n] + s->x50111[n] + s->x50211[n] + s->x50311[n] + s->x51011[n] + s->x51111[n] + s->x51211[n] + s->x51311[n] + s->x52011[n] + s->x52111[n] + s->x52211[n] + s->x52311[n] + s->x60011[n] + s->x60111[n] + s->x60211[n] + s->x60311[n] + s->x61011[n] + s->x61111[n] + s->x61211[n] + s->x61311[n] + s->x62011[n] + s->x62111[n] + s->x62211[n] + s->x62311[n]; // total clock protein

//		r->clct[i+n] = s->Cl[n] + s->BC[n] + s->x00001[n]; // cytoplasmic clock protein

//		r->clnt[i+n] = s->x00011[n] + s->x01011[n] + s->x02011[n] + s->x20011[n] + s->x20111[n] + s->x21011[n] + s->x21111[n] + s->x22011[n] + s->x22111[n] + s->x40011[n] + s->x40111[n] + s->x40211[n] + s->x40311[n] + s->x41011[n] + s->x41111[n] + s->x41211[n] + s->x41311[n] + s->x42011[n] + s->x42111[n] + s->x42211[n] + s->x42311[n] + s->x50011[n] + s->x50111[n] + s->x50211[n] + s->x50311[n] + s->x51011[n] + s->x51111[n] + s->x51211[n] + s->x51311[n] + s->x52011[n] + s->x52111[n] + s->x52211[n] + s->x52311[n] + s->x60011[n] + s->x60111[n] + s->x60211[n] + s->x60311[n] + s->x61011[n] + s->x61111[n] + s->x61211[n] + s->x61311[n] + s->x62011[n] + s->x62111[n] + s->x62211[n] + s->x62311[n]; // nuclear clock protein

		r->revt[i+n] = s->revn[n] + s->cyrev[n] + s->revng[n] + s->cyrevg[n] + s->revngp[n] + s->cyrevgp[n] + s->revnp[n] + s->cyrevp[n];

		r->cre[i+n] = s->CRE[n];
		r->vip[i+n] = s->vip[n] + s->V10[n] + s->V11[n] + s->V12[n]; // all vip (free and bound to receptor)
		r->G[i+n] = s->G[n];
		//Unbound GSK3B in cytoplasm (x00200)
		r->gsk[i+n] = p->Gt[n] - (s->x30300[n]+s->x40300[n]+s->x40310[n]+s->x40311[n]+s->x41300[n]+s->x41310[n]+s->x41311[n]+s->x42300[n]+s->x42310[n]+s->x42311[n]+s->x50300[n]+s->x50310[n]+s->x50311[n]+s->x51300[n]+s->x51310[n]+s->x51311[n]+s->x52300[n]+s->x52310[n]+s->x52311[n]+s->x60300[n]+s->x60310[n]+s->x60311[n]+s->x61300[n]+s->x61310[n]+s->x61311[n]+s->x62300[n]+s->x62310[n]+s->x62311[n]+s->cyrevg[n]+s->revng[n]+s->cyrevgp[n]+s->revngp[n]+s->x00210[n]+s->x30200[n]+s->x40200[n]+s->x40210[n]+s->x40211[n]+s->x41200[n]+s->x41210[n]+s->x41211[n]+s->x42200[n]+s->x42210[n]+s->x42211[n]+s->x50200[n]+s->x50210[n]+s->x50211[n]+s->x51200[n]+s->x51210[n]+s->x51211[n]+s->x52200[n]+s->x52210[n]+s->x52211[n]+s->x60200[n]+s->x60210[n]+s->x60211[n]+s->x61200[n]+s->x61210[n]+s->x61211[n]+s->x62200[n]+s->x62210[n]+s->x62211[n]);

//p->Ct[n] - (s->x30300[n]+s->x40300[n]+s->x40310[n]+s->x40311[n]+s->x41300[n]+s->x41310[n]+s->x41311[n]+s->x42300[n]+s->x42310[n]+s->x42311[n]+s->x50300[n]+s->x50310[n]+s->x50311[n]+s->x51300[n]+s->x51310[n]+s->x51311[n]+s->x52300[n]+s->x52310[n]+s->x52311[n]+s->x60300[n]+s->x60310[n]+s->x60311[n]+s->x61300[n]+s->x61310[n]+s->x61311[n]+s->x62300[n]+s->x62310[n]+s->x62311[n] + s->x00110[n]+s->x10100[n]+s->x20100[n]+s->x20110[n]+s->x20111[n]+s->x21100[n]+s->x21110[n]+s->x21111[n]+s->x22100[n]+s->x22110[n]+s->x22111[n]+s->x30100[n]+s->x40100[n]+s->x40110[n]+s->x40111[n]+s->x41100[n]+s->x41110[n]+s->x41111[n]+s->x42100[n]+s->x42110[n]+s->x42111[n]+s->x50100[n]+s->x50110[n]+s->x50111[n]+s->x51100[n]+s->x51110[n]+s->x51111[n]+s->x52100[n]+s->x52110[n]+s->x52111[n]+s->x60100[n]+s->x60110[n]+s->x60111[n]+s->x61100[n]+s->x61110[n]+s->x61111[n]+s->x62100[n]+s->x62110[n]+s->x62111[n]);

		r->xtra[i+n] = s->V10[n];
	}
}

void rhs_wrapper(double t, Mstate *s, Mstate *s_dot, Mparameters *p, Estate *x, mclock_t *input, int *upstream, int LIGHT) {
	rhs <<< NBLOCKS, NTHREADS >>> (t, s, s_dot, p, x, input, upstream, LIGHT);
}

void lincomb_wrapper(mclock_t c, Mstate *s, Mstate *s1, Mstate *s2) {
	lincomb <<< NBLOCKS, NTHREADS >>> (c, s, s1, s2);
}

void rk4_wrapper(Mstate *s, Mstate *k1, Mstate *k2, Mstate *k3, Mstate *k4, double t, mclock_t idx) {
	rk4 <<< NBLOCKS, NTHREADS >>> (s, k1, k2, k3, k4, t, idx);
}

void record_result_wrapper(int i, Mresult *r, Mstate *s, Mparameters *p) {
	record_result <<< NBLOCKS, NTHREADS >>> (i, r, s, p);
}

