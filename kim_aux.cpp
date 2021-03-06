/* Host functions needed for initializing state and parameter structs for the detailed model from Kim & Forger, MSB 2012 */

/* Includes functions:
    FILL_STATE: sets all elements of state variable arrays contained in struct (s) to values from array (value)
		of length (length)
    READ_PARAMS: reads in WIDTH number of parameters from file "parameters.txt" in local folder
		 and saves them in array (input)
    WRITE_PARAMS: writes WIDTH number of parameters from array (output) to file "parameters.txt" in local folder
		  each row in this file contains parameters for one cell, and each column is a different parameter
    FILL_PARAMS: sets all elements of parameter arrays contained in struct (p) to values from array (value)
		 of length ncells. if NEW = 1, generates a new parameter set and writes it to a file. if NEW = 0,
		 reads in the parameters from file.

    DEFINE_GRID: reads in header line with grid size (rows columns) and then matrix of size rows x columns with
		 a 1 in each entry representing a cell and a 0 everywhere else

    DEFINE_CONNECT: reads in connectivity matrix

    FIND_NBRS: creates a struct nbd with arrays l, r, u, d where l[N] is the index of the cell to the left of cell N
	       r, u, and d are right, up and down respectively (2D neighborhood)
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
#include "kim.h"
#include "parameters.h"
#include "utils.h"
#include <algorithm>

void vec2Mstate(mclock_t *vec, Mstate *M, int n)
{
	M->GR[n] = vec[0];
	M->G[n] = vec[1];
	M->GrR[n] = vec[2];
	M->Gr[n] = vec[3];
	M->GcR[n] = vec[4];
	M->Gc[n] = vec[5];
	M->GBR[n] = vec[6];
	M->GB[n] = vec[7];
	M->GBRb[n] = vec[8];
	M->GBb[n] = vec[9];
	M->MnPo[n] = vec[10];
	M->McPo[n] = vec[11];
	M->MnPt[n] = vec[12];
	M->McPt[n] = vec[13];
	M->MnRt[n] = vec[14];
	M->McRt[n] = vec[15];
	M->MnRev[n] = vec[16];
	M->McRev[n] = vec[17];
	M->MnRo[n] = vec[18];
	M->McRo[n] = vec[19];
	M->MnB[n] = vec[20];
	M->McB[n] = vec[21];
	M->MnNp[n] = vec[22];
	M->McNp[n] = vec[23];
	M->B[n] = vec[24];
	M->Cl[n] = vec[25];
	M->BC[n] = vec[26];
	M->cyrev[n] = vec[27];
	M->revn[n] = vec[28];
	M->cyrevg[n] = vec[29];
	M->revng[n] = vec[30];
	M->cyrevgp[n] = vec[31];
	M->revngp[n] = vec[32];
	M->cyrevp[n] = vec[33];
	M->revnp[n] = vec[34];
	M->gto[n] = vec[35];
	M->x00001[n] = vec[36];
	M->x00011[n] = vec[37];
//	M->x00100[n] = vec[38];
	M->x00110[n] = vec[38];
//	M->x00200[n] = vec[40];
	M->x00210[n] = vec[39];
	M->x01000[n] = vec[40];
	M->x01010[n] = vec[41];
	M->x01011[n] = vec[42];
	M->x02000[n] = vec[43];
	M->x02010[n] = vec[44];
	M->x02011[n] = vec[45];
	M->x10000[n] = vec[46];
	M->x10100[n] = vec[47];
	M->x20000[n] = vec[48];
	M->x20010[n] = vec[49];
	M->x20011[n] = vec[50];
	M->x20100[n] = vec[51];
	M->x20110[n] = vec[52];
	M->x20111[n] = vec[53];
	M->x21000[n] = vec[54];
	M->x21010[n] = vec[55];
	M->x21011[n] = vec[56];
	M->x21100[n] = vec[57];
	M->x21110[n] = vec[58];
	M->x21111[n] = vec[59];
	M->x22000[n] = vec[60];
	M->x22010[n] = vec[61];
	M->x22011[n] = vec[62];
	M->x22100[n] = vec[63];
	M->x22110[n] = vec[64];
	M->x22111[n] = vec[65];
	M->x30000[n] = vec[66];
	M->x30100[n] = vec[67];
	M->x30200[n] = vec[68];
	M->x30300[n] = vec[69];
	M->x40000[n] = vec[70];
	M->x40010[n] = vec[71];
	M->x40011[n] = vec[72];
	M->x40100[n] = vec[73];
	M->x40110[n] = vec[74];
	M->x40111[n] = vec[75];
	M->x40200[n] = vec[76];
	M->x40210[n] = vec[77];
	M->x40211[n] = vec[78];
	M->x40300[n] = vec[79];
	M->x40310[n] = vec[80];
	M->x40311[n] = vec[81];
	M->x41000[n] = vec[82];
	M->x41010[n] = vec[83];
	M->x41011[n] = vec[84];
	M->x41100[n] = vec[85];
	M->x41110[n] = vec[86];
	M->x41111[n] = vec[87];
	M->x41200[n] = vec[88];
	M->x41210[n] = vec[89];
	M->x41211[n] = vec[90];
	M->x41300[n] = vec[91];
	M->x41310[n] = vec[92];
	M->x41311[n] = vec[93];
	M->x42000[n] = vec[94];
	M->x42010[n] = vec[95];
	M->x42011[n] = vec[96];
	M->x42100[n] = vec[97];
	M->x42110[n] = vec[98];
	M->x42111[n] = vec[99];
	M->x42200[n] = vec[100];
	M->x42210[n] = vec[101];
	M->x42211[n] = vec[102];
	M->x42300[n] = vec[103];
	M->x42310[n] = vec[104];
	M->x42311[n] = vec[105];
	M->x50000[n] = vec[106];
	M->x50010[n] = vec[107];
	M->x50011[n] = vec[108];
	M->x50100[n] = vec[109];
	M->x50110[n] = vec[110];
	M->x50111[n] = vec[111];
	M->x50200[n] = vec[112];
	M->x50210[n] = vec[113];
	M->x50211[n] = vec[114];
	M->x50300[n] = vec[115];
	M->x50310[n] = vec[116];
	M->x50311[n] = vec[117];
	M->x51000[n] = vec[118];
	M->x51010[n] = vec[119];
	M->x51011[n] = vec[120];
	M->x51100[n] = vec[121];
	M->x51110[n] = vec[122];
	M->x51111[n] = vec[123];
	M->x51200[n] = vec[124];
	M->x51210[n] = vec[125];
	M->x51211[n] = vec[126];
	M->x51300[n] = vec[127];
	M->x51310[n] = vec[128];
	M->x51311[n] = vec[129];
	M->x52000[n] = vec[130];
	M->x52010[n] = vec[131];
	M->x52011[n] = vec[132];
	M->x52100[n] = vec[133];
	M->x52110[n] = vec[134];
	M->x52111[n] = vec[135];
	M->x52200[n] = vec[136];
	M->x52210[n] = vec[137];
	M->x52211[n] = vec[138];
	M->x52300[n] = vec[139];
	M->x52310[n] = vec[140];
	M->x52311[n] = vec[141];
	M->x60000[n] = vec[142];
	M->x60010[n] = vec[143];
	M->x60011[n] = vec[144];
	M->x60100[n] = vec[145];
	M->x60110[n] = vec[146];
	M->x60111[n] = vec[147];
	M->x60200[n] = vec[148];
	M->x60210[n] = vec[149];
	M->x60211[n] = vec[150];
	M->x60300[n] = vec[151];
	M->x60310[n] = vec[152];
	M->x60311[n] = vec[153];
	M->x61000[n] = vec[154];
	M->x61010[n] = vec[155];
	M->x61011[n] = vec[156];
	M->x61100[n] = vec[157];
	M->x61110[n] = vec[158];
	M->x61111[n] = vec[159];
	M->x61200[n] = vec[160];
	M->x61210[n] = vec[161];
	M->x61211[n] = vec[162];
	M->x61300[n] = vec[163];
	M->x61310[n] = vec[164];
	M->x61311[n] = vec[165];
	M->x62000[n] = vec[166];
	M->x62010[n] = vec[167];
	M->x62011[n] = vec[168];
	M->x62100[n] = vec[169];
	M->x62110[n] = vec[170];
	M->x62111[n] = vec[171];
	M->x62200[n] = vec[172];
	M->x62210[n] = vec[173];
	M->x62211[n] = vec[174];
	M->x62300[n] = vec[175];
	M->x62310[n] = vec[176];
	M->x62311[n] = vec[177];
	M->ltn[n] = vec[178];
	M->vip[n] = vec[179];
	M->V10[n] = vec[180];
	M->V11[n] = vec[181];
	M->V12[n] = vec[182];
	M->V01[n] = vec[183];
	M->V02[n] = vec[184];
	M->cAMP[n] = vec[185];
	M->CREB[n] = vec[186];
	M->CRE[n] = vec[187];
}

void Mstate2vec(Mstate *M, int n, mclock_t *vec)
{
	vec[0] = M->GR[n];
	vec[1] = M->G[n];
	vec[2] = M->GrR[n];
	vec[3] = M->Gr[n];
	vec[4] = M->GcR[n];
	vec[5] = M->Gc[n];
	vec[6] = M->GBR[n];
	vec[7] = M->GB[n];
	vec[8] = M->GBRb[n];
	vec[9] = M->GBb[n];
	vec[10] = M->MnPo[n];
	vec[11] = M->McPo[n];
	vec[12] = M->MnPt[n];
	vec[13] = M->McPt[n];
	vec[14] = M->MnRt[n];
	vec[15] = M->McRt[n];
	vec[16] = M->MnRev[n];
	vec[17] = M->McRev[n];
	vec[18] = M->MnRo[n];
	vec[19] = M->McRo[n];
	vec[20] = M->MnB[n];
	vec[21] = M->McB[n];
	vec[22] = M->MnNp[n];
	vec[23] = M->McNp[n];
	vec[24] = M->B[n];
	vec[25] = M->Cl[n];
	vec[26] = M->BC[n];
	vec[27] = M->cyrev[n];
	vec[28] = M->revn[n];
	vec[29] = M->cyrevg[n];
	vec[30] = M->revng[n];
	vec[31] = M->cyrevgp[n];
	vec[32] = M->revngp[n];
	vec[33] = M->cyrevp[n];
	vec[34] = M->revnp[n];
	vec[35] = M->gto[n];
	vec[36] = M->x00001[n];
	vec[37] = M->x00011[n];
//	vec[38] = M->x00100[n];
	vec[38] = M->x00110[n];
//	vec[40] = M->x00200[n];
	vec[39] = M->x00210[n];
	vec[40] = M->x01000[n];
	vec[41] = M->x01010[n];
	vec[42] = M->x01011[n];
	vec[43] = M->x02000[n];
	vec[44] = M->x02010[n];
	vec[45] = M->x02011[n];
	vec[46] = M->x10000[n];
	vec[47] = M->x10100[n];
	vec[48] = M->x20000[n];
	vec[49] = M->x20010[n];
	vec[50] = M->x20011[n];
	vec[51] = M->x20100[n];
	vec[52] = M->x20110[n];
	vec[53] = M->x20111[n];
	vec[54] = M->x21000[n];
	vec[55] = M->x21010[n];
	vec[56] = M->x21011[n];
	vec[57] = M->x21100[n];
	vec[58] = M->x21110[n];
	vec[59] = M->x21111[n];
	vec[60] = M->x22000[n];
	vec[61] = M->x22010[n];
	vec[62] = M->x22011[n];
	vec[63] = M->x22100[n];
	vec[64] = M->x22110[n];
	vec[65] = M->x22111[n];
	vec[66] = M->x30000[n];
	vec[67] = M->x30100[n];
	vec[68] = M->x30200[n];
	vec[69] = M->x30300[n];
	vec[70] = M->x40000[n];
	vec[71] = M->x40010[n];
	vec[72] = M->x40011[n];
	vec[73] = M->x40100[n];
	vec[74] = M->x40110[n];
	vec[75] = M->x40111[n];
	vec[76] = M->x40200[n];
	vec[77] = M->x40210[n];
	vec[78] = M->x40211[n];
	vec[79] = M->x40300[n];
	vec[80] = M->x40310[n];
	vec[81] = M->x40311[n];
	vec[82] = M->x41000[n];
	vec[83] = M->x41010[n];
	vec[84] = M->x41011[n];
	vec[85] = M->x41100[n];
	vec[86] = M->x41110[n];
	vec[87] = M->x41111[n];
	vec[88] = M->x41200[n];
	vec[89] = M->x41210[n];
	vec[90] = M->x41211[n];
	vec[91] = M->x41300[n];
	vec[92] = M->x41310[n];
	vec[93] = M->x41311[n];
	vec[94] = M->x42000[n];
	vec[95] = M->x42010[n];
	vec[96] = M->x42011[n];
	vec[97] = M->x42100[n];
	vec[98] = M->x42110[n];
	vec[99] = M->x42111[n];
	vec[100] = M->x42200[n];
	vec[101] = M->x42210[n];
	vec[102] = M->x42211[n];
	vec[103] = M->x42300[n];
	vec[104] = M->x42310[n];
	vec[105] = M->x42311[n];
	vec[106] = M->x50000[n];
	vec[107] = M->x50010[n];
	vec[108] = M->x50011[n];
	vec[109] = M->x50100[n];
	vec[110] = M->x50110[n];
	vec[111] = M->x50111[n];
	vec[112] = M->x50200[n];
	vec[113] = M->x50210[n];
	vec[114] = M->x50211[n];
	vec[115] = M->x50300[n];
	vec[116] = M->x50310[n];
	vec[117] = M->x50311[n];
	vec[118] = M->x51000[n];
	vec[119] = M->x51010[n];
	vec[120] = M->x51011[n];
	vec[121] = M->x51100[n];
	vec[122] = M->x51110[n];
	vec[123] = M->x51111[n];
	vec[124] = M->x51200[n];
	vec[125] = M->x51210[n];
	vec[126] = M->x51211[n];
	vec[127] = M->x51300[n];
	vec[128] = M->x51310[n];
	vec[129] = M->x51311[n];
	vec[130] = M->x52000[n];
	vec[131] = M->x52010[n];
	vec[132] = M->x52011[n];
	vec[133] = M->x52100[n];
	vec[134] = M->x52110[n];
	vec[135] = M->x52111[n];
	vec[136] = M->x52200[n];
	vec[137] = M->x52210[n];
	vec[138] = M->x52211[n];
	vec[139] = M->x52300[n];
	vec[140] = M->x52310[n];
	vec[141] = M->x52311[n];
	vec[142] = M->x60000[n];
	vec[143] = M->x60010[n];
	vec[144] = M->x60011[n];
	vec[145] = M->x60100[n];
	vec[146] = M->x60110[n];
	vec[147] = M->x60111[n];
	vec[148] = M->x60200[n];
	vec[149] = M->x60210[n];
	vec[150] = M->x60211[n];
	vec[151] = M->x60300[n];
	vec[152] = M->x60310[n];
	vec[153] = M->x60311[n];
	vec[154] = M->x61000[n];
	vec[155] = M->x61010[n];
	vec[156] = M->x61011[n];
	vec[157] = M->x61100[n];
	vec[158] = M->x61110[n];
	vec[159] = M->x61111[n];
	vec[160] = M->x61200[n];
	vec[161] = M->x61210[n];
	vec[162] = M->x61211[n];
	vec[163] = M->x61300[n];
	vec[164] = M->x61310[n];
	vec[165] = M->x61311[n];
	vec[166] = M->x62000[n];
	vec[167] = M->x62010[n];
	vec[168] = M->x62011[n];
	vec[169] = M->x62100[n];
	vec[170] = M->x62110[n];
	vec[171] = M->x62111[n];
	vec[172] = M->x62200[n];
	vec[173] = M->x62210[n];
	vec[174] = M->x62211[n];
	vec[175] = M->x62300[n];
	vec[176] = M->x62310[n];
	vec[177] = M->x62311[n];
	vec[178] = M->ltn[n];
	vec[179] = M->vip[n];
	vec[180] = M->V10[n];
	vec[181] = M->V11[n];
	vec[182] = M->V12[n];
	vec[183] = M->V01[n];
	vec[184] = M->V02[n];
	vec[185] = M->cAMP[n];
	vec[186] = M->CREB[n];
	vec[187] = M->CRE[n];
}

void vec2Mparams(mclock_t *vec, Mparameters *p, int n)
{
	p->trPo[n] = vec[0];
	p->trPt[n] = vec[1];
	p->trRo[n] = vec[2];
	p->trRt[n] = vec[3];
	p->trB[n] = vec[4];
	p->trRev[n] = vec[5];
	p->trNp[n] = vec[6];
	p->tlp[n] = vec[7];
	p->tlr[n] = vec[8];
	p->tlb[n] = vec[9];
	p->tlrev[n] = vec[10];
	p->tlc[n] = vec[11];
	p->tlnp[n] = vec[12];
	p->agp[n] = vec[13];
	p->dg[n] = vec[14];
	p->ac[n] = vec[15];
	p->dc[n] = vec[16];
	p->ar[n] = vec[17];
	p->dr[n] = vec[18];
	p->cbin[n] = vec[19];
	p->uncbin[n] = vec[20];
	p->bbin[n] = vec[21];
	p->unbbin[n] = vec[22];
	p->cbbin[n] = vec[23];
	p->uncbbin[n] = vec[24];
	p->ag[n] = vec[25];
	p->bin[n] = vec[26];
	p->unbin[n] = vec[27];
	p->binrev[n] = vec[28];
	p->unbinrev[n] = vec[29];
	p->binr[n] = vec[30];
	p->unbinr[n] = vec[31];
	p->binc[n] = vec[32];
	p->unbinc[n] = vec[33];
	p->binrevb[n] = vec[34];
	p->unbinrevb[n] = vec[35];
	p->tmc[n] = vec[36];
	p->tmcrev[n] = vec[37];
	p->nl[n] = vec[38];
	p->ne[n] = vec[39];
	p->nlrev[n] = vec[40];
	p->nerev[n] = vec[41];
	p->lne[n] = vec[42];
	p->nlbc[n] = vec[43];
	p->hoo[n] = vec[44];
	p->hto[n] = vec[45];
	p->phos[n] = vec[46];
	p->lono[n] = vec[47];
	p->lont[n] = vec[48];
	p->lta[n] = vec[49];
	p->ltb[n] = vec[50];
	p->trgto[n] = vec[51];
	p->ugto[n] = vec[52];
	p->Nf[n] = vec[53];
	p->up[n] = vec[54];
	p->uro[n] = vec[55];
	p->urt[n] = vec[56];
	p->umNp[n] = vec[57];
	p->umPo[n] = vec[58];
	p->umPt[n] = vec[59];
	p->umRo[n] = vec[60];
	p->umRt[n] = vec[61];
	p->ub[n] = vec[62];
	p->uc[n] = vec[63];
	p->ubc[n] = vec[64];
	p->upu[n] = vec[65];
	p->urev[n] = vec[66];
	p->uprev[n] = vec[67];
	p->umB[n] = vec[68];
	p->umRev[n] = vec[69];
	p->uv[n] = vec[70];
	p->Vt[n] = vec[71];
	p->vbin[n] = vec[72];
	p->unvbin[n] = vec[73];
	p->cvbin[n] = vec[74];
	p->uncvbin[n] = vec[75];
	p->vs[n] = vec[76];
	p->us[n] = vec[77];
	p->sbin[n] = vec[78];
	p->unsbin[n] = vec[79];
	p->CtrPo[n] = vec[80];
	p->CtrPt[n] = vec[81];
	p->vpr[n] = vec[82];
	p->Ct[n] = vec[83];
	p->Gt[n] = vec[84];
}

void Mparams2vec(Mparameters *p, int n, mclock_t *vec)
{
	vec[0] = p->trPo[n];
	vec[1] = p->trPt[n];
	vec[2] = p->trRo[n];
	vec[3] = p->trRt[n];
	vec[4] = p->trB[n];
	vec[5] = p->trRev[n];
	vec[6] = p->trNp[n];
	vec[7] = p->tlp[n];
	vec[8] = p->tlr[n];
	vec[9] = p->tlb[n];
	vec[10] = p->tlrev[n];
	vec[11] = p->tlc[n];
	vec[12] = p->tlnp[n];
	vec[13] = p->agp[n];
	vec[14] = p->dg[n];
	vec[15] = p->ac[n];
	vec[16] = p->dc[n];
	vec[17] = p->ar[n];
	vec[18] = p->dr[n];
	vec[19] = p->cbin[n];
	vec[20] = p->uncbin[n];
	vec[21] = p->bbin[n];
	vec[22] = p->unbbin[n];
	vec[23] = p->cbbin[n];
	vec[24] = p->uncbbin[n];
	vec[25] = p->ag[n];
	vec[26] = p->bin[n];
	vec[27] = p->unbin[n];
	vec[28] = p->binrev[n];
	vec[29] = p->unbinrev[n];
	vec[30] = p->binr[n];
	vec[31] = p->unbinr[n];
	vec[32] = p->binc[n];
	vec[33] = p->unbinc[n];
	vec[34] = p->binrevb[n];
	vec[35] = p->unbinrevb[n];
	vec[36] = p->tmc[n];
	vec[37] = p->tmcrev[n];
	vec[38] = p->nl[n];
	vec[39] = p->ne[n];
	vec[40] = p->nlrev[n];
	vec[41] = p->nerev[n];
	vec[42] = p->lne[n];
	vec[43] = p->nlbc[n];
	vec[44] = p->hoo[n];
	vec[45] = p->hto[n];
	vec[46] = p->phos[n];
	vec[47] = p->lono[n];
	vec[48] = p->lont[n];
	vec[49] = p->lta[n];
	vec[50] = p->ltb[n];
	vec[51] = p->trgto[n];
	vec[52] = p->ugto[n];
	vec[53] = p->Nf[n];
	vec[54] = p->up[n];
	vec[55] = p->uro[n];
	vec[56] = p->urt[n];
	vec[57] = p->umNp[n];
	vec[58] = p->umPo[n];
	vec[59] = p->umPt[n];
	vec[60] = p->umRo[n];
	vec[61] = p->umRt[n];
	vec[62] = p->ub[n];
	vec[63] = p->uc[n];
	vec[64] = p->ubc[n];
	vec[65] = p->upu[n];
	vec[66] = p->urev[n];
	vec[67] = p->uprev[n];
	vec[68] = p->umB[n];
	vec[69] = p->umRev[n];
	vec[70] = p->uv[n];
	vec[71] = p->Vt[n];
	vec[72] = p->vbin[n];
	vec[73] = p->unvbin[n];
	vec[74] = p->cvbin[n];
	vec[75] = p->uncvbin[n];
	vec[76] = p->vs[n];
	vec[77] = p->us[n];
	vec[78] = p->sbin[n];
	vec[79] = p->unsbin[n];
	vec[80] = p->CtrPo[n];
	vec[81] = p->CtrPt[n];
	vec[82] = p->vpr[n];
	vec[83] = p->Ct[n];
	vec[84] = p->Gt[n];
}

void vec2Mresult(mclock_t *vec, Mresult *r, int n)
{
	r->pom[n] = vec[0];
	r->ptm[n] = vec[1];
	r->rom[n] = vec[2];
	r->rtm[n] = vec[3];
	r->bmm[n] = vec[4];
	r->rvm[n] = vec[5];
	r->npm[n] = vec[6];
	r->pot[n] = vec[7];
	r->ptt[n] = vec[8];
	r->rot[n] = vec[9];
	r->rtt[n] = vec[10];
	r->bmt[n] = vec[11];
	r->clt[n] = vec[12];
	r->clct[n] = vec[13];
	r->clnt[n] = vec[14];
	r->revt[n] = vec[15];
	r->cre[n] = vec[16];
	r->vip[n] = vec[17];
	r->G[n] = vec[18];
	r->gsk[n] = vec[19];
	r->xtra[n] = vec[20];
}

void Mresult2vec(Mresult *r, int n, mclock_t *vec)
{
	vec[0] = r->pom[n];
	vec[1] = r->ptm[n];
	vec[2] = r->rom[n];
	vec[3] = r->rtm[n];
	vec[4] = r->bmm[n];
	vec[5] = r->rvm[n];
	vec[6] = r->npm[n];
	vec[7] = r->pot[n];
	vec[8] = r->ptt[n];
	vec[9] = r->rot[n];
	vec[10] = r->rtt[n];
	vec[11] = r->bmt[n];
	vec[12] = r->clt[n];
	vec[13] = r->clct[n];
	vec[14] = r->clnt[n];
	vec[15] = r->revt[n];
	vec[16] = r->cre[n];
	vec[17] = r->vip[n];
	vec[18] = r->G[n];
	vec[19] = r->gsk[n];
	vec[20] = r->xtra[n];
}	

int Minitialize_repeat(Mstate *M, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	mclock_t buffer, input[Mnvars], init[Mnvars];

	std::cout << "Reading in initial conditions from file " << name << "\n";
	while ( (ifs >> buffer) && (count < Mnvars) )
	{
		if((KO == 1) && (count == 18 || count == 19 || count == 40 || count == 41)) // CRY1KO
			input[count] = 0.0;
		else if((KO == 2) && (count == 14 || count == 15 || count == 43 || count == 44)) // CRY2KO
			input[count] = 0.0;
//		else if((KO == 8) && (count == 14 || count == 15 || count == 18 || count == 19 || count == 40 || count == 41 || count == 43 || count == 44)) // CRY1/2DKO
//			input[count] = 0.0;
		else if((KO == 3) && (count == 10 || count == 11 || count == 46)) // PER1KO
			input[count] = 0.0;
		else if((KO == 7) && (count == 12 || count == 13 || count == 66)) // PER2KO
			input[count] = 0.0;
		else if((KO == 9) && (count == 20 || count == 21 || count == 24)) // BMALKO
			input[count] = 0.0;
		else if((KO == 4) && (count == 179 || count == 180 || count == 181 || count == 182)) // VIPKO
			input[count] = 0.0;
		else if((KO == 5) && (count == 185)) // cAMPKO
			input[count] = 0.0;
		else if((KO == 6) && (count == 16 || count == 17 || count ==  27 || count ==  28 || count ==  29 || count ==  30 || count ==  31 || count ==  32 || count ==  33 || count ==  34)) // REVERBKO
			input[count] = 0.0;
		else
			input[count] = buffer;
		count++;
	}

	if (count < Mnvars)
	{
		std::cout << "\nWARNING: default initialization file contained only " << count << " values but there are " << Mnvars << " states. Remaining states have been initialized to zero.\n";
		for (int i=count+1; i<Mnvars ;i++)
			input[i]=0;
	}

	for(int n = 0; n < ncells; n++)
	{

		for(int i=0; i < Mnvars; i++)
		{
//			if( MRANDOMINIT && (i == 48 || i == 68) ) // perturb initial conditions
			if( MRANDOMINIT ) // perturb initial conditions
			{
				if(input[i] >= 0)
					init[i] = fmax(randn(input[i], input[i]*MISD),0.0);
				else
					init[i] = randn(input[i], fabs(input[i]*MISD));
			}
			else
				init[i] = input[i];
		}

		vec2Mstate(init,M,n);
	}
	
	std::cout << "\nSuccessfully set " << count << " M initial conditions.\n";
			
	ifs.close();

	return 0;
}

int Minitialize(Mstate *M, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	mclock_t buffer, input[Mnvars];

	std::cout << "Reading in initial conditions from file " << name << "\n";
	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Mnvars; i++)
		{
			if(ifs >> buffer)
				count++;

			//Check for knockouts
			if(
				((KO == 1) && (i == 18 || i == 19 || i == 40 || i == 41)) || /* CRY1KO */
				((KO == 2) && (i == 14 || i == 15 || i == 43 || i == 44)) ||  /* CRY2KO */
//				((KO == 8) && (i == 14 || i == 15 || i == 18 || i == 19 || i == 40 || i == 41 || i == 43 || i == 44)) ||  /* CRY1/2DKO */
				((KO == 3) && (i == 10 || i == 11 || i == 46)) || /* PER1KO */
				((KO == 7) && (i == 12 || i == 13 || i == 66)) || /* PER2KO */
				((KO == 9) && (i == 20 || i == 21 || i == 24)) || /* BMALKO */
				((KO == 4) && (i == 179 || i == 180 || i == 181 || i == 182)) || /* VIPKO */
				((KO == 5) && (i == 185)) || /* cAMPKO */
				((KO == 6) && (i == 16 || i == 17 || i ==  27 || i ==  28 || i ==  29 || i ==  30 || i ==  31 || i ==  32 || i ==  33 || i ==  34)) ) /* REVERBKO */
				input[i] = 0.0;
			else
				input[i] = buffer;
		}

		if(MRANDOMINIT) // perturb initial conditions
		{
			for(int i=0; i < Mnvars; i++)
				input[i] = std::max(randn(input[i], input[i]*MISD), 0.0);
		}

		vec2Mstate(input,M,n);
	}
	
	if (count != Mnvars*ncells)
	{
		std::cout << "\nMinit file did not contain the correct number of records (" << Mnvars << " variables, " << ncells << " cells)\n";
		return 1;
	}
	else 
		std::cout << "\nSuccessfully set M initial conditions.\n";

	ifs.close();

	return 0;
}

int Mcheck_init(Mstate *M, Mparameters *p)
{
	for(int n=0; n<ncells; n++)
	{
		if(M->GB[n]>1.0)
			M->GB[n] = 1.0/M->GB[n];
		if(M->GB[n]+M->GBR[n]>1.0)
		{
//			printf("Warning: GB[%d] = %lf, GBR[%d] = %lf (sum > 1), ",n,M->GB[n],n,M->GBR[n]);
			M->GBR[n] = 1.0-M->GB[n];
//			printf("setting GBR[%d] = 1-GB[%d] = %lf\n",n,n,M->GBR[n]);
		}
		if(M->GBb[n]>1.0)
			M->GBb[n] = 1.0/M->GBb[n];
		if(M->GBb[n]+M->GBRb[n]>1.0)
		{
//			printf("Warning: GBb[%d] = %lf, GBRb[%d] = %lf (sum > 1), ",n,M->GBb[n],n,M->GBRb[n]);
			M->GBRb[n] = 1.0-M->GBb[n];
//			printf("setting GBRb[%d] = 1-GBb[%d] = %lf\n",n,n,M->GBRb[n]);
		}

		/* Check Kinases */
		mclock_t Kinases = M->x30300[n]+M->x40300[n]+M->x40310[n]+M->x40311[n]+M->x41300[n]+M->x41310[n]+M->x41311[n]+M->x42300[n]+M->x42310[n]+M->x42311[n]+M->x50300[n]+M->x50310[n]+M->x50311[n]+M->x51300[n]+M->x51310[n]+M->x51311[n]+M->x52300[n]+M->x52310[n]+M->x52311[n]+M->x60300[n]+M->x60310[n]+M->x60311[n]+M->x61300[n]+M->x61310[n]+M->x61311[n]+M->x62300[n]+M->x62310[n]+M->x62311[n];
		//Unbound CK1 in cytoplasm (x00100)
		mclock_t Ck1 = p->Ct[n] - (Kinases + M->x00110[n]+M->x10100[n]+M->x20100[n]+M->x20110[n]+M->x20111[n]+M->x21100[n]+M->x21110[n]+M->x21111[n]+M->x22100[n]+M->x22110[n]+M->x22111[n]+M->x30100[n]+M->x40100[n]+M->x40110[n]+M->x40111[n]+M->x41100[n]+M->x41110[n]+M->x41111[n]+M->x42100[n]+M->x42110[n]+M->x42111[n]+M->x50100[n]+M->x50110[n]+M->x50111[n]+M->x51100[n]+M->x51110[n]+M->x51111[n]+M->x52100[n]+M->x52110[n]+M->x52111[n]+M->x60100[n]+M->x60110[n]+M->x60111[n]+M->x61100[n]+M->x61110[n]+M->x61111[n]+M->x62100[n]+M->x62110[n]+M->x62111[n]);
		//Unbound GSK3B in cytoplasm (x00200)
		mclock_t Gsk = p->Gt[n] - (Kinases + M->cyrevg[n]+M->revng[n]+M->cyrevgp[n]+M->revngp[n]+M->x00210[n]+M->x30200[n]+M->x40200[n]+M->x40210[n]+M->x40211[n]+M->x41200[n]+M->x41210[n]+M->x41211[n]+M->x42200[n]+M->x42210[n]+M->x42211[n]+M->x50200[n]+M->x50210[n]+M->x50211[n]+M->x51200[n]+M->x51210[n]+M->x51211[n]+M->x52200[n]+M->x52210[n]+M->x52211[n]+M->x60200[n]+M->x60210[n]+M->x60211[n]+M->x61200[n]+M->x61210[n]+M->x61211[n]+M->x62200[n]+M->x62210[n]+M->x62211[n]);

		if(Ck1<0.0)
		{
			std::cout << "Ck1[" << n << "] = " << Ck1 << " < 0, error in initial conditions\n";
			return 1;
		}
		if(Gsk<0.0)
		{
			std::cout << "Gsk[" << n << "] = " << Gsk << " < 0, error in initial conditions\n";
			return 1;
		}

		//Check VPAC2R
		mclock_t V00 = p->Vt[n]-M->V10[n]-M->V11[n]-M->V12[n]-M->V01[n]-M->V02[n]; // free receptor
		if(V00<0.0)
		{
			std::cout << "V00[" << n << "] = " << V00 << " < 0, error in initial conditions\n";
//			M->V01[n] = M->V01[n]+V00-.0001;
//			printf("V01[%d] changed from %lf to %lf\n",n,M->V01[n]-V00+.0001,M->V01[n]);
		//	return 1;
		}

	}
	return 0;
}
	
int Mpinitialize_repeat(Mparameters *p, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	mclock_t buffer, input[Mnparams], init[Mnparams];

        std::cout << "Reading in parameters from file " << name << "\n";
	while ( (ifs >> buffer) && (count < Mnparams) )
	{
		input[count] = buffer;
		count++;
	}

	if (count < Mnparams)
	{
		std::cout << "\nWARNING: parameter file contained only " << count << " values but there are " << Mnparams << " parameters. Remaining parameters have been set to zero.\n";
		for (int i=count+1; i<Mnparams ;i++)
			input[i]=0;
	}

	srand (time(NULL));
	for(int n = 0; n < ncells; n++)
	{
		for(int i=0; i < Mnparams; i++)
		{
//			if( MRANDOMPARAMS & ( (i==0) | (i==1) | (i==80) | (i==81) | (i==58) | (i==59)) ) // perturb per1 and per2 transcription and degradation rates
//			if( MRANDOMPARAMS & ( (i==0) | (i==1) | (i==80) | (i==81) ) ) // perturb per1 and per2 transcription rates
			if( MRANDOMPARAMS ) // perturb all parameters
				init[i] = randn(input[i], input[i]*MPSD);
			else
				init[i] = input[i];
			if(
				((KO == 1) && (i == 2)) || /* CRY1KO */
				((KO == 2) && (i == 3)) ||  /* CRY2KO */
				((KO == 8) && (i == 2 || i == 3)) ||  /* CRY1/2DKO */
				((KO == 3) && (i == 0 || i == 80)) || /* PER1KO */
				((KO == 4) && (i == 82)) || /* VIPKO */
				((KO == 5) && (i == 76)) || /* cAMPKO */
				((KO == 6) && (i == 5)) || /* REVERBKO */
				((KO == 9) && (i == 4)) || /* BMALKO */
				((KO == 7) && (i == 1 || i == 81)) ) /* PER2KO */
				init[i] = 0.0;
			else if( KO == 10 ) {  /* Afterhours */
 				if( i == 55 )
					init[i] = init[i]*0.45;
				else if( i == 56 ) 
					init[i] = init[i]*0.88;
			}				
		}

		vec2Mparams(init,p,n);
	}
	
	std::cout << "\nSuccessfully set " << count << " M parameters.\n";
			
	ifs.close();

	return 0;
}

int Mpinitialize(Mparameters *p, const char *name)
{
	std::ifstream ifs(name);
	int count = 0;
	mclock_t buffer, input[Mnparams];

	std::cout << "Reading in parameters from file " << name << "\n";

	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Mnparams; i++)
		{
			if(ifs >> buffer)
				count++;
			if(
				((KO == 1) && (i == 2)) || /* CRY1KO */
				((KO == 2) && (i == 3)) ||  /* CRY2KO */
				((KO == 8) && (i == 2 || i == 3)) ||  /* CRY1/2DKO */
				((KO == 3) && (i == 0 || i == 80)) || /* PER1KO */
				((KO == 4) && (i == 82)) || /* VIPKO */
				((KO == 5) && (i == 76)) || /* cAMPKO */
				((KO == 6) && (i == 5)) || /* REVERBKO */
				((KO == 9) && (i == 4)) || /* BMALKO */
				((KO == 7) && (i == 1 || i == 81)) ) /* PER2KO */
				input[i] = 0.0;
			else if( KO == 10 ) {  /* Afterhours */
 				if( i == 55 )
					input[i] = buffer*0.45;
				else if( i == 56 ) 
					input[i] = buffer*0.88;
				else
					input[i] = buffer;
			}				
			else if(n >= 206)
				input[i] = TSCALE*buffer;
			else
				input[i] = buffer;

		}

/*		if(MRANDOMPARAMS) // perturb all parameters
		{
			for(int i=0; i < Mnparams; i++)
				input[i] = std::max(randn(input[i], input[i]*MPSD), 0.0);
		}
*/
		vec2Mparams(input,p,n);

	}
	
	if (count != Mnparams*ncells)
	{
		std::cout << "\nMparameters file did not contain the correct number of records (" << Mnparams << " variables, " << ncells << " cells)\n";
		return 1;
	}
	else 
		std::cout << "\nSuccessfully set M parameters.\n";

	ifs.close();

	return 0;
}



int write_Mresult(mclock_t *output, std::ofstream& outfile)
{
	outfile.precision(std::numeric_limits<double>::digits10 + 2);

	for (int i=0; i<Mrecsteps; i++) {
		outfile << (double)i*Mdt*Mrecord;
		for (int j=0; j<ncells; j++) {
			outfile << " " << output[j+i*ncells];
		}
		outfile << "\n";
	}
	return 0;
}

int write_Mfinal(Mstate *M, std::ofstream& outfile)
{
	mclock_t *output;
	output = (mclock_t*)malloc(Mnvars*sizeof(mclock_t));

	for(int n = 0; n<ncells; n++)
	{
		Mstate2vec(M, n, output);

		for(int i = 0; i<Mnvars; i++)
		{
			outfile << output[i] << " ";
		}
		outfile << "\n";
	}

	return 0;
}

int write_Mparams(Mparameters *p, const char *name)
{
	int i, j;
	char pfilename[50];
  	sprintf(pfilename,"./Mparameters/%s",name);

	std::ofstream ofs(pfilename);

	if (ofs.fail())
		return 1;

	mclock_t output[Mnparams];

	for (int n=0; n < ncells; n++)
	{
		/* write the parameters from the nth cell to a vector */
		Mparams2vec(p,n,output);

		/* write the vector of parameters to the file */
		for(i = 0; i < Mnparams; i++)
		{
			ofs << output[i] << " ";
		}
		ofs << "\n";

	}
	ofs.close();
	return 0;
}
