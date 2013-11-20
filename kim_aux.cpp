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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "scn.h"
#include "kim.h"
#include "parameters.h"
#include "utils.h"
#include <algorithm>

void vec2Mstate(double *vec, Mstate *M, int n)
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
	M->x00100[n] = vec[38];
	M->x00110[n] = vec[39];
	M->x00200[n] = vec[40];
	M->x00210[n] = vec[41];
	M->x01000[n] = vec[42];
	M->x01010[n] = vec[43];
	M->x01011[n] = vec[44];
	M->x02000[n] = vec[45];
	M->x02010[n] = vec[46];
	M->x02011[n] = vec[47];
	M->x10000[n] = vec[48];
	M->x10100[n] = vec[49];
	M->x20000[n] = vec[50];
	M->x20010[n] = vec[51];
	M->x20011[n] = vec[52];
	M->x20100[n] = vec[53];
	M->x20110[n] = vec[54];
	M->x20111[n] = vec[55];
	M->x21000[n] = vec[56];
	M->x21010[n] = vec[57];
	M->x21011[n] = vec[58];
	M->x21100[n] = vec[59];
	M->x21110[n] = vec[60];
	M->x21111[n] = vec[61];
	M->x22000[n] = vec[62];
	M->x22010[n] = vec[63];
	M->x22011[n] = vec[64];
	M->x22100[n] = vec[65];
	M->x22110[n] = vec[66];
	M->x22111[n] = vec[67];
	M->x30000[n] = vec[68];
	M->x30100[n] = vec[69];
	M->x30200[n] = vec[70];
	M->x30300[n] = vec[71];
	M->x40000[n] = vec[72];
	M->x40010[n] = vec[73];
	M->x40011[n] = vec[74];
	M->x40100[n] = vec[75];
	M->x40110[n] = vec[76];
	M->x40111[n] = vec[77];
	M->x40200[n] = vec[78];
	M->x40210[n] = vec[79];
	M->x40211[n] = vec[80];
	M->x40300[n] = vec[81];
	M->x40310[n] = vec[82];
	M->x40311[n] = vec[83];
	M->x41000[n] = vec[84];
	M->x41010[n] = vec[85];
	M->x41011[n] = vec[86];
	M->x41100[n] = vec[87];
	M->x41110[n] = vec[88];
	M->x41111[n] = vec[89];
	M->x41200[n] = vec[90];
	M->x41210[n] = vec[91];
	M->x41211[n] = vec[92];
	M->x41300[n] = vec[93];
	M->x41310[n] = vec[94];
	M->x41311[n] = vec[95];
	M->x42000[n] = vec[96];
	M->x42010[n] = vec[97];
	M->x42011[n] = vec[98];
	M->x42100[n] = vec[99];
	M->x42110[n] = vec[100];
	M->x42111[n] = vec[101];
	M->x42200[n] = vec[102];
	M->x42210[n] = vec[103];
	M->x42211[n] = vec[104];
	M->x42300[n] = vec[105];
	M->x42310[n] = vec[106];
	M->x42311[n] = vec[107];
	M->x50000[n] = vec[108];
	M->x50010[n] = vec[109];
	M->x50011[n] = vec[110];
	M->x50100[n] = vec[111];
	M->x50110[n] = vec[112];
	M->x50111[n] = vec[113];
	M->x50200[n] = vec[114];
	M->x50210[n] = vec[115];
	M->x50211[n] = vec[116];
	M->x50300[n] = vec[117];
	M->x50310[n] = vec[118];
	M->x50311[n] = vec[119];
	M->x51000[n] = vec[120];
	M->x51010[n] = vec[121];
	M->x51011[n] = vec[122];
	M->x51100[n] = vec[123];
	M->x51110[n] = vec[124];
	M->x51111[n] = vec[125];
	M->x51200[n] = vec[126];
	M->x51210[n] = vec[127];
	M->x51211[n] = vec[128];
	M->x51300[n] = vec[129];
	M->x51310[n] = vec[130];
	M->x51311[n] = vec[131];
	M->x52000[n] = vec[132];
	M->x52010[n] = vec[133];
	M->x52011[n] = vec[134];
	M->x52100[n] = vec[135];
	M->x52110[n] = vec[136];
	M->x52111[n] = vec[137];
	M->x52200[n] = vec[138];
	M->x52210[n] = vec[139];
	M->x52211[n] = vec[140];
	M->x52300[n] = vec[141];
	M->x52310[n] = vec[142];
	M->x52311[n] = vec[143];
	M->x60000[n] = vec[144];
	M->x60010[n] = vec[145];
	M->x60011[n] = vec[146];
	M->x60100[n] = vec[147];
	M->x60110[n] = vec[148];
	M->x60111[n] = vec[149];
	M->x60200[n] = vec[150];
	M->x60210[n] = vec[151];
	M->x60211[n] = vec[152];
	M->x60300[n] = vec[153];
	M->x60310[n] = vec[154];
	M->x60311[n] = vec[155];
	M->x61000[n] = vec[156];
	M->x61010[n] = vec[157];
	M->x61011[n] = vec[158];
	M->x61100[n] = vec[159];
	M->x61110[n] = vec[160];
	M->x61111[n] = vec[161];
	M->x61200[n] = vec[162];
	M->x61210[n] = vec[163];
	M->x61211[n] = vec[164];
	M->x61300[n] = vec[165];
	M->x61310[n] = vec[166];
	M->x61311[n] = vec[167];
	M->x62000[n] = vec[168];
	M->x62010[n] = vec[169];
	M->x62011[n] = vec[170];
	M->x62100[n] = vec[171];
	M->x62110[n] = vec[172];
	M->x62111[n] = vec[173];
	M->x62200[n] = vec[174];
	M->x62210[n] = vec[175];
	M->x62211[n] = vec[176];
	M->x62300[n] = vec[177];
	M->x62310[n] = vec[178];
	M->x62311[n] = vec[179];
	M->ltn[n] = vec[180];
	M->vip[n] = vec[181];
	M->V10[n] = vec[182];
	M->V11[n] = vec[183];
	M->V12[n] = vec[184];
	M->V01[n] = vec[185];
	M->V02[n] = vec[186];
	M->cAMP[n] = vec[187];
	M->CREB[n] = vec[188];
	M->CRE[n] = vec[189];
}

void Mstate2vec(Mstate *M, int n, double *vec)
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
	vec[38] = M->x00100[n];
	vec[39] = M->x00110[n];
	vec[40] = M->x00200[n];
	vec[41] = M->x00210[n];
	vec[42] = M->x01000[n];
	vec[43] = M->x01010[n];
	vec[44] = M->x01011[n];
	vec[45] = M->x02000[n];
	vec[46] = M->x02010[n];
	vec[47] = M->x02011[n];
	vec[48] = M->x10000[n];
	vec[49] = M->x10100[n];
	vec[50] = M->x20000[n];
	vec[51] = M->x20010[n];
	vec[52] = M->x20011[n];
	vec[53] = M->x20100[n];
	vec[54] = M->x20110[n];
	vec[55] = M->x20111[n];
	vec[56] = M->x21000[n];
	vec[57] = M->x21010[n];
	vec[58] = M->x21011[n];
	vec[59] = M->x21100[n];
	vec[60] = M->x21110[n];
	vec[61] = M->x21111[n];
	vec[62] = M->x22000[n];
	vec[63] = M->x22010[n];
	vec[64] = M->x22011[n];
	vec[65] = M->x22100[n];
	vec[66] = M->x22110[n];
	vec[67] = M->x22111[n];
	vec[68] = M->x30000[n];
	vec[69] = M->x30100[n];
	vec[70] = M->x30200[n];
	vec[71] = M->x30300[n];
	vec[72] = M->x40000[n];
	vec[73] = M->x40010[n];
	vec[74] = M->x40011[n];
	vec[75] = M->x40100[n];
	vec[76] = M->x40110[n];
	vec[77] = M->x40111[n];
	vec[78] = M->x40200[n];
	vec[79] = M->x40210[n];
	vec[80] = M->x40211[n];
	vec[81] = M->x40300[n];
	vec[82] = M->x40310[n];
	vec[83] = M->x40311[n];
	vec[84] = M->x41000[n];
	vec[85] = M->x41010[n];
	vec[86] = M->x41011[n];
	vec[87] = M->x41100[n];
	vec[88] = M->x41110[n];
	vec[89] = M->x41111[n];
	vec[90] = M->x41200[n];
	vec[91] = M->x41210[n];
	vec[92] = M->x41211[n];
	vec[93] = M->x41300[n];
	vec[94] = M->x41310[n];
	vec[95] = M->x41311[n];
	vec[96] = M->x42000[n];
	vec[97] = M->x42010[n];
	vec[98] = M->x42011[n];
	vec[99] = M->x42100[n];
	vec[100] = M->x42110[n];
	vec[101] = M->x42111[n];
	vec[102] = M->x42200[n];
	vec[103] = M->x42210[n];
	vec[104] = M->x42211[n];
	vec[105] = M->x42300[n];
	vec[106] = M->x42310[n];
	vec[107] = M->x42311[n];
	vec[108] = M->x50000[n];
	vec[109] = M->x50010[n];
	vec[110] = M->x50011[n];
	vec[111] = M->x50100[n];
	vec[112] = M->x50110[n];
	vec[113] = M->x50111[n];
	vec[114] = M->x50200[n];
	vec[115] = M->x50210[n];
	vec[116] = M->x50211[n];
	vec[117] = M->x50300[n];
	vec[118] = M->x50310[n];
	vec[119] = M->x50311[n];
	vec[120] = M->x51000[n];
	vec[121] = M->x51010[n];
	vec[122] = M->x51011[n];
	vec[123] = M->x51100[n];
	vec[124] = M->x51110[n];
	vec[125] = M->x51111[n];
	vec[126] = M->x51200[n];
	vec[127] = M->x51210[n];
	vec[128] = M->x51211[n];
	vec[129] = M->x51300[n];
	vec[130] = M->x51310[n];
	vec[131] = M->x51311[n];
	vec[132] = M->x52000[n];
	vec[133] = M->x52010[n];
	vec[134] = M->x52011[n];
	vec[135] = M->x52100[n];
	vec[136] = M->x52110[n];
	vec[137] = M->x52111[n];
	vec[138] = M->x52200[n];
	vec[139] = M->x52210[n];
	vec[140] = M->x52211[n];
	vec[141] = M->x52300[n];
	vec[142] = M->x52310[n];
	vec[143] = M->x52311[n];
	vec[144] = M->x60000[n];
	vec[145] = M->x60010[n];
	vec[146] = M->x60011[n];
	vec[147] = M->x60100[n];
	vec[148] = M->x60110[n];
	vec[149] = M->x60111[n];
	vec[150] = M->x60200[n];
	vec[151] = M->x60210[n];
	vec[152] = M->x60211[n];
	vec[153] = M->x60300[n];
	vec[154] = M->x60310[n];
	vec[155] = M->x60311[n];
	vec[156] = M->x61000[n];
	vec[157] = M->x61010[n];
	vec[158] = M->x61011[n];
	vec[159] = M->x61100[n];
	vec[160] = M->x61110[n];
	vec[161] = M->x61111[n];
	vec[162] = M->x61200[n];
	vec[163] = M->x61210[n];
	vec[164] = M->x61211[n];
	vec[165] = M->x61300[n];
	vec[166] = M->x61310[n];
	vec[167] = M->x61311[n];
	vec[168] = M->x62000[n];
	vec[169] = M->x62010[n];
	vec[170] = M->x62011[n];
	vec[171] = M->x62100[n];
	vec[172] = M->x62110[n];
	vec[173] = M->x62111[n];
	vec[174] = M->x62200[n];
	vec[175] = M->x62210[n];
	vec[176] = M->x62211[n];
	vec[177] = M->x62300[n];
	vec[178] = M->x62310[n];
	vec[179] = M->x62311[n];
	vec[180] = M->ltn[n];
	vec[181] = M->vip[n];
	vec[182] = M->V10[n];
	vec[183] = M->V11[n];
	vec[184] = M->V12[n];
	vec[185] = M->V01[n];
	vec[186] = M->V02[n];
	vec[187] = M->cAMP[n];
	vec[188] = M->CREB[n];
	vec[189] = M->CRE[n];
}

void vec2Mparams(double *vec, Mparameters *p, int n)
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
}

void Mparams2vec(Mparameters *p, int n, double *vec)
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
}

void vec2Mresult(double *vec, Mresult *r, int n)
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
	r->BC[n] = vec[19];
}

void Mresult2vec(Mresult *r, int n, double *vec)
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
	vec[19] = r->BC[n];
}	

int Minitialize_repeat(Mstate *M, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Mnvars], init[Mnvars];

        printf("Reading in initial conditions from file %s\n", name);
	infile = open_file(name, "r");
	while ( (fscanf(infile, "%lf,", &buffer) != EOF) && (count < Mnvars) )
	{
		input[count] = buffer;
//		printf("%lf\t",input[count]);
		count++;
	}

	if (count < Mnvars)
	{
		printf("\nWARNING: default initialization file contained only %d values but there are %d states. Remaining states have been initialized to zero.\n",count,Mnvars);
		for (int i=count+1; i<Mnvars ;i++)
			input[i]=0;
	}

	for(int n = 0; n < ncells; n++)
	{

		for(int i=0; i < Mnvars; i++)
		{
			if(MRANDOMINIT) // perturb initial conditions
			{
				if(input[i] >= 0)
					init[i] = fmax(randn(input[i], input[i]*MISD),0.0);
				else
					init[i] = randn(input[i], fabs(input[i]*MISD));
			}
			else
				init[i] = input[i];
			//printf("%lf\t",init[189]);
//			printf("%lf\t", init[37] + init[44] + init[47] + init[52] + init[55] + init[58] + init[61] + init[64] + init[67] + init[74] + init[77] + init[80] + init[83] + s->x41011[n] + s->x41111[n] + s->x41211[n] + s->x41311[n] + s->x42011[n] + s->x42111[n] + s->x42211[n] + s->x42311[n] + s->x50011[n] + s->x50111[n] + s->x50211[n] + s->x50311[n] + s->x51011[n] + s->x51111[n] + s->x51211[n] + s->x51311[n] + s->x52011[n] + s->x52111[n] + s->x52211[n] + s->x52311[n] + s->x60011[n] + s->x60111[n] + s->x60211[n] + s->x60311[n] + s->x61011[n] + s->x61111[n] + s->x61211[n] + s->x61311[n] + s->x62011[n] + s->x62111[n] + s->x62211[n] + s->x62311[n]);
		}

		vec2Mstate(init,M,n);
	}
	
	printf("\nSuccessfully set %d M initial conditions.\n", count);
			
	fclose(infile);

	return 0;
}

int Minitialize(Mstate *M, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Mnvars];

        printf("Reading in initial conditions from file %s\n", name);
	infile = open_file(name, "r");
	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Mnvars; i++)
		{
			if(fscanf(infile, "%lf,", &buffer) != EOF)
				count++;
//			if(i == 16 || i == 17 || i ==  27 || i ==  28 || i ==  29 || i ==  30 || i ==  31 || i ==  32 || i ==  33 || i ==  34) // REVERBKO
//			if(i == 18 || i == 19 || i == 42 || i == 43) // CRY1KO
//			if(i == 14 || i == 15 || i == 45 || i == 46) // CRY2KO
//			if(i == 181 || i == 182 || i == 183 || i == 184) // VIPKO
//			if(i == 187) // cAMPKO
			if(i == -1) // no KO
				input[i] = 0.0;
			else
				input[i] = buffer;
		}

/*		if(MRANDOMINIT) // perturb initial conditions
		{
			for(int i=0; i < Mnvars; i++)
				input[i] = std::max(randn(input[i], input[i]*MISD), 0.0);
		}
*/
		vec2Mstate(input,M,n);

//		printf("%d\t%lf\n",n,M->G[n]);
	}
	
	if (count != Mnvars*ncells)
	{
		printf("\nMinit file did not contain the correct number of records (%d variables, %d cells)\n",Mnvars,ncells);
		return 1;
	}
	else 
		printf("\nSuccessfully set M initial conditions.\n");

	fclose(infile);

	return 0;
}

int Mpinitialize_repeat(Mparameters *p, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Mnparams], init[Mnparams];

        printf("Reading in parameters from file %s\n", name);
	infile = open_file(name, "r");
	while ( (fscanf(infile, "%lf,", &buffer) != EOF) && (count < Mnparams) )
	{
		input[count] = buffer;
//		printf("%lf\t",input[count]);
		count++;
	}

	if (count < Mnparams)
	{
		printf("\nWARNING: parameter file contained only %d values but there are %d parameters. Remaining parameters have been set to zero.\n",count,Mnparams);
		for (int i=count+1; i<Mnparams ;i++)
			input[i]=0;
	}

	for(int n = 0; n < ncells; n++)
	{

		for(int i=0; i < Mnparams; i++)
		{
			if( MRANDOMPARAMS & ( (i==0) | (i==1) ) ) // perturb per1 and per2 transcription rate // perturb all parameters
				init[i] = randn(input[i], input[i]*MPSD);
			else
				init[i] = input[i];
//			printf("%lf\t",init[0]);
		}

		vec2Mparams(init,p,n);
	}
	
	printf("\nSuccessfully set %d M parameters.\n", count);
			
	fclose(infile);

	return 0;
}

int Mpinitialize(Mparameters *p, const char *name)
{
	FILE *infile;
	int count = 0;
	double buffer, input[Mnparams];

        printf("Reading in parameters from file %s\n", name);
	infile = open_file(name, "r");

	for(int n = 0; n < ncells; n++)
	{
		for(int i = 0; i < Mnparams; i++)
		{
			if(fscanf(infile, "%lf,", &buffer) != EOF)
				count++;
//			if(i == 2) // CRY1KO
//			if(i == 3) // CRY2KO
//			if(i == 82) // VIPKO
//			if(i == 76) // cAMPKO
//			if(i == 5) // REVERBKO
			if(i == -1) // no KO
				input[i] = 0.0;
			else
				input[i] = buffer;
			
//			if(n==ncells-1)
//				printf("%lf\t",input[i]);
		}

/*		if(MRANDOMPARAMS) // perturb all parameters
		{
			for(int i=0; i < Mnparams; i++)
				input[i] = std::max(randn(input[i], input[i]*MPSD), 0.0);
		}
*/
		vec2Mparams(input,p,n);

//		printf("%d\t%lf\n",n,M->trPo[n]);
	}
	
	if (count != Mnparams*ncells)
	{
		printf("\nMparameters file did not contain the correct number of records (%d variables, %d cells)\n",Mnparams,ncells);
		return 1;
	}
	else 
		printf("\nSuccessfully set M parameters.\n");

	fclose(infile);

	return 0;
}



int write_Mresult(double *output, FILE *outfile)
{
	for (int i=0; i<Mrecsteps; i++) {
		fprintf(outfile, "%lf", i*Mdt*Mrecord);
		for (int j=0; j<ncells; j++) {
			fprintf(outfile, "\t%.12lf", output[j+i*ncells]);
		}
		fprintf(outfile, "\n");
	}
	return 0;
}

int write_Mfinal(Mstate *M, FILE *outfile)
{
	double *output;
	output = (double*)malloc(Mnvars*sizeof(double));

	for(int n = 0; n<ncells; n++)
	{
		Mstate2vec(M, n, output);

		for(int i = 0; i<Mnvars; i++)
		{
			fprintf(outfile, "%.12lf,",output[i]);
		}
		fprintf(outfile, "\n");
	}

	return 0;
}

/*
int write_Mresult(double *output, FILE *outfile)
{
	for (int i=0; i < Mrecsteps*ncells; i++)
		fprintf(outfile, "\t%.12lf", output[i]);
	fprintf(outfile, "\n");
	
	return 0;
}
*/

int write_Mparams(Mparameters *p, const char *name)
{
	FILE *pfile;
	int i, j;
	char pfilename[50];
  	sprintf(pfilename,"./Mparameters/%s",name);

	pfile=fopen(pfilename, "w");

	if (pfile == NULL)
  	{
    		perror ("The following error occurred");
		return 1;
 	}

	double output[Mnparams];

	for (int n=0; n < ncells; n++)
	{
		/* write the parameters from the nth cell to a vector */
		Mparams2vec(p,n,output);

		/* write the vector of parameters to the file */
		for(i = 0; i < Mnparams; i++)
		{
			fprintf(pfile, "%lf ", output[i]);
		}
		fprintf(pfile,"\n");

	}
	fclose(pfile);
	return 0;
}
