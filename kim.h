#ifndef KIM_H_
#define KIM_H_

#define Mdt 0.005 /* step size (in hours) */
#define Mnvars 188 //number of variables
#define Mnparams 85 //number of parameters

#define Mrecord 100 // record data to file every record time steps
#define Mnstep 86400
//#define Mrecsteps 1 // number of result steps to record
#define Mrecsteps ((int)(Mnstep/Mrecord)+1) // number of result steps to record

/*struct Mstate{
	double GR[ncells], G[ncells], GrR[ncells], Gr[ncells], GcR[ncells], Gc[ncells], GBR[ncells], GB[ncells], GBRb[ncells], GBb[ncells], MnPo[ncells], McPo[ncells], MnPt[ncells], McPt[ncells], MnRt[ncells], McRt[ncells], MnRev[ncells], McRev[ncells], MnRo[ncells], McRo[ncells], MnB[ncells], McB[ncells], MnNp[ncells], McNp[ncells], B[ncells], Cl[ncells], BC[ncells], cyrev[ncells], revn[ncells], cyrevg[ncells], revng[ncells], cyrevgp[ncells], revngp[ncells], cyrevp[ncells], revnp[ncells], gto[ncells], x00001[ncells], x00011[ncells], x00100[ncells], x00110[ncells], x00200[ncells], x00210[ncells], x01000[ncells], x01010[ncells], x01011[ncells], x02000[ncells], x02010[ncells], x02011[ncells], x10000[ncells], x10100[ncells], x20000[ncells], x20010[ncells], x20011[ncells], x20100[ncells], x20110[ncells], x20111[ncells], x21000[ncells], x21010[ncells], x21011[ncells], x21100[ncells], x21110[ncells], x21111[ncells], x22000[ncells], x22010[ncells], x22011[ncells], x22100[ncells], x22110[ncells], x22111[ncells], x30000[ncells], x30100[ncells], x30200[ncells], x30300[ncells], x40000[ncells], x40010[ncells], x40011[ncells], x40100[ncells], x40110[ncells], x40111[ncells], x40200[ncells], x40210[ncells], x40211[ncells], x40300[ncells], x40310[ncells], x40311[ncells], x41000[ncells], x41010[ncells], x41011[ncells], x41100[ncells], x41110[ncells], x41111[ncells], x41200[ncells], x41210[ncells], x41211[ncells], x41300[ncells], x41310[ncells], x41311[ncells], x42000[ncells], x42010[ncells], x42011[ncells], x42100[ncells], x42110[ncells], x42111[ncells], x42200[ncells], x42210[ncells], x42211[ncells], x42300[ncells], x42310[ncells], x42311[ncells], x50000[ncells], x50010[ncells], x50011[ncells], x50100[ncells], x50110[ncells], x50111[ncells], x50200[ncells], x50210[ncells], x50211[ncells], x50300[ncells], x50310[ncells], x50311[ncells], x51000[ncells], x51010[ncells], x51011[ncells], x51100[ncells], x51110[ncells], x51111[ncells], x51200[ncells], x51210[ncells], x51211[ncells], x51300[ncells], x51310[ncells], x51311[ncells], x52000[ncells], x52010[ncells], x52011[ncells], x52100[ncells], x52110[ncells], x52111[ncells], x52200[ncells], x52210[ncells], x52211[ncells], x52300[ncells], x52310[ncells], x52311[ncells], x60000[ncells], x60010[ncells], x60011[ncells], x60100[ncells], x60110[ncells], x60111[ncells], x60200[ncells], x60210[ncells], x60211[ncells], x60300[ncells], x60310[ncells], x60311[ncells], x61000[ncells], x61010[ncells], x61011[ncells], x61100[ncells], x61110[ncells], x61111[ncells], x61200[ncells], x61210[ncells], x61211[ncells], x61300[ncells], x61310[ncells], x61311[ncells], x62000[ncells], x62010[ncells], x62011[ncells], x62100[ncells], x62110[ncells], x62111[ncells], x62200[ncells], x62210[ncells], x62211[ncells], x62300[ncells], x62310[ncells], x62311[ncells], v[ncells];
};

struct Mparameters{
	double trPo[ncells], trPt[ncells], trRo[ncells], trRt[ncells], trB[ncells], trRev[ncells], trNp[ncells], tlp[ncells], tlr[ncells], tlb[ncells], tlrev[ncells], tlc[ncells], tlnp[ncells], agp[ncells], dg[ncells], ac[ncells], dc[ncells], ar[ncells], dr[ncells], cbin[ncells], uncbin[ncells], bbin[ncells], unbbin[ncells], cbbin[ncells], uncbbin[ncells], ag[ncells], bin[ncells], unbin[ncells], binrev[ncells], unbinrev[ncells], binr[ncells], unbinr[ncells], binc[ncells], unbinc[ncells], binrevb[ncells], unbinrevb[ncells], tmc[ncells], tmcrev[ncells], nl[ncells], ne[ncells], nlrev[ncells], nerev[ncells], lne[ncells], nlbc[ncells], hoo[ncells], hto[ncells], phos[ncells], lono[ncells], lont[ncells], lta[ncells], ltb[ncells], trgto[ncells], ugto[ncells], Nf[ncells], up[ncells], uro[ncells], urt[ncells], umNp[ncells], umPo[ncells], umPt[ncells], umRo[ncells], umRt[ncells], ub[ncells], uc[ncells], ubc[ncells], upu[ncells], urev[ncells], uprev[ncells], umB[ncells], umRev[ncells], uv[ncells];
};
*/

struct Mresult{
	double pom[ncells*Mrecsteps], ptm[ncells*Mrecsteps], rom[ncells*Mrecsteps], rtm[ncells*Mrecsteps], bmm[ncells*Mrecsteps], rvm[ncells*Mrecsteps], npm[ncells*Mrecsteps], pot[ncells*Mrecsteps], ptt[ncells*Mrecsteps], rot[ncells*Mrecsteps], rtt[ncells*Mrecsteps], bmt[ncells*Mrecsteps], clt[ncells*Mrecsteps], clct[ncells*Mrecsteps], clnt[ncells*Mrecsteps], revt[ncells*Mrecsteps], cre[ncells*Mrecsteps], vip[ncells*Mrecsteps], G[ncells*Mrecsteps], BC[ncells*Mrecsteps], xtra[ncells*Mrecsteps];
};

#endif
