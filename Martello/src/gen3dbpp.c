
/* ======================================================================
      3DBPP PROBLEMS, Silvano Martello, David Pisinger, Daniele Vigo
   ====================================================================== */

/* This code generates instances for the three-dimensional bin-packing 
 * problem and solves them using the 3DBPP algorithm by Martello, Pisinger,
 * Vigo. 
 *
 * A description of the test instances is found in the following papers:
 *
 *   S.Martello, D.Pisinger, D.Vigo (1998)
 *   "An exact algorithm for the three-dimensional bin packing problem"
 *   submitted.
 *
 *   S.Martello, D.Pisinger, D.Vigo (1998)
 *   "The three-dimensional bin packing problem"
 *   to appear in Operations Research.
 *
 * The algorithm prompts for three arguments:
 *   n      The size of the instance, i.e. number of boxes.
 *   bindim The size of the bin, typically 40-100.
 *   type   A value between 1-9 selecting one of the instance types
 *          described in the above papers.
 *
 * Results are written to the file "3dbpp.out".
 * 
 * (c) Copyright 1998,
 *
 *   David Pisinger                        Silvano Martello, Daniele Vigo
 *   DIKU, University of Copenhagen        DEIS, University of Bologna
 *   Universitetsparken 1                  Viale Risorgimento 2
 *   Copenhagen, Denmark                   Bologna, Italy
 * 
 * This code can be used free of charge for research and academic purposes 
 * only. 
 */

#define TESTS          10     /* Number of test to run for each type */
#define MAXITEMS     2001     /* Max number of items plus one */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>


/* ======================================================================
				   macros
   ====================================================================== */

#define srand(x)     srand48x(x)
#define randm(x)     (lrand48x() % (long) (x))
#define rint(a,b)    (randm((b)-(a)+1) + (a))

#define TRUE  1           /* logical variables */
#define FALSE 0

#define VOL(i)                 ((i)->w * (ptype) (i)->h * (i)->d)
#define DIF(i,j)               ((int) ((j) - (i) + 1))


/* ======================================================================
				 type declarations
   ====================================================================== */

typedef short         boolean; /* logical variable      */
typedef short         ntype;   /* number of states,bins */
typedef short         itype;   /* can hold up to W,H,D  */
typedef long          stype;   /* can hold up to W*H*D  */
typedef long          ptype;   /* product multiplication */

typedef int (*funcptr) (const void *, const void *);

/* item record */
typedef struct irec {
  ntype    no;           /* item number */
  itype    w;            /* item x-size */
  itype    h;            /* item y-size */
  itype    d;            /* item z-size */
  itype    x;            /* optimal x-position */
  itype    y;            /* optimal y-position */
  itype    z;            /* optimal z-position */
  ntype    bno;          /* bin number */
  stype    vol;          /* volume of item */
} item;


/* ======================================================================
				global variables
   ====================================================================== */

FILE *trace;


/* =======================================================================
                                random
   ======================================================================= */

/* to generate the same instances as at HP9000 - UNIX, */
/* here follows C-versions of SRAND48, and LRAND48.  */

unsigned long _h48, _l48;

void srand48x(long s)
{
  _h48 = s;
  _l48 = 0x330E;
}

long lrand48x(void)
{
  _h48 = (_h48 * 0xDEECE66D) + (_l48 * 0x5DEEC);
  _l48 = _l48 * 0xE66D + 0xB;
  _h48 = _h48 + (_l48 >> 16);
  _l48 = _l48 & 0xFFFF;
  return (_h48 >> 1);
}


/* ======================================================================
				   timeused
   ====================================================================== */

void timeused(double *time)
{
  static double tstart, tend, tprev;

  if (time == NULL) {
    clock(); /* one extra call to initialize clock */
    tstart = tprev = clock();
  } else {
    tend = clock();
    if (tend < tprev) tstart -= ULONG_MAX; /* wraparound occured */
    tprev = tend;
    *time = (tend-tstart) / CLOCKS_PER_SEC; /* convert to seconds */
  }
}


/* ======================================================================
				specialbin
   ====================================================================== */

void specialbin(item *f, item *l, int W, int H, int D)
{
  item *m, *i, *j, *k;
  int w, h, d, r;

  /* vis(1,"specialbin [%d] %d,%d,%d\n", DIF(f,l), W, H, D); */
  if (f == l) { f->w = W; f->h = H; f->d = D; return; }
  if (DIF(f,l) == 5) {
    w = W/3; h = H/3; i = f+1; j = f+2; k = f+3; 
    i->w =   W-w; i->h =     h; i->d = D;
    j->w =     w; j->h =   H-h; j->d = D;
    k->w =   W-w; k->h =     h; k->d = D; 
    l->w =     w; l->h =   H-h; l->d = D;
    f->w = W-2*w; f->h = H-2*h; f->d = D;
    return;
  }

  m = f + (l-f) / 2;
  for (;;) {
    switch (rint(1,3)) {
      case 1: if (W < 2) break;
              w = rint(1,W-1); 
  	      specialbin(f, m, w, H, D);
	      specialbin(m+1, l, W-w, H, D);
	      return;
      case 2: if (H < 2) break;
	      h = rint(1,H-1); 
	      specialbin(f, m, W, h, D); 
	      specialbin(m+1, l, W, H-h, D);
	      return;
      case 3: if (D < 2) break;
	      d = rint(1,D-1);
	      specialbin(f, m, W, H, d); 
	      specialbin(m+1, l, W, H, D-d);
	      return;
    }
  }
}


/* ======================================================================
				randomtype
   ====================================================================== */

void randomtype(item *i, int W, int H, int D, int type)
{
  itype w, h, d, t;

  if (type <= 5) { /* Martello, Vigo */
    t = rint(1,10); if (t <= 5) type = t;
  }
  switch (type) {
    /* Martello, Vigo */
    case  1: w = rint(1,W/2);   h = rint(2*H/3,H); d = rint(2*D/3,D); break;
    case  2: w = rint(2*W/3,W); h = rint(1,H/2);   d = rint(2*D/3,D); break;
    case  3: w = rint(2*W/3,W); h = rint(2*H/3,H); d = rint(1,D/2);   break;
    case  4: w = rint(W/2,H);   h = rint(H/2,H);   d = rint(D/2,D);   break;
    case  5: w = rint(1,W/2);   h = rint(1,H/2);   d = rint(1,D/2);   break;

    /* Berkey, Wang */
    case 6: w = rint(1,10);    h = rint(1,10);    d = rint(1,10);    break;
    case 7: w = rint(1,35);    h = rint(1,35);    d = rint(1,35);    break;
    case 8: w = rint(1,100);   h = rint(1,100);   d = rint(1,100);   break;
  }
  i->w = w; i->h = h; i->d = d;
  i->x = 0; i->y = 0; i->z = 0;
}


/* ======================================================================
				allgood
   ====================================================================== */

boolean allgood(stype totvol, item *f, item *l)
{
  item *j, *m;
  stype vol;

  for (vol = 0, j = f, m = l+1; j != m; j++) {
    if ((j->w < 1) || (j->h < 1) || (j->d < 1)) return FALSE;
    vol += VOL(j);
  }
  return (vol == totvol);
}


/* ======================================================================
				maketest
   ====================================================================== */

void maketest(item *f, item *l, itype *W, itype *H, itype *D,
	      stype bdim, int type)
{
  register item *i, *j, *k, *m;
  int no;

  /* set bin dimensions */
  *W = bdim; *H = bdim; *D = bdim;

  /* make maxtypes item types */
  for (i = f, m = l+1, no = 1; i != m; i++, no++) {
    randomtype(i, *W, *H, *D, type);
    i->no = no;
  }

  /* make two complete bins when test */
  if (type == 9) {
    no = DIF(f,l)/3;
    k = f + no; m = k + no;
    for (;;) {
      specialbin(f  , k, *W, *H, *D); 
      specialbin(k+1, m, *W, *H, *D); 
      specialbin(m+1, l, *W, *H, *D); 
      if (allgood(3*bdim*bdim*bdim, f, l)) break;
    }
  }
}




/* ======================================================================
        standalone instance writer  (added: emits instances, no solver)
   ====================================================================== */

/* Writes the instance in the Martello-Pisinger-Vigo "3dbpp" text format:
 *    line 1:  n  <bin W> <bin H> <bin D>
 *    line j:  <item no> <w> <h> <d>
 * Item generation, RNG and seeding are UNCHANGED from the authors'
 * test3dbpp.c: seed = v + n for test v (1..10) of each (n, class).
 */
int main(int argc, char *argv[])
{
  int v, n, bdim, type, i;
  itype W, H, D;
  item tab[MAXITEMS], *f, *l;
  char name[512];
  FILE *out;
  const char *dir;

  if (argc != 5) {
    fprintf(stderr, "usage: %s n bindim type outdir\n", argv[0]);
    return 1;
  }
  n    = atoi(argv[1]);
  bdim = atoi(argv[2]);
  type = atoi(argv[3]);
  dir  = argv[4];

  if (n < 1 || n >= MAXITEMS) { fprintf(stderr,"bad n\n"); return 1; }

  trace = stderr;  /* original code keeps a global trace handle */

  for (v = 1; v <= TESTS; v++) {
    srand(v + n);                 /* IDENTICAL seeding to test3dbpp.c */
    f = &tab[0]; l = &tab[n-1];
    maketest(f, l, &W, &H, &D, bdim, type);

    sprintf(name, "%s/class%02d_n%03d_%02d.3dbpp", dir, type, n, v);
    out = fopen(name, "w");
    if (out == NULL) { fprintf(stderr,"cannot write %s\n", name); return 1; }
    fprintf(out, "%d %d %d %d\n", n, (int)W, (int)H, (int)D);
    for (i = 0; i < n; i++)
      fprintf(out, "%d %d %d %d\n", i+1, (int)tab[i].w, (int)tab[i].h, (int)tab[i].d);
    fclose(out);
  }
  return 0;
}
