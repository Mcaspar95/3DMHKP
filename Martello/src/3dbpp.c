
/* ======================================================================
      3D BIN PACKING, Silvano Martello, David Pisinger, Daniele Vigo
   ====================================================================== */

/* This code solves the three-dimensional bin-packing problem, which
 * asks for an orthogonal packing of a given set of rectangular-shaped
 * boxes into the minimum number of three-dimensional rectangular bins.
 * Each box j=1,..,n is characterized by a width w_j, height h_j, and
 * depth d_j. An unlimited number of indentical three-dimensional bins,
 * having width W, height H and depth D is available. The boxes have fixed
 * orientation, i.e. they may not be rotated, but no further restriction
 * is imposed.
 *
 * A description of this code is found in the following papers:
 *
 *   S.Martello, D.Pisinger, D.Vigo (1998)
 *   "An exact algorithm for the three-dimensional bin packing problem"
 *   submitted.
 * 
 *   S.Martello, D.Pisinger, D.Vigo (1998)
 *   "The three-dimensional bin packing problem"
 *   to appear in Operations Research.
 *
 * The present code is written in ANSI-C, and has been compiled with
 * the GNU-C compiler using option "-ansi -pedantic" as well as the
 * HP-UX C compiler using option "-Aa" (ansi standard).
 *
 * This file contains the callable routine binpack3d with prototype
 *
 *   void binpack3d(int n, int W, int H, int D,
 *          int *w, int *h, int *d, 
 *          int *x, int *y, int *z, int *bno,
 *          int *lb, int *ub, int timelimit);
 *
 * the meaning of the parameters is the following:
 *   n         Size of problem, i.e. number of boxes to be packed.
 *             This value must be smaller than MAXITEMS defined below.
 *   W,H,D     Width, height and depth of every bin.
 *   w,h,d     Integer arrays of length n, where w[j], h[j], d[j]
 *             are the dimensions of box j for j=0,..,n-1.
 *   x,y,z,bno Integer arrays of length n where the solution found
 *             is returned. For each box j=0,..,n-1, the bin number
 *             it is packed into is given by bno[j], and x[j], y[j], z[j] 
 *             are the coordinates of it lower-left-backward corner.
 *   lb        Lower bound on the solution value (returned by the procedure).
 *   ub        Objective value of the solution found, i.e. number of bins
 *             used to pack the n boxes. (returned by the procedure).
 *   timelimit Time limit for solving the problem expressed in seconds.
 *             If set to zero, the algorithm will run until an optimal
 *             solution is found; otherwise it terminates after timelimit
 *             seconds with a heuristic solution. 
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

#define MAXITEMS      201    /* max number of items (plus one item) */
#define MAXBPP    1000000    /* max numer of iterations in 1-dim bpp */
#define MAXITER      1000    /* max number of iterations in fast onebin_fill */
#define MAXCLOSE       16    /* max level for which try_close is applied */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <values.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <time.h>  
#include <limits.h>

/* ======================================================================
				   macros
   ====================================================================== */

#define TRUE  1           /* logical variables */
#define FALSE 0

#define WDIM  0           /* rotations of boxes */
#define HDIM  1
#define DDIM  2

#define VOL(i)            ((i)->w * (ptype) (i)->h * (i)->d)
#define MINIMUM(i,j)      ((i) < (j) ? (i) : (j))
#define MAXIMUM(i,j)      ((i) > (j) ? (i) : (j))
#define DIF(i,j)          ((int) ((j) - (i) + 1))
#define SWAPINT(a,b)      { register int t; t=*(a);*(a)=*(b);*(b)=t; }
#define SWAP(a,b)         { register item t; t=*(a);*(a)=*(b);*(b)=t; }
#define SWAPI(a,b)        { register itype t; t=(a);(a)=(b);(b)=t; }
#define SWAPP(a,b)        { register point t; t=*(a);*(a)=*(b);*(b)=t; }
#define DF(a,b)           ((r=(a).y-(b).y) != 0 ? r : (a).x-(b).x)


/* ======================================================================
				 type declarations
   ====================================================================== */

typedef short         boolean; /* logical variable      */
typedef short         ntype;   /* number of states,bins */
typedef short         itype;   /* can hold up to W,H,D  */
typedef long          stype;   /* can hold up to W*H*D  */
typedef long          ptype;   /* product multiplication */

/* item record */
typedef struct irec {
  ntype    no;           /* item number */
  itype    w;            /* item width  (x-size) */
  itype    h;            /* item height (y-size) */
  itype    d;            /* item depth  (z-size) */
  itype    x;            /* optimal x-position */
  itype    y;            /* optimal y-position */
  itype    z;            /* optimal z-position */
  ntype    bno;          /* bin number */
  boolean  k;            /* is the item chosen */
  stype    vol;          /* volume of item */
  struct irec *ref;      /* reference to original item (if necessary) */
} item;

/* all problem information */
typedef struct {
  itype    W;            /* x-size of bin             */
  itype    H;            /* y-size of bin             */
  itype    D;            /* z-size of bin             */
  stype    BVOL;         /* volume of a bin           */
  ntype    n;            /* number of items           */
  item     *fitem;       /* first item in problem     */
  item     *litem;       /* last item in problem      */
  item     *fsol;        /* first item in current solution */
  item     *lsol;        /* last item in current solution  */
  item     *fopt;        /* first item in optimal solution */
  item     *lopt;        /* last item in optimal solution  */
  boolean  *closed;      /* for each bin indicator whether closed */
  item     *fclosed;     /* first item in closed bins */
  item     *lclosed;     /* last item in closed bins */
  ntype    noc;          /* number of closed bins */
  itype    mindim;       /* currently smallest box length  */
  itype    maxdim;       /* currently largest box length   */
  stype    maxfill;      /* the best filling found         */
  int      mcut;         /* how many siblings at each node in b&b */
  boolean  optimal;      /* a solution which packs all found */

  /* different bounds */
  ntype    bound0;       /* Bound L_0 at root node */
  ntype    bound1;       /* Bound L_1 at root node */
  ntype    bound2;       /* Bound L_2 at root node */
  ntype    lb;           /* best of the above */
  ntype    z;            /* currently best solution */

  /* controle of 3d filler */
  long     maxiter;      /* max iterations in onebin_fill */
  long     miss;         /* number of items not packed in onebin_fill  */

  /* debugging and controle information */
  long     iterates;     /* iterations in branch-and-bound */
  long     exfill;       /* number of calls to onebin_fill algorithm */
  long     iter3d;       /* iterations in onebin_fill */
  long     zlayer;       /* heuristic solution layer */
  long     zmcut;        /* heuristic solution mcut */
  double   time;         /* computing time */
  double   lhtime;       /* layer heuristic computing time */
  double   mhtime;       /* mcut heuristic computing time */
  long     didpush;      /* did the lower bound push up bound */
  long     maxclose;     /* max number of closed bins at any time */
} allinfo;

/* structure for greedy algorithm */
typedef struct {
  int      lno;          /* layer number */
  int      d;            /* depth of layer */
  int      bno;          /* bin no assigned to layer */
  int      z;            /* z level of layer within bin */
  int      b;            /* temporary bin number */
} heurpair;

/* structure for extreme points in a single bin */
typedef struct {
  itype    x;            /* x-coordinate */
  itype    y;            /* y-coordinate */
  itype    z;            /* z-coordinate */
} point;

/* pointer to comparison function */
typedef int (*funcptr) (const void *, const void *); 


/* ======================================================================
				  global variables
   ====================================================================== */

/* boolean variable to indicate time-out situation */
boolean stopped; 

/* counter used to ensure that 1D BPP at most performs MAXBPP iterations */
int bpiterat;

/* maximum amount of time to be used */
int maxtime;

/* =======================================================================
				  error
   ======================================================================= */

void error(char *str, ...)
{
  va_list args;

  va_start(args, str);
  vprintf(str, args); printf("\n");
  va_end(args);
  printf("IRREGULAR PROGRAM TERMINATION\n");
  exit(-1);
}

/* **********************************************************************
   **********************************************************************
			     Timing routines 
   **********************************************************************
   ********************************************************************** */

/* This timing routine is based on the ANSI-C procedure "clock", which
 * has a resolution of 1000000 ticks per second. This however implies
 * that we pass the limit of a long integer after only 4295 seconds.
 * The following routine attempts to correct such situations by adding
 * the constant ULONG_MAX to the counter whenever wraparound can be
 * detected. But the user is advised to use a timing routine like "times"
 * (which however is not ANSI-C standard) for measuring longer time
 * periods.
 */

void timer(double *time)
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

/* test for time limit */
void timelimit(long max)
{
  double t;
  if (max == 0) return;
  timer(&t); if (t >= max) stopped = TRUE; 
}


/* **********************************************************************
   **********************************************************************
			     Small procedures
   **********************************************************************
   ********************************************************************** */

/* ======================================================================
			    simple comparisions
   ====================================================================== */

/* Comparisons used as argument to qsort. */

int dcomp(item *a, item *b) 
{ int r; r = b->d - a->d; if (r != 0) return r; else return b->no - a->no; }
int hcomp(item *a, item *b) 
{ int r; r = b->h - a->h; if (r != 0) return r; else return b->no - a->no; }
int vcomp(item *a, item *b) /* volume decr. */
{ int r; r = b->vol-a->vol; if (r != 0) return r; else return b->no - a->no; }
int xcomp(heurpair *a, heurpair *b) /* depth decr. */
{ int r; r = b->d - a->d; if (r != 0) return r; else return b->lno - a->lno; }
int lcomp(heurpair *a, heurpair *b) /* layer number decr. */
{ int r; r = a->lno-b->lno; if (r != 0) return r; else return b->d - a->d; }


/* ======================================================================
				  palloc
   ====================================================================== */

/* Memory allocation and freeing, with implicit check */

void *palloc(long sz, long no)
{
  long size;
  void *p;

  size = sz * no;
  if (size == 0) size = 1;
  p = (void *) malloc(size);
  if (p == NULL) error("no memory size %ld", size);
  return p;
}

void pfree(void *p)
{
  if (p == NULL) error("freeing null");
  free(p);
}


/* ======================================================================
			     checksol
   ====================================================================== */

/* check correctnes of solution, i.e. no items overlap, no duplicated items */

void checksol(allinfo *a, item *f, item *l)
{
  item *i, *j, *m;

  for (i = f, m = l+1; i != m; i++) { 
    if (!i->k) continue;  /* item currently not chosen */
    for (j = f; j != m; j++) {
      if (i == j) continue;
      if (i->no == j->no) error("duplicated item %d\n", i->no); 
      if (!j->k) continue;
      if (i->bno != j->bno) continue;
      if ((i->x + i->w > j->x) && (j->x + j->w > i->x) &&
	  (i->y + i->h > j->y) && (j->y + j->h > i->y) &&
	  (i->z + i->d > j->z) && (j->z + j->d > i->z)) {
	error("overlap item %d,%d: [%d,%d,%d] [%d,%d,%d]",
	      i->no, j->no, i->w, i->h, i->d, j->w, j->h, j->d);
      }
    }
  }
}


/* ======================================================================
			       savesol
   ====================================================================== */

/* save an updated solution, checking its validity */

void savesol(allinfo *a, item *f, item *l, ntype z)
{
  item *i, *j, *k, *m;

  /* first check validity */
  if (z >= a->z) error("not improved");
  for (i = f, m = l+1; i != m; i++) {
    if ((1 <= i->bno) && (i->bno <= z)) continue; 
    error("illegal bin %d, item %d", i->bno, i->no); 
  }

  /* now do the saving */
  a->z = z;
  for (i = f, k = a->fopt, m = l+1; i != m; i++, k++) *k = *i;
  for (i = a->fclosed, m = a->lclosed+1; i != m; i++, k++) *k = *i;
  for (i = a->fopt, m = a->lopt+1; i != m; i++) i->k = TRUE;
  if (DIF(a->fopt,k-1) != a->n) error("not correct amount of items");
  checksol(a, a->fopt, a->lopt);
}


/* ======================================================================
			       isortincr
   ====================================================================== */

/* A specialized routine for sorting integers in increasing order. */
/* qsort could be used equally well, but this routine is faster. */

void isortincr(int *f, int *l)
{
  register int mi;
  register int *i, *j, *m;
  register int d;

  d = l - f + 1;
  if (d < 1) error("negative interval in isortincr");
  if (d == 1) return;
  m = f + d / 2; if (*f > *m) SWAPINT(f, m);
  if (d > 2) { if (*m > *l) { SWAPINT(m, l); if (*f > *m) SWAPINT(f, m); } }
  if (d <= 3) return;
  mi = *m; i = f; j = l;
  for (;;) {
    do i++; while (*i < mi);
    do j--; while (*j > mi);
    if (i > j) break; else SWAPINT(i, j);
  }
  isortincr(f, i-1); isortincr(i, l);
}


/* ======================================================================
			       isortdecr
   ====================================================================== */

/* A specialized routine for sorting integers in decreasing order. */
/* qsort could be used equally well, but this routine is faster. */

void isortdecr(int *f, int *l)
{
  register int mi;
  register int *i, *j, *m;
  register int d;

  d = l - f + 1;
  if (d < 1) error("negative interval in isortdecr");
  if (d == 1) return;
  m = f + d / 2; if (*f < *m) SWAPINT(f, m);
  if (d > 2) { if (*m < *l) { SWAPINT(m, l); if (*f < *m) SWAPINT(f, m); } }
  if (d <= 3) return;
  mi = *m; i = f; j = l;
  for (;;) {
    do i++; while (*i > mi);
    do j--; while (*j < mi);
    if (i > j) break; else SWAPINT(i, j);
  }
  isortdecr(f, i-1); isortdecr(i, l);
}


/* ======================================================================
			       psortdecr
   ====================================================================== */

/* A specialized routine for sorting extreme points according to decreasing */
/* y-coordinate (decreasing x-coordinate in case of ties) */

void psortdecr(point *f, point *l)
{
  register point mi;
  register point *i, *j, *m;
  register int d, r;

  d = l - f + 1; 
  if (d <= 1) return;
  m = f + d / 2; if (DF(*f,*m)<0) SWAPP(f,m); 
  if (d == 2) return;
  if (DF(*m,*l)<0) { SWAPP(m,l); if (DF(*f,*m)<0) SWAPP(f,m); }
  if (d <= 3) return;
  mi = *m; i = f; j = l;
  for (;;) {
    do i++; while (DF(*i,mi) > 0);
    do j--; while (DF(*j,mi) < 0);
    if (i > j) break; else SWAPP(i, j);
  }
  psortdecr(f, i-1); psortdecr(i, l);
}


/* **********************************************************************
   **********************************************************************
			      Lower Bounds
   **********************************************************************
   ********************************************************************** */

/* ======================================================================
			      bound_zero
   ====================================================================== */

/* The continuous bound L_0 */

int bound_zero(allinfo *a, item *f, item *l)
{
  item *i, *m;
  stype vsum, lb;

  vsum = 0; 
  for (i = f, m = l+1; i != m; i++) vsum += i->vol;
  lb = (stype) ceil(vsum / (double) a->BVOL);
  return lb;
}


/* ======================================================================
                               rotate_solution
   ====================================================================== */

/* rotates the solution. After 3 rotations we return to original problem */

void rotate_solution(allinfo *a, item *f, item *l)
{
  register item *i, *m;
  register itype w, x;

  for (i = f, m = l+1; i != m; i++) {
    w = i->w; i->w = i->h; i->h = i->d; i->d = w;
    x = i->x; i->x = i->y; i->y = i->z; i->z = x;
  }
}


/* ======================================================================
			       rotate_problem
   ====================================================================== */

/* rotates the dimensions by one step */

void rotate_problem(allinfo *a, item *f, item *l)
{
  register item *i, *m;
  register itype w, x;

  for (i = f, m = l+1; i != m; i++) {
    w = i->w; i->w = i->h; i->h = i->d; i->d = w;
    x = i->x; i->x = i->y; i->y = i->z; i->z = x;
  }
  w = a->W; a->W = a->H; a->H = a->D; a->D = w;
}


/* ======================================================================
			       choose_items
   ====================================================================== */

/* returns a set of items with w > W2 and d > D2. This set is used in */
/* bound_one */

void choose_items(allinfo *a, item *f, item *l, int W2, int D2, 
		  item *fitem, item **litem)
{
  item *i, *k, *m;

  for (i = f, m = l+1, k = fitem; i != m; i++) {
    if ((i->w > W2) && (i->d > D2)) { *k = *i; k++; }
  }
  *litem = k-1;
}


/* ======================================================================
			       find_plist
   ====================================================================== */

/* returns a zero-terimanted list of distinct dimensions */

void find_plist(item *fitem, item *litem, itype M, int dim, int *pl)
{
  register item *i, *m;
  register int *k, *j, *l;

  i = fitem; m = litem+1; k = pl;
  switch (dim) {
    case WDIM: for (; i != m; i++) {
		 if (i->w <= M) { *k = i->w; k++; } 
	       } break;
    case HDIM: for (; i != m; i++) {
		 if (i->h <= M) { *k = i->h; k++; } 
	       } break;
    case DDIM: for (; i != m; i++) {
		 if (i->d <= M) { *k = i->d; k++; } 
	       } break;
  }
  if (k == pl) { *k = 0; return; }
  isortincr(pl, k-1);  /* sort the dimensions */
  for (j = pl+1, l = pl; j != k; j++) { /* remove duplicates */
    if (*j != *l) { l++; *l = *j; } 
  }
  l++; *l = 0;
}


/* ======================================================================
			       bound_one
   ====================================================================== */

/* Derive bound L_1 for a fixed dimension */

int bound_one_x(allinfo *a, item *f, item *l)
{
  register item *i, *m;
  register itype H, H2, h;
  register int p, j1, j2, j3, j2h, j2hp, j3h;
  int *pp, lb, lb_one, alpha, beta;
  item fitem[MAXITEMS], *litem;
  int plist[MAXITEMS];

  if (l == f-1) return 0;
  lb = 1; H = a->H; H2 = H/2;
  choose_items(a, f, l, a->W/2, a->D/2, fitem, &litem);
  if (litem == fitem-1) { /* empty */ return lb; }

  find_plist(fitem, litem, H2, HDIM, plist);
  for (pp = plist; *pp != 0; pp++) {
    p = *pp; j1 = j2 = j3 = j2h = j2hp = j3h = 0;
    for (i = fitem, m = litem+1; i != m; i++) {
      h = i->h; 
      if (h > H-p) j1++;
      if ((H-p >= h) && (h > H2)) { j2++; j2h += h; j2hp += (H-h)/p; }
      if ((H2 >= h) && (h >= p)) { j3++; j3h += h; }
    }
    alpha = (int) ceil((j3h - (j2 * H - j2h)) / (double) H);
    beta  = (int) ceil((j3 - j2hp) / (double) (H/p));
    if (alpha < 0) alpha = 0;
    if (beta  < 0) beta  = 0;
    lb_one = j1 + j2 + MAXIMUM(alpha, beta);
    if (lb_one > lb) lb = lb_one;
  }
  return lb;
}


/* Derive bound L_1 as the best of all L_1 bounds for three rotations */

int bound_one(allinfo *a, item *f, item *l)
{
  int i, lb, lbx;

  lb = 0;
  for (i = WDIM; i <= DDIM; i++) {
    lbx = bound_one_x(a, f, l);
    if (lbx > lb) lb = lbx; 
    rotate_problem(a, f, l);
  } 
  return lb;
}


/* ======================================================================
			       bound_two
   ====================================================================== */

/* Derive bound L_2 for a fixed dimension */

int bound_two_x(allinfo *a, item *f, item *l)
{
  register item *i, *m;
  register itype W, H, D, w, h, d, W2, D2;
  register int p, q, k1h, k23v;
  int hlb1, lb, lb1, lbx, fract;
  int plist[MAXITEMS], qlist[MAXITEMS];
  int *qq, *pp;
  double WD, BVOL;

  /* derive bound_one */
  lb = lb1 = bound_one_x(a, f, l);
  W = a->W; H = a->H; D = a->D; hlb1 = H * lb1; 
  W2 = W/2; D2 = D/2; WD = W*(double)D; BVOL = a->BVOL;

  /* run through all values of p, q */
  find_plist(f, l, W2, WDIM, plist);
  find_plist(f, l, D2, DDIM, qlist);
  for (pp = plist; *pp != 0; pp++) {
    p = *pp;
    for (qq = qlist; *qq != 0; qq++) {
      q = *qq;
      k1h = k23v = 0;
      for (i = f, m = l+1; i != m; i++) {
	w = i->w; h = i->h; d = i->d;
	if ((w > W - p) && (d > D - q)) { k1h += h; continue; }
	if ((w >= p) && (d >= q)) { k23v += i->vol; }
      }
      fract = (int) ceil((k23v - (hlb1 - k1h)*WD) / BVOL);
      if (fract < 0) fract = 0;
      lbx = lb1 + fract;
      if (lbx > lb) lb = lbx;
    }
  }
  return lb;
}


/* Derive bound L_2 as the best of all L_2 bounds for three rotations */

int bound_two(allinfo *a, item *f, item *l)
{
  int i, lb, lbx;

  lb = 0;
  for (i = WDIM; i <= DDIM; i++) {
    lbx = bound_two_x(a, f, l);
    if (lbx > lb) lb = lbx;
    rotate_problem(a, f, l);
  } 
  return lb;
}


/* **********************************************************************
   **********************************************************************
			    heuristic filling
   **********************************************************************
   ********************************************************************** */

/* ======================================================================
			      onelayer
   ====================================================================== */

/* Fill a layer of depth f->d by arranging the items in a number of */
/* vertical shelfs. The items $i$ packed are assigned coordinates */
/* (i->x, i->y) and the field i->k is set to the argument d (layer no). */

void onelayer(allinfo *a, item *f, item *m, item *l, int d)
{
  int n, s, t;
  itype r; /* remaining width */
  itype width[MAXITEMS], height[MAXITEMS], x[MAXITEMS];
  item *i;
  
  n = DIF(f,l)+1;
  qsort(f, DIF(f,m), sizeof(item), (funcptr) hcomp);
  r = a->W;
  x[0] = 0; width[0] = 0; height[0] = 0;
  for (s = 1, i = f; i != l+1; s++) {
    x[s] = x[s-1] + width[s-1];
    height[s] = 0;
    width[s] = i->w; if (width[s] > r) width[s] = r;
    r -= width[s];
    for ( ; i != l+1; i++) {
      for (t = s; t != 0; t--) {
	if (i->w <= width[t]) {
	  if (height[t] + i->h <= a->H) {
	    i->y = height[t]; i->x = x[t]; i->k = d;
	    height[t] += i->h; break;
	  }
	}
      }
      if ((t == 0) && (r > 0)) break; /* new strip */
    }
  }
}


/* ======================================================================
				  countarea
   ====================================================================== */

/* Select a subset of the items such that the selected items have a total */
/* area of two times the face of a bin (the parameter: barea) */

item *countarea(item *f, item *l, stype barea)
{
  item *i;
  stype area, d;

  for (area = 0, i = f; i != l+1; i++) {
    d = i->h * (ptype) i->w;
    area += d;
    if (area > 2*barea) return i-1;
  }
  return l;
}


/* ======================================================================
				  remitems
   ====================================================================== */

/* Remove the items which were chosen for a layer, i.e. where i->k != 0. */
/* The depth of the layer is set equal to the deepest box chosen. */

item *remitems(item *f, item *l, itype *depth)
{
  item *i, *j;
  itype d;

  for (i = f, j = l, d = 0; i != j+1; ) {
    if (i->k) { if (i->d > d) d = i->d; i++; } else { SWAP(i,j); j--; }
  }
  *depth = d;
  return i;
}


/* ======================================================================
				  assignitems
   ====================================================================== */

/* Assign z-coordinates to the items, once they layers have been combined */
/* to individual bins by solving a 1-dimensional Bin-packing Problem. */

void assignitems(heurpair *t, heurpair *u, ntype maxbin, item *f, item *l)
{
  item *i, *m;
  heurpair *h;
  itype b, z;

  /* derive z-coordinates for each layer */
  for (b = 1; b <= maxbin; b++) {
    z = 0;
    for (h = t; h <= u; h++) if (h->bno == b) { h->z = z; z += h->d; }
  }

  for (i = f, m = l+1; i != m; i++) {
    h = t + i->k - 1;
    i->z = h->z; i->bno = h->bno;
  }
}


/* ======================================================================
				 onedim_binpack
   ====================================================================== */

/* One-dimensional bin-packing algorithm. In each iteration, the next */
/* item is assigned to every open bin as well as to a new bin. The  */
/* algorithm terminates when MAXBPP iterations have been performed, */
/* returning the heuristic solution found. */

void onedim_binpack(heurpair *i, heurpair *f, heurpair *l, 
		    int *b, int bno, itype *z)
{
  int j, *bc;
  heurpair *k;

  bpiterat++; if (bpiterat > MAXBPP) return;
  if (bno >= *z) return; /* no hope of improvement */

  if (i > l) {
    *z = bno; 
    for (k = f; k <= l; k++) k->bno = k->b;
  } else {
    for (j = 0; j < bno; j++) {
      bc = b + j;
      if (i->d <= *bc) {
	*bc -= i->d; i->b = j+1;
	onedim_binpack(i+1, f, l, b, bno, z);
	*bc += i->d;
      }
    }
    bc = b + bno;
    *bc -= i->d; i->b = bno+1;
    onedim_binpack(i+1, f, l, b, bno+1, z);
    *bc += i->d;
  }
}


/* ======================================================================
			      dfirst_heuristic
   ====================================================================== */

/* Heuristic algorithm for the 3D BPP. A number of layers are constructed */
/* using the shelf approach to pack every layer. Then the individual layers */
/* are combined to bins by solving a 1D BPP defined in the layer depths. */

void dfirst_heuristic(allinfo *a)
{
  item *j, *f, *l, *m;
  int i, n, h, b[MAXITEMS];
  heurpair t[MAXITEMS];
  itype d, z;

  /* initialize items */
  for (j = a->fitem, m = a->litem+1; j != m; j++) {
    j->bno = j->x = j->y = j->z = 0; j->k = FALSE;
  }

  /* fill layer one by one */
  for (f = a->fitem, l = a->litem, h = 0; ; h++) {
    n = DIF(f,l); if (n == 0) break;
    qsort(f, n, sizeof(item), (funcptr) dcomp);
    m = countarea(f, l, a->W * (ptype) a->H);
    onelayer(a, f, m, l, h+1);
    f = remitems(f, l, &d);
    t[h].d = d; t[h].bno = h+1;  /* initially put layers into separate bins */
    t[h].z = 0; t[h].lno = h+1;  /* this ensures fes. solution if terminate */
  }

  /* split into bins by solving 1-dim binpacking */
  for (i = 0; i < h; i++) b[i] = a->D;  /* all bins are empty */
  qsort(t, h, sizeof(heurpair), (funcptr) xcomp);
  z = h+1; bpiterat = 0;
  onedim_binpack(t, t, t+h-1, b, 0, &z); 
  qsort(t, h, sizeof(heurpair), (funcptr) lcomp); /* order according to lno */

  /* now assign bin number to each items */
  assignitems(t, t+h-1, z, a->fitem, a->litem);
  if (z < a->zlayer) a->zlayer = z;
  if (a->zlayer < a->z) savesol(a, a->fitem, a->litem, a->zlayer);
}


/* ======================================================================
				dfirst3_heuristic
   ====================================================================== */

/* Call the heuristic dfirst_heuristic, for three different rotations */
/* of the problem */

void dfirst3_heuristic(allinfo *a)
{
  int i;
  double t1, t2;
 
  timer(&t1); 
  a->zlayer = a->n;  /* very bad incumbent solution */
  for (i = WDIM; i <= DDIM; i++) {
    dfirst_heuristic(a);
    rotate_solution(a, a->fopt, a->lopt);
    rotate_problem(a, a->fitem, a->litem);
  }
  timer(&t2); 
  a->lhtime = t2 - t1;
}



/* **********************************************************************
   **********************************************************************
			     fill one 3D bin
   **********************************************************************
   ********************************************************************** */


/* ======================================================================
				envelope
   ====================================================================== */

/* Find the two-dimensional envelope of the boxes given by extreme */
/* points f to l. */

void envelope(point *f, point *l, point *s1, point **sm,
	     itype W, itype H, itype D, itype RW, itype RH, itype cz, 
             point **ll, int *nz, stype *area)
{
  register point *i, *s, *t;
  register itype x, xx, y, z, ix, iy, iz, mz;
  register stype sum;

  /* find corner points and area */
  x = xx = z = 0; y = H; sum = 0; mz = D;
  for (i = t = f, s = s1; i != l; i++) {
    iz = i->z; if (iz <= cz) continue;
    if (iz < mz) mz = iz; /* find minimum next z coordinate */
    ix = i->x; if (ix <= x) {
      if (iz > z) { *t = *i; t++; } 
      continue;
    }
    iy = i->y;
    if ((x <= RW) && (iy <= RH)) { 
      s->x = x; s->y = iy; s->z = cz; s++; 
      sum += (x - xx) * (ptype) y; 
      y = iy; xx = x;
    }
    x = ix; z = iz; *t = *i; t++;
  }
  if (y != 0) sum += (W - xx) * (ptype) y;
  *sm = s-1;
  *area = sum;
  *nz = mz;
  *ll = t-1;
}

/* ======================================================================
			     checkdom
   ====================================================================== */

/* The 3D envelope is found by deriving a number of 2D envelopes. This */
/* may however introduce some "false" corner points, which are marked */
/* by the following algorithm. */

void checkdom(point *s1, point *sl, point *sm)
{
  register point *s, *t, *u;

  if (sl == s1-1) return;
  for (s = s1, t = sl+1, u = sm+1; t != u; t++) {
    while (s->x < t->x) { s++; if (s > sl) return; }
    if ((s->x == t->x) && (s->y == t->y)) t->z = 0;
  } 
}


/* ======================================================================
			     removedom
   ====================================================================== */

/* Remove "false" corner points marked by algorithm "checkdom". */

point *removedom(point *s1, point *sl)
{
  register point *i, *m, *k;
  register int pz;
 
  for (i = k = s1, m = sl+1, pz = 0; i != m; i++) {
    if (i->z == 0) continue; 
    *k = *i; k++; 
  }
  return k-1;
}

/* ======================================================================
			     initboxes
   ====================================================================== */

/* Initialize boxes. Already placed boxes define extreme points, and thus */
/* they are placed into the list fc,..,lc. Not placed boxes are used to */
/* derive minimum dimensions. */

void initboxes(item *f, item *l, point *fc, point **lc, 
               int *minw, int *minh, int *mind)
{
  register item *j, *m;
  register point *k;
  register int minx, miny, minz;

  minx = *minw; miny = *minh; minz = *mind; 
  for (j = f, k = fc, m = l+1; j != m; j++) {
    if (j->k) { /* defines an extreme box */
      k->x = j->x+j->w; k->y = j->y+j->h; k->z = j->z+j->d; k++;
    } else { /* free box */
      if (j->w < minx) minx = j->w;
      if (j->h < miny) miny = j->h;
      if (j->d < minz) minz = j->d;
    }
  }
  *minw = minx; *minh = miny; *mind = minz; 
  *lc = k-1;
}


/* ======================================================================
			      findplaces
   ====================================================================== */

/* Find all corner points, where a new box may be placed as well as the */
/* volume of the "envelope" occupied by already placed boxes. Already */
/* placed boxes are given by f,..,l, while a list of possible placings */
/* is returned in s1,..,sm. The return value of the procedure is an upper */
/* bound on the possible filling of this bin. */

stype findplaces(allinfo *a, item *f, item *l, 
                 point *s1, point **sm, stype fill)
{
  register point *k;
  int minw, minh, mind, W, H, D, RW, RH, z, zn;
  point *sk, *lc, *sl, *st, *s0;
  stype vol, area;
  point fc[MAXITEMS+1];

  /* select items which are chosen, and find min dimensions of unchosen */
  minw = W = a->W; minh = H = a->H; mind = D = a->D; 
  initboxes(f, l, fc, &lc, &minw, &minh, &mind);

  /* sort the boxes according to max y (first) max x (second) */
  if (lc >= fc) psortdecr(fc, lc); /* order decreasing */

  /* for each z-coordinate find the 2D envelope */
  vol = 0; sl = s1-1; sk = s1;
  RW = W - minw; RH = H - minh; 
  lc++; k = lc; k->x = W+1; k->y = 0; k->z = a->D+1;
  for (z = 0; z != D; z = zn) {
    /* find 2D envelope for all boxes which cover *z */
    envelope(fc, lc+1, sl+1, &st, W, H, D, RW, RH, z, &lc, &zn, &area);
    if (zn + mind > D) zn = D; /* nothing fits between zn and D */
    vol += area * (ptype) (zn - z); /* update volume */
    checkdom(sk, sl, st); /* check for dominance */
    sk = sl+1; sl = st; 
    if (z == 0) s0 = sl;
  }
  *sm = removedom(s0+1, sl);
  return fill + (a->BVOL - vol); /* bound is curr filling + all free vol */
}


/* ======================================================================
				branch
   ====================================================================== */

/* Recursive algorithm for solving a knapsack filling of a single bin. */
/* In each iteration, the set of feasible positions for placing a new */
/* box is found, and an upper bound on the filling is derived. If the */
/* bound indicates that an improved solution still may be obtained, every */
/* box is tried to be placed at every feasible position, before calling */
/* the algorithm recursively. */

void branch(allinfo *a, item *f, item *l, int miss, stype fill)
{
  register item *i;
  register point *s;
  boolean sym;
  int onex, oney, onez, d;
  stype bound;
  point *sl, s1[9*MAXITEMS];

  if (stopped) return;
  a->iter3d++;
  if (a->iter3d % 100000 == 0) timelimit(maxtime);
  if (a->iter3d == a->maxiter) a->optimal = TRUE;
  if (a->optimal) return;

  /* find min/max dimensions of remaining items */
  if (miss == 0) {
    /* none left -> good: save solution */
    memcpy(a->fsol, f, sizeof(item) * DIF(f,l));
    a->maxfill = a->BVOL; a->optimal = TRUE; a->miss = miss;
  } else {
    /* check if better filling */
    if (fill > a->maxfill) {
      memcpy(a->fsol, f, sizeof(item) * DIF(f,l)); 
      a->maxfill = fill; a->miss = miss;
    }

    /* find bound and positions to place new items */
    bound = findplaces(a, f, l, s1, &sl, fill);

    if (bound > a->maxfill) {
      /* for each position in S, try to place an item there */
      for (s = s1; s != sl+1; s++) {
	for (i = f, d = 0; i != l+1; i++) {
	  if (i->k) continue; /* already chosen */

	  /* see if item fits at position s */
	  if ((int) (s->x) + i->w > a->W) continue;
	  if ((int) (s->y) + i->h > a->H) continue;
	  if ((int) (s->z) + i->d > a->D) continue;

	  /* place item and call recursively */
	  i->k = TRUE; i->x = s->x; i->y = s->y; i->z = s->z;
	  branch(a, f, l, miss - 1, fill + i->vol);
	  i->k = FALSE; i->x = i->y = i->z = 0; d++;
	  if (d == a->mcut) break; /* terminate after mcut branches */
	  if (a->optimal) break;
	}
      }
    }
  }
}


/* ======================================================================
				onebin_fill
   ====================================================================== */

/* Knapsack filling of a single bin. The following procedure initializes */
/* some variables before calling the recursive branch-and-bound algorithm. */
 
boolean onebin_fill(allinfo *a, item *f, item *l, boolean fast)
{
  item *i, *j, *m;
  stype vol;
  int b, n;

  /* initialize items */
  a->exfill++;
  for (i = f, m = l+1, vol = 0; i != m; i++) {
    i->x = 0; i->y = 0; i->z = 0; i->k = 0; vol += i->vol;
  }

  /* try to fill one bin with items [f,l] */
  n = DIF(f,l);
  a->iter3d  = 0;
  a->maxfill = vol-1; /* lower bound equal to all minus one */
  a->miss = n; 
  a->maxiter = (fast ? MAXITER : 0); /* limited or infinitly many */
  a->optimal = FALSE;
  a->mcut = 0; /* try all branches */
  branch(a, f, l, n, 0);

  /* copy solution */
  if (a->maxfill == a->BVOL) { /* NB: maxfill = BVOL, when optimal found */
    for (i = a->fsol, j = f, m = l+1; j != m; i++, j++) *j = *i;
    return TRUE;
  } 
  return FALSE;
}


/* ======================================================================
				mcut_heuristic
   ====================================================================== */

/* Knapsack filling of a single bin, solved heuristically. The heuristic */
/* is based on the exact algorithm for knapsack filling a single bin, where */
/* only a limited number of sub-nodes are considered at every branching node */
/* (the so-called m-cut approach) */
 
void mcut_heuristic(allinfo *a)
{
  item *f, *l, *i, *j, *m;
  int b, n;

  /* initialize items */
  for (i = a->fitem, m = a->litem+1; i != m; i++) {
    i->bno = i->x = i->y = i->z = 0; i->k = FALSE;
  }

  /* fill bins one by one */
  f = a->fitem; l = a->litem;
  for (b = 1; ; b++) {
    /* fill one bin */
    for (i = f; i <= l; i++) i->k = FALSE; /* item not chosen */
    a->iter3d = a->maxfill = 0;
    a->miss = DIF(f,l);
    a->maxiter = 5000;
    a->optimal = FALSE;
    n = DIF(f,l);
    a->mcut = 2;
    if (n < 15) a->mcut = 3;
    if (n < 10) a->mcut = 4;
    branch(a, f, l, n, 0);

    /* copy solution */
    for (i = a->fsol, j = f, m = l+1; j != m; i++, j++) *j = *i;

    /* remove chosen items */
    for (i = f; i <= l;) if (i->k) { i->bno = b; SWAP(i,l); l--; } else i++;

    /* check if finished */
    if (l == f-1) break;
  }
  if (b < a->zmcut) a->zmcut = b; /* save solution */
  if (a->zmcut < a->z) savesol(a, a->fitem, a->litem, a->zmcut);
}

/* ======================================================================
				mcut3_heuristic
   ====================================================================== */

/* Knapsack filling of a single bin, solved heuristically. Three different */
/* rotations of the problem are considered, and the best found solution */
/* is selected. */

void mcut3_heuristic(allinfo *a)
{
  int i;
  double t1, t2;
 
  timer(&t1); 
  a->zmcut = a->n;  /* very bad lower bound */
  for (i = WDIM; i <= DDIM; i++) {
    mcut_heuristic(a);
    rotate_solution(a, a->fopt, a->lopt);
    rotate_problem(a, a->fitem, a->litem);
  }
  timer(&t2);
  a->mhtime = t2 - t1;
}


/* **********************************************************************
   **********************************************************************
 	       branch-and-bound for 3D bin-packing problem
   **********************************************************************
   ********************************************************************** */


/* ======================================================================
				 fits
   ====================================================================== */

/* The routine "fitsm" checks whether a given subset of boxes fits into */
/* a single bin. To improve performance, specialized algorithms are derived */
/* for cases with two to four boxes */

boolean fits2(item *i, item *j, itype W, itype H, itype D)
{
  if (i->w + j->w <= W) { j->x = i->w; return TRUE; }
  if (i->h + j->h <= H) { j->y = i->h; return TRUE; }
  if (i->d + j->d <= D) { j->z = i->d; return TRUE; }
  return FALSE;
}

boolean fits2p(item *i, item *j, itype W, itype H, itype D)
{
  if (i->w + j->w <= W) return TRUE;
  if (i->h + j->h <= H) return TRUE;
  if (i->d + j->d <= D) return TRUE;
  return FALSE;
}

boolean fits3(item *i, item *j, item *k, itype W, itype H, itype D)
{
  item *t;
  itype w, h, d, r;

  /* with 3 items, the solution can always be cut by guillotine cuts */
  for (r = 1; r <= 3; r++) {
    /* cut (i,j) and (k) in one of three dimensions */
    w = W - k->w; h = H - k->h; d = D - k->d; 
    if ((i->w<=w) && (j->w<=w) && fits2(i,j,w,H,D)) { k->x = w; return TRUE; } 
    if ((i->h<=h) && (j->h<=h) && fits2(i,j,W,h,D)) { k->y = h; return TRUE; } 
    if ((i->d<=d) && (j->d<=d) && fits2(i,j,W,H,d)) { k->z = d; return TRUE; } 
    t = i; i = j; j = k; k = t;
  }
  return FALSE; 
}  

boolean guil31(item *i, item *j, item *k, item *l, itype W, itype H, itype D)
{ 
  register itype w;
  w = W - l->w; 
  if ((i->w<=w) && (j->w<=w) && (k->w<=w) && fits3(i,j,k,w,H,D)) { 
    l->x = w; return TRUE; 
  } 
  return FALSE;
} 

boolean guil22(item *i, item *j, item *k, item *l, itype W, itype H, itype D)
{ 
  register itype w;
  w = W - MAXIMUM(k->w,l->w);  
  if ((i->w<=w) && (j->w<=w) && fits2p(i,j,w,H,D) && fits2(k,l,W-w,H,D)) { 
    fits2(i,j,w,H,D); k->x += w; l->x += w; return TRUE; 
  }
  return FALSE;
}

boolean circ(item *i, item *j, item *k, item *l, itype W, itype H, itype D)
{ 
  register itype w;
  if ((i->w + j->w <= W) && (i->h + l->h <= H) && 
      (l->w + k->w <= W) && (j->h + k->h <= H)) { 
    if ((j->h < i->h) && (l->h < k->h) && (i->w < l->w) && (k->w < j->w)) { 
      j->x = i->w; l->y = i->h; k->x = l->w; k->y = j->h; return TRUE; 
    } 
    if ((i->h < j->h) && (k->h < l->h) && (l->w < i->w) && (j->w < k->w)) { 
      j->x = i->w; l->y = i->h; k->x = l->w; k->y = j->h; return TRUE;  
    } 
  } 
  return FALSE;
}

boolean fits4one(item *i, item *j, item *k, item *l, itype W, itype H, itype D)
{
  itype w;

  /* guillotine (i,j,k) and (l) */
  if (guil31(i, j, k, l, W, H, D)) return TRUE;
  if (guil31(j, k, l, i, W, H, D)) return TRUE;
  if (guil31(k, l, i, j, W, H, D)) return TRUE;
  if (guil31(l, i, j, k, W, H, D)) return TRUE;

  /* guillotine (i,j) and (k,l) */
  if (guil22(i, j, k, l, W, H, D)) return TRUE;
  if (guil22(i, k, j, l, W, H, D)) return TRUE;
  if (guil22(i, l, j, k, W, H, D)) return TRUE;

  /* circular cut, a symmetrical solution with "i" in left-down corner exists */
  /*         LLLK        LKKK   */
  /* either  I  K   or   L  J   */
  /*         IJJJ        IIIJ   */
  if (circ(i, j, k, l, W, H, D)) return TRUE;
  if (circ(i, k, l, j, W, H, D)) return TRUE;
  if (circ(i, l, j, k, W, H, D)) return TRUE;
  if (circ(i, l, k, j, W, H, D)) return TRUE;
  if (circ(i, j, l, k, W, H, D)) return TRUE;
  if (circ(i, k, j, l, W, H, D)) return TRUE;
  return FALSE;
}


boolean fits4(allinfo *a, item *f, item *l)
{
  boolean fits;
  int h;

  fits = FALSE; 
  for (h = WDIM; h <= DDIM; h++) {
    if (!fits) fits = fits4one(f, f+1, f+2, l, a->W, a->H, a->D); 
    rotate_problem(a, f, l);
  } 
  return fits;
}


boolean fitsm(allinfo *a, item *t, item *k, boolean fast)
{
  boolean fits;
  ntype lb;

  lb = bound_two(a, t, k);
  if (lb > 1) return FALSE;
  fits = onebin_fill(a, t, k, fast); 
  return fits;
}


/* ======================================================================
				does_fit
   ====================================================================== */

/* The following procedure checks whether a new box "j" fits into the the */
/* bin "bno" (together with already placed boxes in the bin). If the answer */
/* is "no" then definitly it is not possible to place the box into the bin. */

boolean does_fit(allinfo *a, item *j, int bno)
{
  register item *i, *k, *m;
  item t[MAXITEMS];
  boolean fits;
  int size;


  for (i = a->fitem, m = j, k = t-1; i != m; i++) {
    if (i->bno == bno) { k++; *k = *i; k->x = k->y = k->z = 0; k->ref = i; }
  }
  k++; *k = *j; k->x = k->y = k->z = 0; k->ref = j; k->k = TRUE; 

  size = DIF(t,k); 
  switch (size) {
    case 0: error("no items in does_fit");
    case 1: fits = TRUE; k->x = k->y = k->z = 0; break;
    case 2: fits = fits2(t, k, a->W, a->H, a->D); break;
    case 3: fits = fits3(t, t+1, k, a->W, a->H, a->D); break;
    case 4: fits = fits4(a, t, k); break;
    default: fits = fitsm(a, t, k, FALSE); break;
  }

  if (fits) {
    /* copy solution back */
    for (i = t, m = k+1; i != m; i++) {
      k = i->ref; k->x = i->x; k->y = i->y; k->z = i->z; k->k = TRUE;
    }
  }
  return fits;
}


/* ======================================================================
				can_fill
   ====================================================================== */

/* This is a heuristic version of "does_fit". If the answer is "yes" then */
/* a heuristic solution has been found where items f,..,l fit into a bin. */
/* If the answer is "no" then a filling may still be possible, but it was */
/* not found by the heuristic. */ 

boolean can_fill(allinfo *a, item *f, item *l)
{
  item *i, *m;
  boolean fits;
 
  for (i = f, m = l+1; i != m; i++) { i->x = i->y = i->z = 0; }
  switch (DIF(f,l)) {
    case 0: error("no items in can_fill");
    case 1: fits = TRUE; break;
    case 2: fits = fits2(f, l, a->W, a->H, a->D); break;
    case 3: fits = fits3(f, f+1, l, a->W, a->H, a->D); break;
    case 4: fits = fits4(a, f, l); break;
    default: fits = fitsm(a, f, l, TRUE); break;
  }
  return fits;
}


/* ======================================================================
				try_close
   ====================================================================== */

/* A bin may be closed if no more boxes fit into it. The present version */
/* uses a more advanced criterion: First, the set of boxes which fit into */
/* bin "bno" is derived. This is done, by testing whether each box (alone) */
/* fits together with the already placed boxes in the bin. Having derived */
/* the set of additional boxes that (individually) fits into the bin, it is */
/* tested whether a solution exists where all the additional boxes are */
/* placed in the bin. If this is the case, we have found a optimal placing */
/* of the additional boxes, and thus we may close the bin. */

boolean try_close(allinfo *a, item **curr, ntype bno,
                 item *oldf, item **oldl, item **oldlc, ntype *oldnoc, 
                 boolean *oldclosed, int level)
{
  register item *j, *m, *k, *r, *i, *s;
  register stype vol;
  item f[MAXITEMS];
  ntype h, n, b;
  boolean didclose, fits;

  if (level > MAXCLOSE) return FALSE;
  i = *curr; didclose = FALSE;
  for (b = 1; b <= bno; b++) {
    if (i > a->litem) break;
    if (a->closed[b]) continue;
    for (j = a->fitem, m = i, k = f, n = 0, vol = 0; j != m; j++) {
      if (j->bno == b) { *k = *j; k->ref = j; k++; n++; vol += j->vol; }
    }
    if (n == 0) error("bin with no items");
    if (vol < a->BVOL/2) continue;
    for (j = i, h = 0, m = a->litem+1; j != m; j++) {
      if ((j->no > a->n) || (j->no < 1)) error("bad no");
      fits = does_fit(a, j, b);
      if (fits) { *k = *j; k->ref = j; k++; h++; vol += j->vol; }
      if (vol > a->BVOL) break;
    }
    if (vol > a->BVOL) continue;
    if (can_fill(a, f, k-1)) {
      if (!didclose) { /* take backup of table when first bin closed */
        memcpy(oldclosed, a->closed, sizeof(boolean)*(bno+1));
        memcpy(oldf, a->fitem, sizeof(item)*DIF(a->fitem,a->litem));
        *oldl = a->litem; *oldlc = a->lclosed; *oldnoc = a->noc;
      }
      a->closed[b] = TRUE; s = a->lclosed; didclose = TRUE;
      a->noc++; if (a->noc > a->maxclose) a->maxclose = a->noc;
      for (j = f; j != k; j++) { 
        r = j->ref; r->bno = b; r->k = TRUE; 
        r->x = j->x; r->y = j->y; r->z = j->z; 
      }
      for (j = k = a->fitem, m = a->litem+1; j != m; j++) {
        if (j->bno == b) { s++; *s = *j; } else { *k = *j; k++; } 
      }
      a->litem = k-1; a->lclosed = s;
      i -= n; /* reposition current item */
    }
  }
  *curr = i;
  return didclose;
}


/* ======================================================================
                                free_close
   ====================================================================== */

/* Reopen a closed bin when backtracking. */

void free_close(allinfo *a, ntype bno, 
                item *oldf, item *oldl, item *oldlc, ntype oldnoc,
                boolean *oldclosed)
{
  item *j, *m;
  ntype b;
  int *ft;

  a->litem = oldl; a->lclosed = oldlc; a->noc = oldnoc;
  memcpy(a->fitem, oldf, sizeof(item)*DIF(a->fitem,oldl));
  memcpy(a->closed, oldclosed, sizeof(boolean)*(bno+1));
}


/* ======================================================================
				rec_binpack
   ====================================================================== */

/* Recursive algorithm for 3D Bin-packing Problem. In each iteration, the */
/* next item "i" is assigned to every open bin, as well as to a new bin. */

void rec_binpack(allinfo *a, item *i, int bno, ntype lb, int level)
{
  item of[MAXITEMS], *ol, *ox;
  boolean ocl[MAXITEMS];
  ntype b, oc;
  boolean more;

  if (bno >= a->z) return; /* used too many bins */
  if (a->z == a->lb) return; /* optimal solution found */
  a->iterates++;
  if (a->iterates % 100 == 0) timelimit(maxtime);
  if (stopped) return;

  if (i == a->litem+1) {
    /* all items assigned, must be better solution */
    savesol(a, a->fitem, a->litem, bno);
  } else {
    more = try_close(a, &i, bno, of, &ol, &ox, &oc, ocl, level);
    if (i == a->litem+1) { /* all items went into closed bins */
      savesol(a, a->fitem, a->litem, bno);
    } else {
      if (more) lb = a->noc + bound_two(a, a->fitem, a->litem);
      if (lb < a->z) {
        for (b = 1; b <= bno; b++) {
          if (a->closed[b]) continue; /* cannot add to closed bin */
            if (does_fit(a, i, b)) {
  	    i->bno = b;
	    rec_binpack(a, i+1, bno, lb, level+1);
	    i->bno = 0;
  	  }
        }
        i->bno = bno+1; i->x = i->y = i->z = 0;
        a->closed[i->bno] = FALSE;
        rec_binpack(a, i+1, bno+1, lb, level+1);
        i->bno = 0;
      }
    }
    /* restore */
    if (more) free_close(a, bno, of, ol, ox, oc, ocl);
  }
}


/* **********************************************************************
   **********************************************************************
			     Main procedure
   **********************************************************************
   ********************************************************************** */

/* ======================================================================
				clearitems
   ====================================================================== */

void clearitems(allinfo *a)
{
  item *i, *m;

  for (i = a->fitem, m = a->litem+1; i != m; i++) {
    i->x = i->y = i->z = i->bno = 0; i->k = FALSE; i->vol = VOL(i);
  }
  /* sort nonincreasing volume */
  qsort(a->fitem, (m-a->fitem), sizeof(item), (funcptr) vcomp);
}


/* ======================================================================
				inititems
   ====================================================================== */

void inititems(allinfo *a, int *w, int *h, int *d)
{
  item *i, *m;
  int k;

  for (i = a->fitem, m = a->litem+1, k = 0; i != m; i++, k++) {
    i->no = k+1; i->w = w[k]; i->h = h[k]; i->d = d[k];
  }

  clearitems(a);
}


/* ======================================================================
				returnitems
   ====================================================================== */

void returnitems(allinfo *a, int *x, int *y, int *z, int *bno)
{
  item *i, *m;
  int k;

  for (i = a->fopt, m = a->lopt+1; i != m; i++) {
    k = i->no-1; x[k] = i->x; y[k] = i->y; z[k] = i->z; bno[k] = i->bno;
  }
}


/* ======================================================================
				binpack3d
   ====================================================================== */

void binpack3d(int n, int W, int H, int D,
               int *w, int *h, int *d, 
               int *x, int *y, int *z, int *bno,
               int *lb, int *ub, int timelimit)
{
  allinfo a;
  char s[100], t[100];
  item t0[MAXITEMS], t1[MAXITEMS], t2[MAXITEMS], t3[MAXITEMS];
  boolean cl[MAXITEMS];
  int zbefore;

  /* start the timer */ 
  timer(NULL); stopped = FALSE; maxtime = timelimit;
 
  /* copy info to a structure */
  if (n+1 > MAXITEMS) error("too big instance");
  a.n = n; a.W = W; a.H = H; a.D = D;
  a.fitem    = t0; 
  a.litem    = a.fitem + a.n - 1; 
  a.fsol     = t1;
  a.lsol     = a.fsol + a.n - 1;
  a.fopt     = t2;
  a.lopt     = a.fopt + a.n - 1;
  a.fclosed  = t3;
  a.lclosed  = a.fclosed - 1;
  a.noc      = 0;
  a.closed   = cl;
  a.BVOL     = W * (ptype) H * D;
  a.optimal  = FALSE;
  a.maxfill  = 0;
  a.exfill   = 0;
  a.iterates = 0;
  a.didpush  = 0;
  a.maxclose = 0;
  a.z        = a.n+1;

  /* copy items to internal structure */
  inititems(&a, w, h, d);

  /* find bounds */
  a.bound0 = bound_zero(&a, a.fitem, a.litem);
  a.bound1 = bound_one(&a, a.fitem, a.litem);
  a.bound2 = bound_two(&a, a.fitem, a.litem);
  a.lb = a.bound2;

  /* find heuristic solution */
  dfirst3_heuristic(&a);
  mcut3_heuristic(&a);

  /* outer tree enummeration */
  clearitems(&a); /* clear positions */
  zbefore = a.z;
  rec_binpack(&a, a.fitem, 0, a.lb, 1);
  timer(&(a.time));

  /* check found solution */
  /* checksol(&a, a.fopt, a.lopt); */

  /* copy items back to arrays */
  returnitems(&a, x, y, z, bno);
  *ub = a.z;
  *lb = (stopped ? a.lb : a.z);
}


