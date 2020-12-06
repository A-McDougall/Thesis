import sys
import string
import os																																																																							# Used for clearing the screen
import math
import csv																																																																						# Used for reading in data	
import time
import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.compiler
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

os.system('clear')

################################
# ----- Define Variables ----- #
SUB_PIXELS			 = 100*100			# If this value is changed, change the "#define SUB_PIXELS 10000" within the C code	
SUB_PIXELS_DIM_X	 = 500			# If this value is changed, change the "#define SUB_PIXELS_DIM_X %%%" Within the C code	

#-- For u0, phi search space.
MAX_THREADS			=np.int32(512)		# If this value is changed, change the "#define MAX_THREADS 512" within the C code
DATA_LIMIT			=np.int32(1024)		# This value cannot exceed 3584

#-- For Mag. Map Generation.
num_points			=np.int32(1000)			# In 1 dimension
threads_per_block	=np.int32(500)		# This number cannot exceed MAX_THREADS +1

#-- LC generation parameter.
t					=np.int32(100000)					# No. of points on the LC

MAX_DATA_SITES		=	10				# If this value is changed, change the "#define MAX_DATA_SITES %%" within the C code	

t0					=np.float64(0)
tE					=np.float64(0)
u0					=np.float64(0)
phi					=np.float64(0)
rho					=np.float64(0)
d					=np.float64(0)
q					=np.float64(0)
e1					=np.float64(0)
e2					=np.float64(0)
a					=np.float64(0)
b					=np.float64(0)
c					=np.float64(0)
q					=np.float64(0)
vt					=np.float64(0)
vtn					=np.float64(0)
step_size			=np.float64(0)
dx					=np.float64(0)
x0					=np.float64(0)
dy					=np.float64(0)
y0					=np.float64(0)
current_best_chi2	=np.float64(sys.maxint)
current_best_d		=np.float64(sys.maxint)
current_best_q		=np.float64(sys.maxint)
current_best_u0		=np.float64(sys.maxint)
current_best_phi	=np.float64(sys.maxint)
current_best_tE		=np.float64(sys.maxint)
current_best_t0		=np.float64(sys.maxint)
current_best_Fo		=np.float64(sys.maxint)
current_best_rho	=np.float64(sys.maxint)
map_limits			=np.float64(2.25)									# Redundant, no longer required as dynamic map sizing.

i					=np.int32(0)
j					=np.int32(0)


tau					= np.zeros([t],np.float64)
u1					= np.zeros([t],np.float64)
u2					= np.zeros([t],np.float64)
A_interp			= np.zeros([t],np.float64)
A					= np.zeros((num_points,num_points),np.float64)
zeta				= np.complex64([0]*t)

trajectory_x		= np.array((num_points,num_points),np.float64)
trajectory_y		= np.array((num_points,num_points),np.float64)

blockshape_A		= (int(threads_per_block), 1, 1)
gridshape_A			= (int(num_points)/2, int(num_points)/int(threads_per_block))

# Initial search u0, phi local increased area search resoltuion
u0_local_steps			= np.int32(50)																																																								# calc the array of values u0 to be searched
phi_local_steps			= np.int32(50)
u0_local_area			= np.float32(2.5)
phi_local_area			= np.float32(2.5)


# Follow up higher resolution search resolution parameters
d_hr_steps			= np.int32(15)
q_hr_steps			= np.int32(15)
u0_hr_steps			= np.int32(50)
phi_hr_steps		= np.int32(50)

d_hr_area			= np.float32(1.5)
q_hr_area			= np.float32(1.5)
u0_hr_area			= np.float32(2.5)
phi_hr_area			= np.float32(2.5)



#-- Data loading

# REMEMBER TO INSERT A NEW ESTIMATE FOR t0 & tE.
file_id ='events/MOA_2012_BLG_0486/'																																																# the Id of the file which the code will be working on.
# REMEMBER TO INSERT A NEW ESTIMATE FOR t0 & tE.

with open("current_working_file.txt", "w") as f:
	f.write(file_id)																																																																				# Write the data file name for other programs to use

#sys.exit()

#-- d paramter search space
ld		= -0.05#-0.7																																																																					# Lower limit for the parameter d
ud		= 0.0#0.7																																																																								# Upper limit for the parameter d
dd		= 0.05																																																																						# Step size for the parameter d
#-- q paramter search space																																																																# repeat for each paramter below...
lq		= 0.8#np.log10(10**-5)					# = 10^-5
uq		= 1#np.log10(10**0)					# = 1
dq		= 0.2								# = log(1.9952...)
#-- rho paramter search space
lrho	= 0.001
urho	= 0.001
drho	= 0.001
#-- phi paramter search space
lphi	= (2*np.pi) / 200
uphi	= (2*np.pi)
dphi	= (2*np.pi) / 200
#-- u0 paramter search space
lu0		= np.log10(0.0001)				# 
uu0		= 0.2							# uu0 = 1.58
du0		= 0.04
#-- t0 paramter search space
est_t0	= 6140																																																																				# estimate of models t0 value, for simplex starting point
#-- tE paramter search space
est_tE	= 90																																																																					# estimate of models tE value, for simplex starting point
#-- Al parameter estimate				# A_max lower limit
Al		= 1

map_limits = (2.0/3.0)*(math.atan(2*ud-7.2))+2.46
with open(str(file_id)+"output/global_vars_"+str(file_id[7:24])+".txt","w") as f:
	f.write('%f, %f, %f, %f, %f, %f'% (map_limits, num_points, threads_per_block, SUB_PIXELS, SUB_PIXELS_DIM_X, DATA_LIMIT))

###############################

# Time devices, initialise variables for timing
start = drv.Event()	# To use: start.record()
end = drv.Event()	# To use: end.record()

###############################



def dims(map_limits, num_points):																																																												# Set up the limits of the magnification map
	x0 = -map_limits
	dx = (map_limits - x0)/(num_points-1)
	dy = dx
	y0 = x0
	
	return dx, x0, dy, y0


def zeta(u0, phi, t, t0, tE, map_limits):																																																																# Produce the (x,y) coordinates on the mag map (u1,u2) for a given number of points (t).
	step_size = ((t0+map_limits*tE) - (t0-map_limits*tE))/(t)	
	tau = np.arange(t0-map_limits*tE,t0+map_limits*tE, step_size)
	#zeta		= np.zeros([t],np.complex64)
	
	u1 = np.float32( ((tau - t0)/tE)*np.cos(phi) - u0*np.sin(phi) )																																															# the x-coorindate
	u2 = np.float32( ((tau - t0)/tE)*np.sin(phi) + u0*np.cos(phi) )																																															# the y-coorindate
	
	return u1, u2, tau


def rd_zeta(u0, phi, ts, t0, tE, map_limits):																																																														# produce the (x,y) coordinates of real data (ts)	
	tau = ts
	#zeta		= np.zeros([t],np.complex64)
	
	u1 = np.float32( ((tau - t0)/tE)*np.cos(phi) - u0*np.sin(phi) )
	u2 = np.float32( ((tau - t0)/tE)*np.sin(phi) + u0*np.cos(phi) )
	
	return u1, u2, tau


gpu_mag_maps= SourceModule("""
	#include <pycuda-complex.hpp>
	#include <stdio.h>
	#include <math.h>
	#include <float.h>
	#define	COMPLEXTYPE	pycuda::complex<double>
	
	#include "texture_fetch_functions.h"
	#include "texture_types.h"
	
	#define NRANSI																																																																									// Used by Numerical Recipies (NR)
	#include "/home/amm315/code/binary/include/nrutil.h"																																																								// 
	#define EPSS 1.0e-7																																																																								// For NR ZRoots
	#define MR 8																																																																											// v
	#define MT 10																																																																										// v
	#define MAXIT (MT*MR)																																																																				// v
	#define EPS 2.0e-6																																																																								// v
	#define MAXM 100																																																																								// v
	#define	TRUE	1																																																																									// v
	#define	FALSE	0																																																																									// ---
//																																																																																		// For NR Amoeba
	#define NDIM 2						// This value needs to be included in the #define GET_SUM " for (...j<=NDIM...)" and in p[(i-1)*NDIM+j]
	#define MP 3																																																																											// v
	#define FTOL 1.0e-5																																																																							// v
	#define NMAX 5000																																																																							// v
	#define GET_PSUM \
						for (j=1;j<=2;j++) {\
						for (sum=0.0,i=1;i<=mpts;i++) sum += p[(i-1)*2+j];\
						psum[j]=sum;}
	#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}																																																										// ---
//																																																																																		// For NR Zroots
	#define	NUM_POLYS	6																																																																					// v
	#define	MAX_THREADS 512				// Must be a value of 2^n																																																																// v
	#define	MAX_DATA_POINTS 3584																																																																				// v
	#define SUB_PIXELS 10000																																																																			// v
	#define SUB_PIX_DIM 100 			// sqrt(SUB_PIXELS)
	#define INV_SUB_PIX_DIM 0.01 		// = 1/(sqrt(SUB_PIXELS))
	#define SUB_PIXELS_DIM_X 500																																																																// ---
	
	#define	MAX_DATA_SITES	10
	
	texture<float, 2> texref;	
	
//////////////////////////////////////////////////////////////	zroots - root solving routine

	__device__ void laguer(COMPLEXTYPE as[], int m, COMPLEXTYPE *x, int *its)																																										// NR routine 'laguer', read "Numerical Recipes in C 2nd Ed.", and the NR exercise book [C] 2nd Ed.		
	{
		int iter,j;
		double abx,abp,abm,err;
		COMPLEXTYPE dx,x1,b,d,f,g,h,sq,gp,gm,g2;
		double frac[MR+1] = {0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0};
		
		for (iter=1;iter<=MAXIT;iter++)
		{
			*its=iter;
			b=as[m];
			err=abs(b);
			d=f=COMPLEXTYPE((double)0.0,(double)0.0);
			abx=abs(*x);
			for (j=m-1;j>=0;j--)
			{
				f= (*x * f) + d;
				d= (*x * d) + b;
				b= (*x * b) + as[j];
				err=abs(b) + abx*err;
			}
			err *= EPSS;
			if (abs(b) <= err) return;
			g= d / b;
			g2= g * g;
			h= (g2 - ((double)2.0 * (f / b)));
			sq=sqrt(((double)(m-1) * (((double)m * h) - g2))); // line 46
			gp= g + sq;
			gm= g - sq;
			abp=abs(gp);
			abm=abs(gm);
			if (abp < abm) gp=gm;
			dx=((fmax(abp,abm) > (double)0.0 ? (COMPLEXTYPE((double)m , (double)0.0) / gp)
				: (exp(log(1+abx)) * COMPLEXTYPE(cos((double)iter),sin((double)iter)))));
			x1= *x - dx;
			if (real(*x) == real(x1) && imag(*x) == imag(x1)) return;
			if (iter % MT) *x=x1;
			else *x= *x - (frac[iter/MT] * dx);
		}
//		nrerror("too many iterations in laguer");
		return;
	}
	
	
	__device__ void zroots(COMPLEXTYPE as[], int m, COMPLEXTYPE roots[], int polish)																																							// NR routine 'zroots', read "Numerical Recipes in C 2nd Ed.", and the NR exercise book [C] 2nd Ed.
	{
		void laguer(COMPLEXTYPE as[], int m, COMPLEXTYPE *x, int *its);
		int i,its,j,jj;
		COMPLEXTYPE x,b,c,ad[MAXM];

		for (j=0;j<=m;j++) ad[j]=as[j];
		for (j=m-1;j>=1-1;j--)
		{
			x=COMPLEXTYPE((double)0.0,(double)0.0);
			laguer(ad,j,&x,&its);
			if (abs(imag(x)) <= (double)2.0*EPS*abs(real(x)))
			{
				x=COMPLEXTYPE(real(x),(double)0.0);
			}
			roots[j]=x;
			b=ad[j];
			for (jj=j-1;jj>=0;jj--)
			{
				c=ad[jj];
				ad[jj]=b;
				b= (x * b) + c;
			}
		}
		if (polish)
			for (j=1-1;j<=m-1;j++)
				laguer(as,m,&roots[j],&its);
		for (j=2-1;j<=m-1;j++)
		{
			x=roots[j];
			for (i=j-1;i>=1-1;i--)
			{
				if (real(roots[i]) <= real(x)) break;
				roots[i+1]=roots[i];
			}
			roots[i+1]=x;
		}
	}
//////////////////////// zroots - end


//////////////////////////////////////////////////////////////	Jenkins-Traub


	// MCON PROVIDES MACHINE CONSTANTS USED IN VARIOUS PARTS OF THE PROGRAM.
	// THE USER MAY EITHER SET THEM DIRECTLY OR USE THE STATEMENTS BELOW TO
	// COMPUTE THEM. THE MEANING OF THE FOUR CONSTANTS ARE -
	// ETA       THE MAXIMUM RELATIVE REPRESENTATION ERROR WHICH CAN BE DESCRIBED
	//           AS THE SMALLEST POSITIVE FLOATING-POINT NUMBER SUCH THAT
	//           1.0_dp + ETA &gt; 1.0.
	// INFINY    THE LARGEST FLOATING-POINT NUMBER
	// SMALNO    THE SMALLEST POSITIVE FLOATING-POINT NUMBER
	// BASE      THE BASE OF THE FLOATING-POINT NUMBER SYSTEM USED
	//
	__device__ static void mcon( double *eta, double *infiny, double *smalno, double *base )
   {
	   *base = (double)2.0;
	   *eta = DBL_EPSILON;
	   *infiny = DBL_MAX;
	   *smalno = DBL_MIN;
   }


	// MODULUS OF A COMPLEX NUMBER AVOIDING OVERFLOW.
	//
	__device__ double cmod(double r, double i)
   {
	   double ar, ai;

	   ar = fabs( r );
	   ai = fabs( i );
	   if( ar < ai )
		  return ai * sqrt( 1.0 + pow( ( ar / ai ), 2.0 ) );

	   if( ar > ai )
		  return ar * sqrt( 1.0 + pow( ( ai / ar ), 2.0 ) );

	   return ar * sqrt( 2.0 );
   }
   
   	// EVALUATES A POLYNOMIAL  P  AT  S  BY THE HORNER RECURRENCE
	// PLACING THE PARTIAL SUMS IN Q AND THE COMPUTED VALUE IN PV.
	//  
	__device__ void polyev(const int nn, const double sr, const double si, const double pr[], const double pi[], double qr[], double qi[], double *pvr, double *pvi )  
   {
		int i;
		double t;

		qr[ 0 ] = pr[ 0 ];
		qi[ 0 ] = pi[ 0 ];
		*pvr = qr[ 0 ];
		*pvi = qi[ 0 ];

		for( i = 1; i <= nn; i++ )
		{
		  t = ( *pvr ) * sr - ( *pvi ) * si + pr[ i ];
		  *pvi = ( *pvr ) * si + ( *pvi ) * sr + pi[ i ];
		  *pvr = t;
		  qr[ i ] = *pvr;
		  qi[ i ] = *pvi;
		}
   }

	// COMPLEX DIVISION C = A/B, AVOIDING OVERFLOW.
	//
	__device__ void cdivid(double ar, double ai, double br, double bi, double *cr, double *ci )
   {
	   double r, d, t, infin;

	   if( br == 0 && bi == 0 )
		  {
		  // Division by zero, c = infinity
		  mcon( &t, &infin, &t, &t );
		  *cr = infin;
		  *ci = infin;
		  return;
		  }

	   if( fabs( br ) < fabs( bi ) )
		  {
		  r = br/ bi;
		  d = bi + r * br;
		  *cr = ( ar * r + ai ) / d;
		  *ci = ( ai * r - ar ) / d;
		  return;
		  }

	   r = bi / br;
	   d = br + r * bi;
	   *cr = ( ar + ai * r ) / d;
	   *ci = ( ai - ar * r ) / d;
   }

	// BOUNDS THE ERROR IN EVALUATING THE POLYNOMIAL BY THE HORNER RECURRENCE.
	// QR,QI - THE PARTIAL SUMS
	// MS    -MODULUS OF THE POINT
	// MP    -MODULUS OF POLYNOMIAL VALUE
	// ARE, MRE -ERROR BOUNDS ON COMPLEX ADDITION AND MULTIPLICATION
	//
	__device__ double errev( const int nn, const double qr[], const double qi[], const double ms, const double mp, const double are, const double mre )
   {
	   int i;
	   double e;

	   e = cmod( qr[ 0 ], qi[ 0 ] ) * mre / ( are + mre );
	   for( i = 0; i <= nn; i++ )
		  e = e * ms + cmod( qr[ i ], qi[ i ] );

	   return e * ( are + mre ) - mp * mre;
   }

	// COMPUTES  T = -P(S)/H(S).
	// BOOL   - LOGICAL, SET TRUE IF H(S) IS ESSENTIALLY ZERO.
	__device__ void calct( int *bol, int *nn, double *sr, double *si, double *hr, double *hi, double *qhr, double *qhi, double *are, double *pvr, double *pvi, double *tr, double *ti)
   {
	   int n;
	   double hvr, hvi;

	   n = *nn;

	   // evaluate h(s)
	   polyev( n - 1, *sr, *si, hr, hi, qhr, qhi, &hvr, &hvi );
	   *bol = cmod( hvr, hvi ) <= *are * 10 * cmod( hr[ n - 1 ], hi[ n - 1 ] ) ? 1 : 0;
	   if( !*bol )
		  {
		  cdivid( -*pvr, -*pvi, hvr, hvi, tr, ti );
		  return;
		  }

	   tr = 0;
	   ti = 0;
   }

	// COMPUTES  THE DERIVATIVE  POLYNOMIAL AS THE INITIAL H
	// POLYNOMIAL AND COMPUTES L1 NO-SHIFT H POLYNOMIALS.
	//
	__device__ static void noshft( const int l1 , int *nn, double *hr, double *hi, double *pr, double *pi, double *tr, double *ti, double *eta)
   {
	   int i, j, jj, n, nm1;
	   double xni, t1, t2;

	   n = *nn;
	   nm1 = n - 1;
	   for( i = 0; i < n; i++ )
		  {
		  xni = *nn - i;
		  hr[ i ] = xni * pr[ i ] / n;
		  hi[ i ] = xni * pi[ i ] / n;
		  }
	   for( jj = 1; jj <= l1; jj++ )
		  {
		  if( cmod( hr[ n - 1 ], hi[ n - 1 ] ) > *eta * 10 * cmod( pr[ n - 1 ], pi[ n - 1 ] ) )
			 {
			 cdivid( -pr[ *nn ], -pi[ *nn ], hr[ n - 1 ], hi[ n - 1 ], tr, ti );
			 for( i = 0; i < nm1; i++ )
				{
				j = *nn - i - 1;
				t1 = hr[ j - 1 ];
				t2 = hi[ j - 1 ];
				hr[ j ] = *tr * t1 - *ti * t2 + pr[ j ];
				hi[ j ] = *tr * t2 + *ti * t1 + pi[ j ];
				}
			 hr[ 0 ] = pr[ 0 ];
			 hi[ 0 ] = pi[ 0 ];
			 }
		  else
			 {
			 // If the constant term is essentially zero, shift H coefficients
			 for( i = 0; i < nm1; i++ )
				{
				j = *nn - i - 1;
				hr[ j ] = hr[ j - 1 ];
				hi[ j ] = hi[ j - 1 ];
				}
			 hr[ 0 ] = 0;
			 hi[ 0 ] = 0;
			 }
		  }
   }
   

	// CALCULATES THE NEXT SHIFTED H POLYNOMIAL.
	// BOOL   -  LOGICAL, IF .TRUE. H(S) IS ESSENTIALLY ZERO
	//
	__device__ void nexth( const int bol, int *nn, double *hr, double *hi, double *qpr, double *qpi, double *qhr, double *qhi, double *tr, double *ti)
   {
	   int j, n;
	   double t1, t2;

	   n = *nn;
	   if( !bol )
		  {
		  for( j = 1; j < n; j++ )
			 {
			 t1 = qhr[ j - 1 ];
			 t2 = qhi[ j - 1 ];
			 hr[ j ] = *tr * t1 - *ti * t2 + qpr[ j ];
			 hi[ j ] = *tr * t2 + *ti * t1 + qpi[ j ];
			 }
		  hr[ 0 ] = qpr[ 0 ];
		  hi[ 0 ] = qpi[ 0 ];
		  return;
		  }

	   // If h[s] is zero replace H with qh
	   for( j = 1; j < n; j++ )
		  {
		  hr[ j ] = qhr[ j - 1 ];
		  hi[ j ] = qhi[ j - 1 ];
		  }
	   hr[ 0 ] = 0;
	   hi[ 0 ] = 0;
   }

	// CARRIES OUT THE THIRD STAGE ITERATION.
	// L3 - LIMIT OF STEPS IN STAGE 3.
	// ZR,ZI   - ON ENTRY CONTAINS THE INITIAL ITERATE, IF THE
	//           ITERATION CONVERGES IT CONTAINS THE FINAL ITERATE ON EXIT.
	// CONV    -  .TRUE. IF ITERATION CONVERGES
	//
	__device__ void vrshft(const int l3, double *zr, double *zi, int *conv, int *nn, double *sr, double *si, double *pr, double *pi, double *qpr, double *qpi, double *pvr, double *pvi, double *are, double *mre, double *eta, double *hr, double *hi, double *qhr, double *qhi, double *tr, double *ti, double *infin)
   {
	   int b, bol;
	   int i, j;
	   double mp, ms, omp, relstp, r1, r2, tp;

	   *conv = 0;
	   b = 0;
	   *sr = *zr;
	   *si = *zi;

	   // Main loop for stage three
	   for( i = 1; i <= l3; i++ )
		  {
		  // Evaluate P at S and test for convergence
		  polyev( *nn, *sr, *si, pr, pi, qpr, qpi, pvr, pvi );
		  mp = cmod( *pvr, *pvi );
		  ms = cmod( *sr, *si );
		  if( mp <= 20 * errev( *nn, qpr, qpi, ms, mp, *are, *mre ) )
			 {
			 // Polynomial value is smaller in value than a bound onthe error
			 // in evaluationg P, terminate the ietartion
			 *conv = 1;
			 *zr = *sr;
			 *zi = *si;
			 return;
			 }
		  if( i != 1 )
			 {
			 if( !( b || mp < omp || relstp >= 0.05 ) )
				{
				// Iteration has stalled. Probably a cluster of zeros. Do 5 fixed 
				// shift steps into the cluster to force one zero to dominate
				tp = relstp;
				b = 1;
				if( relstp < *eta ) tp = *eta;
				r1 = sqrt( tp );
				r2 = *sr * ( 1 + r1 ) - *si * r1;
				*si = *sr * r1 + *si * ( 1 + r1 );
				*sr = r2;
				polyev( *nn, *sr, *si, pr, pi, qpr, qpi, pvr, pvi );
				for( j = 1; j <= 5; j++ )
				   {
				   calct(&bol, nn, sr, si, hr, hi, qhr, qhi, are, pvr, pvi, tr, ti);
				   nexth(bol, nn, hr, hi, qpr, qpi, qhr, qhi, tr, ti);
				   }
				omp = *infin;
				goto _20;
				}
			 
			 // Exit if polynomial value increase significantly
			 if( mp *0.1 > omp ) return;
			 }

		  omp = mp;

		  // Calculate next iterate
	_20:  calct(&bol, nn, sr, si, hr, hi, qhr, qhi, are, pvr, pvi, tr, ti);
		  nexth(bol, nn, hr, hi, qpr, qpi, qhr, qhi, tr, ti);
		  calct(&bol, nn, sr, si, hr, hi, qhr, qhi, are, pvr, pvi, tr, ti);
		  if( !bol )
			 {
			 relstp = cmod( *tr, *ti ) / cmod( *sr, *si );
			 *sr += *tr;
			 *si += *ti;
			 }
		  }
   }

	// COMPUTES L2 FIXED-SHIFT H POLYNOMIALS AND TESTS FOR CONVERGENCE.
	// INITIATES A VARIABLE-SHIFT ITERATION AND RETURNS WITH THE
	// APPROXIMATE ZERO IF SUCCESSFUL.
	// L2 - LIMIT OF FIXED SHIFT STEPS
	// ZR,ZI - APPROXIMATE ZERO IF CONV IS .TRUE.
	// CONV  - LOGICAL INDICATING CONVERGENCE OF STAGE 3 ITERATION
	//
	__device__ void fxshft(int l2, double *zr, double *zi, int *conv, int *nn, double *sr, double *si, double *pr, double *pi, double *qpr, double *qpi, double *pvr, double *pvi, double *tr, double *ti, double *hr, double *hi, double *qhr, double *qhi, double *are, double *shr, double *shi, double *mre, double *eta, double *infin)
   {
	   int i, j, n;
	   int test, pasd, bol;
	   double otr, oti, svsr, svsi;

	   n = *nn;
	   polyev(*nn, *sr, *si, pr, pi, qpr, qpi, pvr, pvi );
	   test = 1;
	   pasd = 0;

	   // Calculate first T = -P(S)/H(S)
	   calct(&bol, nn, sr, si, hr, hi, qhr, qhi, are, pvr, pvi, tr, ti);

	   // Main loop for second stage
	   for( j = 1; j <= l2; j++ )
		  {
		  otr = *tr;
		  oti = *ti;

		  // Compute the next H Polynomial and new t
		  nexth(bol, nn, hr, hi, qpr, qpi, qhr, qhi, tr, ti);
		  calct( &bol, nn, sr, si, hr, hi, qhr, qhi, are, pvr, pvi, tr, ti);
		  *zr = *sr + *tr;
		  *zi = *si + *ti;

		  // Test for convergence unless stage 3 has failed once or this
		  // is the last H Polynomial
		  if( !( bol || !test || j == 12 ) )
			 if( cmod( *tr - otr, *ti - oti ) < 0.5 * cmod( *zr, *zi ) )
				{
				if( pasd )
				   {
				   // The weak convergence test has been passwed twice, start the third stage
				   // Iteration, after saving the current H polynomial and shift
				   for( i = 0; i < n; i++ )
					  {
					  shr[ i ] = hr[ i ];
					  shi[ i ] = hi[ i ];
					  }
				   svsr = *sr;
				   svsi = *si;
				   vrshft( 10, zr, zi, conv, nn, sr, si, pr, pi, qpr, qpi, pvr, pvi, are, mre, eta, hr, hi, qhr, qhi, tr, ti, infin);
				   if( *conv ) return;

				   //The iteration failed to converge. Turn off testing and restore h,s,pv and T
				   test = 0;
				   for( i = 0; i < n; i++ )
					  {
					  hr[ i ] = shr[ i ];
					  hi[ i ] = shi[ i ];
					  }
				   *sr = svsr;
				   *si = svsi;
				   polyev( *nn, *sr, *si, pr, pi, qpr, qpi, pvr, pvi );
				   calct(&bol, nn, sr, si, hr, hi, qhr, qhi, are, pvr, pvi, tr, ti);
				   continue;
				   }
				pasd = 1;
				}
			 else
				pasd = 0;
		  }

	   // Attempt an iteration with final H polynomial from second stage
	   vrshft( 10, zr, zi, conv, nn, sr, si, pr, pi, qpr, qpi, pvr, pvi, are, mre, eta, hr, hi, qhr, qhi, tr, ti, infin);
   }

	// CAUCHY COMPUTES A LOWER BOUND ON THE MODULI OF THE ZEROS OF A
	// POLYNOMIAL - PT IS THE MODULUS OF THE COEFFICIENTS.
	//
	__device__ void cauchy(int nn, double pt[], double q[], double *fn_val )
   {
	   int i, n;
	   double x, xm, f, dx, df;

	   pt[ nn ] = -pt[ nn ];

	   // Compute upper estimate bound
	   n = nn;
	   x = exp( log( -pt[ nn ] ) - log( pt[ 0 ] ) ) / n;
	   if( pt[ n - 1 ] != 0 )
		  {
		  // Newton step at the origin is better, use it
		  xm = -pt[ nn ] / pt[ n - 1 ];
		  if( xm < x ) x = xm;
		  }

	   // Chop the interval (0,x) until f < 0
	   while(1)
		  {
		  xm = x * 0.1;
		  f = pt[ 0 ];
		  for( i = 1; i <= nn; i++ )
			 f = f * xm + pt[ i ];
		  if( f <= 0 )
			 break;
		  x = xm;
		  }
	   dx = x;
	   
	   // Do Newton iteration until x converges to two decimal places
	   while( fabs( dx / x ) > 0.005 )
		  {
		  q[ 0 ] = pt[ 0 ];
		  for( i = 1; i <= nn; i++ )
			 q[ i ] = q[ i - 1 ] * x + pt[ i ];
		  f = q[ nn ];
		  df = q[ 0 ];
		  for( i = 1; i < n; i++ )
			 df = df * x + q[ i ];
		  dx = f / df;
		  x -= dx;
		  }

	   *fn_val = x;
   }

	// RETURNS A SCALE FACTOR TO MULTIPLY THE COEFFICIENTS OF THE POLYNOMIAL.
	// THE SCALING IS DONE TO AVOID OVERFLOW AND TO AVOID UNDETECTED UNDERFLOW
	// INTERFERING WITH THE CONVERGENCE CRITERION.  THE FACTOR IS A POWER OF THE
	// BASE.
	// PT - MODULUS OF COEFFICIENTS OF P
	// ETA, INFIN, SMALNO, BASE - CONSTANTS DESCRIBING THE FLOATING POINT ARITHMETIC.
	//
	__device__ double scale(int nn, double pt[], double eta, double infin, double smalno, double base )
   {
	   int i, l;
	   double hi, lo, max, min, x, sc;
	   double fn_val;

	   // Find largest and smallest moduli of coefficients
	   hi = sqrt( infin );
	   lo = smalno / eta;
	   max = 0;
	   min = infin;

	   for( i = 0; i <= nn; i++ )
		  {
		  x = pt[ i ];
		  if( x > max ) max = x;
		  if( x != 0 && x < min ) min = x;
		  }

	   // Scale only if there are very large or very small components
	   fn_val = 1;
	   if( min >= lo && max <= hi ) return fn_val;
	   x = lo / min;
	   if( x <= 1 )
		  sc = 1 / ( sqrt( max )* sqrt( min ) );
	   else
		  {
		  sc = x;
		  if( infin / sc > max ) sc = 1;
		  }
	   l = (int)( log( sc ) / log(base ) + 0.5 );
	   fn_val = pow( base , l );
	   return fn_val;
   }
   
   
	__device__ int cpoly(const double *opr, const double *opi, int degree, double *zeror, double *zeroi)
   {
   		double sr, si, tr, ti, pvr, pvi, are, mre, eta, infin;
		int nn;
		
	   int cnt1, cnt2, idnn2, i, conv;
	   double pr[NUM_POLYS], pi[NUM_POLYS], hr[NUM_POLYS], hi[NUM_POLYS], qpr[NUM_POLYS], qpi[NUM_POLYS], qhr[NUM_POLYS], qhi[NUM_POLYS], shr[NUM_POLYS], shi[NUM_POLYS];
	   double xx, yy, cosr, sinr, smalno, base, xxx, zr, zi, bnd;

	   mcon( &eta, &infin, &smalno, &base );
	   are = eta;
	   mre = 2.0 * sqrt( 2.0 ) * eta;
	   xx = 0.70710678;
	   yy = -xx;
	   cosr = -0.060756474;
	   sinr = -0.99756405;
	   nn = degree;  

	   // Algorithm fails if the leading coefficient is zero
	   if( opr[ 0 ] == 0 && opi[ 0 ] == 0 )
		  return -1;

	   // Allocate arrays
//	   pr = new double [ degree+1 ];
//	   pi = new double [ degree+1 ];
//	   hr = new double [ degree+1 ];
//	   hi = new double [ degree+1 ];
//	   qpr= new double [ degree+1 ];
//	   qpi= new double [ degree+1 ];
//	   qhr= new double [ degree+1 ];
//	   qhi= new double [ degree+1 ];
//	   shr= new double [ degree+1 ];
//	   shi= new double [ degree+1 ];

	   // Remove the zeros at the origin if any
	   while( opr[ nn ] == 0 && opi[ nn ] == 0 )
		  {
		  idnn2 = degree - nn;
		  zeror[ idnn2 ] = 0;
		  zeroi[ idnn2 ] = 0;
		  nn--;
		  }

	   // Make a copy of the coefficients
	   for( i = 0; i <= nn; i++ )
		  {
		  pr[ i ] = opr[ i ];
		  pi[ i ] = opi[ i ];
		  shr[ i ] = cmod( pr[ i ], pi[ i ] );
		  }

	   // Scale the polynomial
	   bnd = scale( nn, shr, eta, infin, smalno, base );
	   if( bnd != 1 )
		  for( i = 0; i <= nn; i++ )
			 {
			 pr[ i ] *= bnd;
			 pi[ i ] *= bnd;
			 }

	search: 
	   if( nn <= 1 )
		  {
		  cdivid( -pr[ 1 ], -pi[ 1 ], pr[ 0 ], pi[ 0 ], &zeror[ degree-1 ], &zeroi[ degree-1 ] );
		  goto finish;
		  }

	   // Calculate bnd, alower bound on the modulus of the zeros
	   for( i = 0; i<= nn; i++ )
		  shr[ i ] = cmod( pr[ i ], pi[ i ] );

	   cauchy( nn, shr, shi, &bnd );
	   
	   // Outer loop to control 2 Major passes with different sequences of shifts
	   for( cnt1 = 1; cnt1 <= 2; cnt1++ )
		  {
		  // First stage  calculation , no shift
		  noshft( 5, &nn, hr, hi, pr, pi, &tr, &ti, &eta);

		  // Inner loop to select a shift
		  for( cnt2 = 1; cnt2 <= 9; cnt2++ )
			 {
			 // Shift is chosen with modulus bnd and amplitude rotated by 94 degree from the previous shif
			 xxx = cosr * xx - sinr * yy;
			 yy = sinr * xx + cosr * yy;
			 xx = xxx;
			 sr = bnd * xx;
			 si = bnd * yy;

			 // Second stage calculation, fixed shift
			 fxshft( 10 * cnt2, &zr, &zi, &conv, &nn, &sr, &si, pr, pi, qpr, qpi, &pvr, &pvi, &tr, &ti, hr, hi, qhr, qhi, &are, shr, shi, &mre, &eta, &infin);
			 if( conv )
				{
				// The second stage jumps directly to the third stage ieration
				// If successful the zero is stored and the polynomial deflated
				idnn2 = degree - nn;
				zeror[ idnn2 ] = zr;
				zeroi[ idnn2 ] = zi;
				nn--;
				for( i = 0; i <= nn; i++ )
				   {
				   pr[ i ] = qpr[ i ];
				   pi[ i ] = qpi[ i ];
				   }
				goto search;
				}
			 // If the iteration is unsuccessful another shift is chosen
			 }
		  // if 9 shifts fail, the outer loop is repeated with another sequence of shifts
		  }

	   // The zerofinder has failed on two major passes
	   // return empty handed with the number of roots found (less than the original degree)
	   degree -= nn;

	finish:
	   // Deallocate arrays
//	   delete [] pr;
//	   delete [] pi;
//	   delete [] hr;
//	   delete [] hi;
//	   delete [] qpr;
//	   delete [] qpi;
//	   delete [] qhr;
//	   delete [] qhi;
//	   delete [] shr;
//	   delete [] shi;

	   return degree;       
   }

//////////////////////// Jenkins-Traub - end


//////////////////////////////////////////////////////////////	Sort comparisons
__device__ void insertion_sort(double *a, COMPLEXTYPE *b, int n)
{
	int k;
	for (k = 1; k < n; ++k)
	{
		double key = a[k];
		COMPLEXTYPE key2 = b[k];
		int i = k - 1;
		while ((i >= 0) && (key < a[i]))
		{
			a[i + 1] = a[i];
			b[i + 1] = b[i];
			--i;
		}
		a[i + 1] = key;
		b[i + 1] = key2;
	}
}
//////////////////////// Sort comparisons - end


//////////////////////////////////////////////////////////////	Magnification Calculations
	__global__ void magnification(double e1, double e2, double a, double dx, double x0, double dy, double y0, double *A)																											// Produce a magnification map
	{	
		int xpos = threadIdx.x + (blockIdx.y * blockDim.x);																																																									// using thread ids, determine the x-position (the u1 coordinate)	
		int ypos = blockIdx.x;																																																																						// using block ids, determine the y-position (the u2 coordinate)
		int n, j, polish;																																																																										// - only calcuate one half of the u2 coordiantes, as the map is symetrical so can be mirrored.
		int m = NUM_POLYS;																																																																						// Number of coefficients in the polynomial to solve.
		double zr[NUM_POLYS], zi[NUM_POLYS], detj[NUM_POLYS], opr[NUM_POLYS], opi[NUM_POLYS], test_ans[NUM_POLYS-1];
		COMPLEXTYPE coef5, coef4, coef3, coef2, coef1, coef0, zetabar, zeta;																																														// COMPLEXTYPE is a special data type from pycuda to help utilise complex numbers in CUDA
		COMPLEXTYPE  z[NUM_POLYS], zeta_test[NUM_POLYS-1], zbar[NUM_POLYS-1], as[NUM_POLYS];
		
	//	Determine the complex conjugate of the source position
		zeta = COMPLEXTYPE(x0 + dx*xpos, y0 + dy*ypos);
		zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
				
	//	Determine the coefficients of the 5th order polynomial
		as[5] = zetabar*zetabar - a*a;
		as[4] = -zeta*pow(zetabar,2) - (double)2*e1*a + zetabar + zeta*pow(a,2) + a;
		as[3] = (double)4*e1*a*zetabar - (double)2*pow(zetabar,2)*pow(a,2) - (double)2*a*zetabar - (double)2*zeta*zetabar + (double)2*pow(a,4);
		as[2] = -(double)2*zeta*pow(a,4) - (double)4*zeta*e1*a*zetabar + (double)2*e1*a + 4*pow(a,3)*e1 + (double)2*zeta*pow(zetabar,2)*pow(a,2) - zeta - a - (double)2*pow(a,3) + (double)2*zeta*a*zetabar;
		as[1] = (double)2*zeta*zetabar*pow(a,2) - (double)4*zeta*e1*a + pow(zetabar,2)*pow(a,4) - (double)4*e1*pow(a,2) - pow(a,6) + (double)2*pow(a,2) + (double)4*pow(e1,2)*pow(a,2) + (double)2*zeta*a + (double)2*zetabar*pow(a,3) - (double)4*e1*pow(a,3)*zetabar;
		as[0] = -zeta*pow(a,2) + zeta*pow(a,6) + pow(a,5) - (double)4*zeta*pow(e1,2)*pow(a,2) - pow(a,3) + (double)4*zeta*zetabar*pow(a,3)*e1 + (double)4*zeta*e1*pow(a,2) - zeta*pow(zetabar,2)*pow(a,4) - pow(a,4)*zetabar - (double)2*zeta*pow(a,3)*zetabar + (double)2*pow(a,3)*e1 - (double)2*e1*pow(a,5);
		
		for  (j=0;j<m;j++)
		{
			opr[j] = real(as[5-j]);
			opi[j] = imag(as[5-j]);		
		}
		
		if ( ((1/e1)-1) > 0.012 )
		{
			polish=FALSE;
			zroots(as,m,z,polish);	
		}
		else
		{
			cpoly(opr, opi, m-1, zr, zi);																																																																							// Call the NR zroots function to solve for the possible roots of the polynomial.	
			for  (j=0;j<m-1;j++)
			{
				z[j] = COMPLEXTYPE(zr[j], zi[j]);	
			}
		}
		
	//	Determine the complex conjugate of the images positions
		n = 0;
		for (j=0;j<m-1;j++)
		{	
			zbar[j] = COMPLEXTYPE(real(z[j]), -imag(z[j]));

		//	Test the possible solutions
			zeta_test[j] = z[j] - (e1 / (zbar[j]-a)) - (e2 / (zbar[j]+a));
			test_ans[j] = abs(zeta_test[j] - zeta);
		}
		
		insertion_sort(test_ans, zbar, m-1);
		
		if (test_ans[m-2] < 0.0001)
		{
			n = 5;
		}
		else
		{
			n = 3;
		}
		
		A[xpos + ypos*2*(gridDim.x)] = 0;																																																																	// Initialise the mag. at the maps pixel to be zero
		A[xpos + 2*(gridDim.x)*blockDim.x*gridDim.y - 2*gridDim.x - ypos*2*(gridDim.x)] = 0;																																								// Mirror this operation across the symmetrical axis
		
		for (j=0;j<n;j++)
		{	
			detj[j] = 1 - abs( pow((e1 / pow(zbar[j] - a , 2)) +  (e2 / pow(zbar[j] + a , 2)) , 2) );																																			// Determine the magnification contribution that the real image solutions cause.
			
			A[xpos + ypos*2*(gridDim.x)] += 1/abs(detj[j]);																																																										// Sum the magnification contributions from all real image solutions of a given source position.
		}
		A[xpos + 2*(gridDim.x)*blockDim.x*gridDim.y - 2*gridDim.x - ypos*2*(gridDim.x)] = A[xpos + ypos*2*(gridDim.x)];																										// Mirror this mag. value to the corresponding symmetrical pixel
		
	}
	
//////////////////////// Magnification - end
		
	#undef EPS
	#undef MAXM
	#undef EPSS
	#undef MR
	#undef MT
	#undef MAXIT
	#undef NRANSI
	
	""")

gpu_finite_source=SourceModule("""
		#include <pycuda-complex.hpp>
	#include <stdio.h>
	#include <math.h>
	#define	COMPLEXTYPE	pycuda::complex<double>
	
	#include "texture_fetch_functions.h"
	#include "texture_types.h"
	
	#define NRANSI																																																																									// Used by Numerical Recipies (NR)
	#include "/home/amm315/code/binary/include/nrutil.h"																																																								// 
	#define EPSS 1.0e-7																																																																								// For NR ZRoots
	#define MR 8																																																																											// v
	#define MT 10																																																																										// v
	#define MAXIT (MT*MR)																																																																				// v
	#define EPS 2.0e-6																																																																								// v
	#define MAXM 100																																																																								// v
	#define	TRUE	1																																																																									// v
	#define	FALSE	0																																																																									// ---
//																																																																																		// For NR Amoeba
	#define NDIM 2						// This value needs to be included in the #define GET_SUM " for (...j<=NDIM...)" and in p[(i-1)*NDIM+j]
	#define MP 3																																																																											// v
	#define FTOL 1.0e-5																																																																							// v
	#define NMAX 5000																																																																							// v
	#define GET_PSUM \
						for (j=1;j<=2;j++) {\
						for (sum=0.0,i=1;i<=mpts;i++) sum += p[(i-1)*2+j];\
						psum[j]=sum;}
	#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}																																																										// ---
//																																																																																		// For NR Zroots
	#define	NUM_POLYS	6																																																																					// v
	#define	MAX_THREADS 512				// Must be a value of 2^n																																																																// v
	#define	MAX_DATA_POINTS 3584																																																																				// v
	#define SUB_PIXELS 10000																																																																			// v
	#define SUB_PIX_DIM 100 			// sqrt(SUB_PIXELS)
	#define INV_SUB_PIX_DIM 0.01 		// = 1/(sqrt(SUB_PIXELS))
	#define SUB_PIXELS_DIM_X 500																																																																// ---
	
	#define	MAX_DATA_SITES	10
	
	texture<float, 2> texref;	
	
	//////////////////////////////////////////////////////////////	Kernel Calculation
	
	__global__ void kernel_calc(double dx , double rho2, double *kernel)																																																		// Routine to produce a kernel for use in generating a convoled mag. map, to investigate finite sources
	{
		__shared__ int area[SUB_PIXELS];
		int index = threadIdx.x;
		int pixel_id_x = blockIdx.x;																																																																					// Pixel x-axis id on the kernel
		int pixel_id_y = blockIdx.y;																																																																					// Pixel y-axis id on the kernel
		int mid_point = gridDim.x;																																																																					// - Only computing 1/4 of the whole kernel as it will be mirrored
		int sub_pixel_x,sub_pixel_y, k;																																																																			// Each pixel on the kernel is subdivided to produce a rough calculation as to what percentage of that kernels pixel is included within the finite source
		double distance;
		

		while (index < SUB_PIXELS)																																																																			// To allow for iterations exceeding the thread limit, in this case the thread will loop around and re-compute a new value which would otherwise be outside the limit of number of threads.
		{
			sub_pixel_y = index*INV_SUB_PIX_DIM;																																																												// Determine the sub_pixel location (x-axis)
			sub_pixel_x = index - (SUB_PIX_DIM*sub_pixel_y);																																																							// Determine the sub_pixel location (y-axis)
			
			distance = pow((pixel_id_x - mid_point + INV_SUB_PIX_DIM*sub_pixel_x)*dx,2) + pow((pixel_id_y - mid_point + INV_SUB_PIX_DIM*sub_pixel_y)*dx,2);						// Calculate the distance away the centre of the sub pixel is from the centre of the finite source.
			
			area[index] = distance <= rho2 ? 1:0;
			
			index += SUB_PIXELS_DIM_X;
		}
		
		__syncthreads();

		if (threadIdx.x < SUB_PIX_DIM)																																																																	// Count up the number of sub pixels in each pixel which are inside the finit source.
		{
			for (k=1;k<SUB_PIX_DIM;k++)
			{
				area[threadIdx.x] += area[threadIdx.x+SUB_PIX_DIM*k];
			}
		}

		__syncthreads();
		
		if (threadIdx.x == 0)
		{
			kernel[blockIdx.x + (gridDim.x *blockIdx.y)] = 0;
			for (k=0;k<SUB_PIX_DIM;k++)
			{
				kernel[blockIdx.x+(gridDim.x)*blockIdx.y] += area[k];
			}
			kernel[blockIdx.x + (gridDim.x *blockIdx.y)] = kernel[blockIdx.x + (gridDim.x *blockIdx.y)] / SUB_PIXELS;																													// Store the percent of the kernels pixel inside the finite source by summing the sub pixels inside and dividing by the total number of sub pixels. (the more sub pixels per kernel pixel, the more accurate this result).
		}
		
	}

//////////////////////// Kernel Calculation - end

//////////////////////////////////////////////////////////////	Finite Source Map
	
	__global__ void convolve(int size, float *kernel, float *convolved)																																																				// Func. to produce a convolved mag map from the original mag. map, by using the computed kernel. Used to make interpolations for finite sources.
	{
		int xpos = threadIdx.x + (blockIdx.y * blockDim.x);																																																										// Determine the u1 position on the mag. map
		int ypos = blockIdx.x;																																																																							// Determine the u2 position on the mag. map
		int mid_point = size/2;																																																																							// As with mag. map generation, only half the u2 values due to symmetry.
		int i, j;
		
		convolved[xpos + ypos*2*(gridDim.x)] = 0;																																																													// Pre define the array sizes for the convolved map to be zeros
		convolved[xpos + 2*(gridDim.x)*blockDim.x*gridDim.y - 2*gridDim.x - ypos*2*(gridDim.x)] = 0;																																				// Repeat for the mirrored values
		
		for (i=-mid_point;i<=mid_point;i++)																																																																// Scroll through the kernel in the x-axis
		{
			for (j=-mid_point;j<=mid_point;j++)																																																															// Scroll through the kernel in the y-axis
			{
				convolved[xpos + ypos*2*(gridDim.x)] += tex2D(texref,xpos+i,ypos+j) * kernel[(i+mid_point) + size*(j+mid_point)];																								// Sum the mag map pixel values around the convolved map pixel location, going through the dims of the kernel, and multiple the mag. map pixel by the corresponding kernel pixel value.
			}
		}
		
		convolved[xpos + 2*(gridDim.x)*blockDim.x*gridDim.y - 2*gridDim.x - ypos*2*(gridDim.x)] = convolved[xpos + ypos*2*(gridDim.x)];																		// Mirror the convolved pixel value, due to symmetry.

	}

//////////////////////// Finite Source Map - end
		
	#undef EPS
	#undef MAXM
	#undef EPSS
	#undef MR
	#undef MT
	#undef MAXIT
	#undef NRANSI
	
	""")

gpu_trajectories = SourceModule("""
	#include <pycuda-complex.hpp>
	#include <stdio.h>
	#include <math.h>
	#define	COMPLEXTYPE	pycuda::complex<double>
	
	#include "texture_fetch_functions.h"
	#include "texture_types.h"
	
	#define NRANSI																																																																									// Used by Numerical Recipies (NR)
	#include "/home/amm315/code/binary/include/nrutil.h"																																																								// 
	#define EPSS 1.0e-7																																																																								// For NR ZRoots
	#define MR 8																																																																											// v
	#define MT 10																																																																										// v
	#define MAXIT (MT*MR)																																																																				// v
	#define EPS 2.0e-6																																																																								// v
	#define MAXM 100																																																																								// v
	#define	TRUE	1																																																																									// v
	#define	FALSE	0																																																																									// ---
//																																																																																		// For NR Amoeba
	#define NDIM 2						// This value needs to be included in the #define GET_SUM " for (...j<=NDIM...)" and in p[(i-1)*NDIM+j]
	#define MP 3																																																																											// v
	#define FTOL 1.0e-5																																																																							// v
	#define NMAX 500																																																																							// v
	#define GET_PSUM \
						for (j=1;j<=2;j++) {\
						for (sum=0.0,i=1;i<=mpts;i++) sum += p[(i-1)*2+j];\
						psum[j]=sum;}
	#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}																																																										// ---
//																																																																																		// For NR Zroots
	#define EPSILON 1.0e-14
	#define	NUM_POLYS	6																																																																					// v
	#define	MAX_THREADS 512				// Must be a value of 2^n																																																																// v
	#define	MAX_DATA_POINTS 8192		// Upperl limit is 8192																																																																				// v
	#define SUB_PIXELS 10000																																																																			// v
	#define SUB_PIX_DIM 100 			// sqrt(SUB_PIXELS)
	#define INV_SUB_PIX_DIM 0.01 		// = 1/(sqrt(SUB_PIXELS))
	#define SUB_PIXELS_DIM_X 500																																																																// ---
	
	#define	MAX_DATA_SITES	10
	
	texture<float, 2> texref;
	
//////////////////////////////////////////////////////////////	Bicubic Interpolation
//																																																																																			// From CUDA SDK modified slightly to remove a few c++ functions that would not compile instead I have hard coded the data types into the code
	// w0, w1, w2, and w3 are the four cubic B-spline basis functions
	__host__ __device__	float w0(float a)
	{
	//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
		return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
	}

	__host__ __device__	float w1(float a)
	{
	//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
		return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
	}

	__host__ __device__	float w2(float a)
	{
	//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
		return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
	}

	__host__ __device__	float w3(float a)
	{
		return (1.0f/6.0f)*(a*a*a);
	}
	
	
	// filter 4 values using cubic splines
// line 136
//	template<class T>
	__device__ float cubicFilter(float x, float c0, float c1, float c2, float c3)
	{
		float r;
		r = c0 * w0(x);
		r += c1 * w1(x);
		r += c2 * w2(x);
		r += c3 * w3(x);
		return r;
	}
	
	
	// slow but precise bicubic lookup using 16 texture lookups
// line 150
//	template<class T, class R>  // texture data type, return type
	__device__ float tex2DBicubic(float x, float y)																																																														// Succesfull bicubic interpolating routine that reads map from texture memory.
	{
		x -= 0.5f;
		y -= 0.5f;
		float px = floor(x);
		float py = floor(y);
		float fx = x - px;
		float fy = y - py;

		return cubicFilter(fy,
							  cubicFilter(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
							  cubicFilter(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
							  cubicFilter(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
							  cubicFilter(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
							  );
	}
	
//////////////////////// Bicubic - end

//////////////////////////////////////////////////////////////	Interpolation
	
	__global__ void LC(int num_points, float dim_min, float dim_max, int data_rows, float map_limit, double b, float *u1, float *u2, float *A_interp)																													// Routine used to interpolate the mag. map values from the texture memory.
	{
		int index = threadIdx.x + (blockDim.x * blockIdx.x);
		
		float pixel_x, pixel_y;
		double u, usq;
		
		if (index < data_rows)
		{
			if ( (fabs(u1[index]) < map_limit) && (fabs(u2[index]) < map_limit) )
			{
				pixel_x = num_points * (u1[index] - dim_min) / (dim_max-dim_min);
				pixel_y = num_points * (u2[index] - dim_min) / (dim_max-dim_min);
				
				A_interp[index] = tex2DBicubic(pixel_x, pixel_y);
			}
			else
			{
				usq = (u1[index]-b)*(u1[index]-b) + u2[index]*u2[index];
				u = sqrt(usq);
				
				A_interp[index] = (float)( (usq + 2) / (u*sqrt(usq+4)) );
			}
		}
	}

//////////////////////// Interpolation - end


////////////////////////////////////////////////////////////// Chi^2 calculation																																																					// Chi**2 calculation function for MOA data, called by nuermous other routines.
	__device__ double chi2_func(double *variables, double *u0, double *phi, double *b, double *map_limit, int *num_points, float *dim_min, float *dim_max, int *TIME_STEPS, int *pow_2_n, double *a0, double *a1, double *ts, float *mags, float *sigs)
	{
		__shared__ double ynew[MAX_THREADS], a0try, a1try;
		__shared__ float interp_mag[MAX_DATA_POINTS];
		__shared__ float ma[MAX_THREADS], mb[MAX_THREADS], md[MAX_THREADS], me[MAX_THREADS], mf[MAX_THREADS];
		__shared__ int power_limit;
		__shared__ int check;
		double t0, tE, u0n, t0n, u1, u2, usq, u;
		float pixel_x, pixel_y;
		int k, index;
		int thread_max = blockDim.x;
		
		
		index = threadIdx.x;
		
		if (threadIdx.x == 0)
		{
			check = *a1==1.0 ? 1:0;
			power_limit = *pow_2_n;																																																																					// Store the global var as a shared var to prevent issues caused when mutliple threads are trying to access the same global var, also shared mem. is faster then global mem.
		}
		
		__syncthreads();
		t0 = variables[1];																																																																										// Set t0 from global input
		tE = variables[2];																																																																										// Set tE from global input
		
		u0n = (double)( *u0 - sin(*phi)*(*b) );																																																																// Calculate the new u0 for the middle of mass coordiante system
		t0n = (double)( t0 - tE*((*u0)/tan(*phi) - u0n/tan(*phi)) );																																																							// Calculate the new t0 for the middle of mass coordiante system
		
		ynew[threadIdx.x] = 0;
		
		__syncthreads();
		
		if (check == 1)
		{
			__syncthreads();
			ma[threadIdx.x] = mb[threadIdx.x] = 0;
			
			__syncthreads();
			while (index < *TIME_STEPS)																																																																			// Set while for cases where more data points then max threads allowed
			{
				u1 = (double)( ((ts[index] - t0n)/tE)*cos(*phi) - u0n*sin(*phi) );																																																	// Determine the u1 coordinates on the mag. map
				u2 = (double)( ((ts[index] - t0n)/tE)*sin(*phi) + u0n*cos(*phi) );																																																	// Determine the u2 coordiantes on the mag. map
				
				if ( (fabs(u1) < *map_limit) && (fabs(u2) < *map_limit) )																																																							// Test if (u1,u2) coordinates are inside the mag map dims
				{			
					pixel_x = (float)( (*num_points) * ((float)u1 - *dim_min) / (*dim_max-*dim_min) );																																									// 	-	If yes then interpolate mag. from map in texture memory.
					pixel_y = (float)( (*num_points) * ((float)u2 - *dim_min) / (*dim_max-*dim_min) );
					
					interp_mag[index] = tex2DBicubic(pixel_x, pixel_y);
				}
				else																																																																															//	-	Else calculate a magnification based on the single lens model approx
				{
					usq = (u1-*b)*(u1-*b) + u2*u2;
					u = sqrt(usq);
					interp_mag[index] = (float)( (usq + 2) / (u*sqrt(usq+4)) );
				}
				
				ma[threadIdx.x] += (mags[index] * (interp_mag[index]-1)) / (sigs[index]*sigs[index]);
				mb[threadIdx.x] += ((interp_mag[index]-1)*(interp_mag[index]-1)) / (sigs[index]*sigs[index]);
				
				index += thread_max;																																																																				// For looping over the data pohints if they exceed the max threads allowed
			}
			while (index < power_limit)																																																																					// As the dims of the data array are 2^n, ensure any remaining array values which would be outside the data array range are assigned to zero
			{																																																																																		// this is important for the parrallel summing techniques that require the arrays to be 2^n in size.
				interp_mag[index] = 0;
				index += thread_max;
			}
//																																			// then repeat this procedure with half again, and loop until all values are summed down to a single element.
//																																			// This method has been taken from the CUDA SDK.
			k = thread_max / 2;																																																																									// Utilising the array size of 2^n, it is possible to do a parallel sumation, by computing half the array dimms and summing that value with its index + half the arrays size.
			__syncthreads();
			while (k!=0)																																																																													// Calculate the summed Fo contributing values
			{
				if (threadIdx.x < k)
				{
					ma[threadIdx.x] += ma[threadIdx.x + k];
					mb[threadIdx.x] += mb[threadIdx.x + k];
				}
				__syncthreads();
				k /= 2;
			}
			
			__syncthreads();
			index = threadIdx.x;
			while (index < *TIME_STEPS)																																																																			// Calculate the squared difference between the model value and the real data
			{
				ynew[threadIdx.x] += pow( ( mags[index] - (interp_mag[index]-1) * ma[0]/mb[0] ) ,2) / (sigs[index]*sigs[index]);
				index += thread_max;
			}
			
			k = thread_max / 2;
			__syncthreads();
			while (k!=0)																																																																													// Parrallel sum the squared difference values
			{
				if (threadIdx.x < k)
				{
					ynew[threadIdx.x] += ynew[threadIdx.x + k];
				}
				__syncthreads();
				k /= 2;
			}
			
			__syncthreads();
			if (threadIdx.x == 0)
			{
	//			*chi2	= ynew[0];
				*a0		= (double)(ma[0]/mb[0]);																																																												// Store the source flux parameter at its associated pointer so it can be used outside the function
			}
			
			__syncthreads();
			return ynew[0];
			
			
			
		}
		else
		{



			__syncthreads();
			ma[threadIdx.x] = mb[threadIdx.x] = md[threadIdx.x] = me[threadIdx.x] = mf[threadIdx.x] = 0;
			
			__syncthreads();
			while (index < *TIME_STEPS)																																																																			// Set while for cases where more data points then max threads allowed
			{
				u1 = (double)( ((ts[index] - t0n)/tE)*cos(*phi) - u0n*sin(*phi) );																																																	// Determine the u1 coordinates on the mag. map
				u2 = (double)( ((ts[index] - t0n)/tE)*sin(*phi) + u0n*cos(*phi) );																																																	// Determine the u2 coordiantes on the mag. map
				
				if ( (fabs(u1) < *map_limit) && (fabs(u2) < *map_limit) )																																																							// Test if (u1,u2) coordinates are inside the mag map dims
				{			
					pixel_x = (float)( (*num_points) * ((float)u1 - *dim_min) / (*dim_max-*dim_min) );																																									// 	-	If yes then interpolate mag. from map in texture memory.
					pixel_y = (float)( (*num_points) * ((float)u2 - *dim_min) / (*dim_max-*dim_min) );
					
					interp_mag[index] = tex2DBicubic(pixel_x, pixel_y);
				}
				else																																																																															//	-	Else calculate a magnification based on the single lens model approx
				{
					usq = (u1-*b)*(u1-*b) + u2*u2;
					u = sqrt(usq);
					interp_mag[index] = (float)( (usq + 2) / (u*sqrt(usq+4)) );
				}
				
				ma[threadIdx.x] += (interp_mag[index]*interp_mag[index]) / (sigs[index]*sigs[index]);
				mb[threadIdx.x] += -(interp_mag[index]) / (sigs[index]*sigs[index]);
				md[threadIdx.x] += 1 / (sigs[index]*sigs[index]);
				me[threadIdx.x] += (mags[index]*interp_mag[index]) / (sigs[index]*sigs[index]);
				mf[threadIdx.x] += (mags[index]) / (sigs[index]*sigs[index]);
				
				index += thread_max;																																																																				// For looping over the data pohints if they exceed the max threads allowed
			}
			while (index < power_limit)																																																																					// As the dims of the data array are 2^n, ensure any remaining array values which would be outside the data array range are assigned to zero
			{																																																																																		// this is important for the parrallel summing techniques that require the arrays to be 2^n in size.
				interp_mag[index] = 0;
				index += thread_max;
			}
			
			k = thread_max / 2;																																																																									// Utilising the array size of 2^n, it is possible to do a parallel sumation, by computing half the array dimms and summing that value with its index + half the arrays size.
			__syncthreads();
			while (k!=0)																																																																													// Calculate the summed Fo contributing values
			{
				if (threadIdx.x < k)
				{
					ma[threadIdx.x] += ma[threadIdx.x + k];
					mb[threadIdx.x] += mb[threadIdx.x + k];
					md[threadIdx.x] += md[threadIdx.x + k];
					me[threadIdx.x] += me[threadIdx.x + k];
					mf[threadIdx.x] += mf[threadIdx.x + k];
				}
				__syncthreads();
				k /= 2;
			}
			
			__syncthreads();
			if (threadIdx.x == 0)
			{
				a0try = (md[0]*me[0] - mb[0]*mf[0]) / (ma[0]*md[0] - mb[0]*mb[0]);
				a1try = (-mb[0]*me[0] + ma[0]*mf[0]) / (ma[0]*md[0] - mb[0]*mb[0]);
			}
			__syncthreads();
			
			index = threadIdx.x;
			while (index < *TIME_STEPS)																																																																			// Calculate the squared difference between the model value and the real data
			{
				ynew[threadIdx.x] += pow( (a0try*interp_mag[index] - a1try - mags[index]) ,2) / (sigs[index]*sigs[index]);
				index += thread_max;
			}
			
			k = thread_max / 2;
			__syncthreads();
			while (k!=0)																																																																													// Parrallel sum the squared difference values
			{
				if (threadIdx.x < k)
				{
					ynew[threadIdx.x] += ynew[threadIdx.x + k];
				}
				__syncthreads();
				k /= 2;
			}
			
			if (threadIdx.x == 0)
			{
	//			*chi2	= ynew[0];
				*a0		= (double)a0try;
				*a1		= (double)a1try;
			}
			
			__syncthreads();
			return ynew[0];
			
		}
		
	}

//////////////////////// Chi^2 calculation - end


//////////////////////////////////////////////////////////////	Chi^2 type controller
	__device__ double chi2_controller(int single, double *variables, double *u0, double *phi, double *b, double *map_limit, int *num_points, float *dim_min, float *dim_max, int *TIME_STEPS, int *pow_2_n, double *a0, double *a1, double *ts, float *mags, float *sigs, int *data_sets, int *data_type)
	{
		__shared__ double chi2;
		__shared__ int start;
		double  ytry, a0try, a1try;
		int i;
		
		start = 0;
		chi2 = 0.0;
		
		
		for (i=0; i<*data_sets; i++)
		{
			ytry = 0.0;
			a0try = 0.0;
			a1try = 0.0;
			
			switch(data_type[i])
			{
				case 0:									// Particularily the case for OGLE and MOA
					if (threadIdx.x == 0)
					{
						a1try = 1.0;
					}
					__syncthreads();
					ytry = chi2_func(variables, u0, phi, b, map_limit, num_points, dim_min, dim_max,&(TIME_STEPS[i]), &(pow_2_n[i]), &a0try, &a1try, &(ts[start]), &(mags[start]), &(sigs[start]));
					
					if (threadIdx.x == 0)
					{
						chi2 += ytry;
						if (single == 1)
						{
							a0[i] = a0try;
							a1[i] = 1.0;
						}
					}
					break;
					
				case 1:
					ytry = 0.0;
					a0try = 0.0;
					a1try = 0.0;
					__syncthreads();
					ytry = chi2_func(variables, u0, phi, b, map_limit, num_points, dim_min, dim_max,&(TIME_STEPS[i]), &(pow_2_n[i]), &a0try, &a1try, &(ts[start]), &(mags[start]), &(sigs[start]));
					
					if (threadIdx.x == 0)
					{
						chi2 += ytry;
						if (single == 1)
						{
							a0[i] = a0try;
							a1[i] = a1try;
						}
					}
					break;
			}
			
			if (threadIdx.x == 0)
			{
				start += TIME_STEPS[i];
			}
		}
		return chi2;
	}
//////////////////////// Chi^2 type controller - end


//////////////////////////////////////////////////////////////	For simplex downhill
//																																																																																				// NR routine 'amotry', read "Numerical Recipes in C 2nd Ed.", and the NR exercise book [C] 2nd Ed. 
//																																																																																				// Modified a lot to hard code in a lot of unessary variables, and remove NR specific functions which defined arrays.
	__device__ double amotry(double p[], double y[], double psum[], int ihi, double fac,
		double *u0, double *phi, double *b, double *map_limit, int *num_points, float *dim_min, float *dim_max, int *TIME_STEPS, int *pow_2_n, double *ts, float *mags, float *sigs, int *data_sets, int *data_type)
	{
		__shared__ double ptry[NDIM+1], fac1,fac2,ytry;
		double temp;
		int j;
		
	//	ptry=vector(1,NDIM);																																																																								// Replaced NR array initialising method with standard C practice, ensure the array size is +1 what it needs to be, due to NR conversion of code from Fortran where array indexing starts at 1 instead of 0
		if (threadIdx.x == 0)																																																																									// Made access/writing to global variables be performed on single threads only, then sync threads. Prevents all threads trying to acces global vars simultaneously which is slower and can cause issues.
		{
			fac1=(1.0-fac)/NDIM;
			fac2=fac1-fac;
			for (j=1;j<=NDIM;j++)
			{
				ptry[j]=psum[j]*fac1-p[(ihi-1)*NDIM+j]*fac2;																																																										// Replaced NR's 2D array technique with a 1D array of dims x*y, where you index by x + y*size_x
			}																																																																																	// Write ptry as a shared variable to prevent multi thread access
		}
		__syncthreads();
		
		ytry=chi2_controller(0, ptry, u0, phi, b, map_limit, num_points, dim_min, dim_max, TIME_STEPS, pow_2_n, &temp, &temp, ts, mags, sigs, data_sets, data_type);											// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
		__syncthreads();
		
		if (threadIdx.x == 0)																																																																									// Single thread performing to prevent multithreaded simultaneous access (slow/erroneous)
		{
			if (ytry < y[ihi])
			{
				y[ihi]=ytry;
				for (j=1;j<=NDIM;j++)
				{
					psum[j] += ptry[j]-p[(ihi-1)*NDIM+j];																																																													// Replaced NR's 2D array technique with a 1D array of dims x*y, where you index by x + y*size_x
					p[(ihi-1)*NDIM+j]=ptry[j];																																																																		// Replaced NR's 2D array technique with a 1D array of dims x*y, where you index by x + y*size_x
				}
			}
		}
		__syncthreads();
		// free_vector(ptry,1,NDIM);																																																																					// Does not work on a device in CUDA, free array should not be required, and should not affect the rest of the routine.
		return ytry;																																																																													// Return the chi^2 value of the attempted parameters
	}
	
////////////
//																																																																																				// NR routine 'amoeba', read "Numerical Recipes in C 2nd Ed.", and the NR exercise book [C] 2nd Ed. 
	__device__ void amoeba(double p[], double y[], double ftol,
		int *nfunk,
		double *u0, double *phi, double *b, double *map_limit, int *num_points, float *dim_min, float *dim_max, int *TIME_STEPS, int *pow_2_n, double *ts, float *mags, float *sigs, int *data_sets, int *data_type)
	{
		__shared__ int break_out;
		__shared__ double psum[NDIM+1];
		int i,ilo, ihi, inhi,j,mpts=NDIM+1;
		double rtol,sum,swap,ysave,ytry, temp;
		
		if (threadIdx.x == 0)																																																																									// Single thread performing to prevent multithreaded simultaneous access (slow/erroneous)
		{
			break_out = 0;
			*nfunk=0;																																																																												// Single thread to define a global var
			GET_PSUM
		}
		__syncthreads();
		
		for (;;)
		{
			ilo=1;
			ihi = y[1]>y[2] ? (inhi=2,1) : (inhi=1,2);
			for (i=1;i<=mpts;i++)
			{
				if (y[i] <= y[ilo]) ilo=i;
				if (y[i] > y[ihi])
				{
					inhi=ihi;
					ihi=i;
				} else if (y[i] > y[inhi] && i != ihi) inhi=i;
			}
			
			if (threadIdx.x == 0)
			{
				rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo]));																																																								// Single thread test for local minima
				if (rtol < ftol)
				{
					SWAP(y[1],y[ilo])
					for (i=1;i<=NDIM;i++) SWAP(p[i],p[(ilo-1)*NDIM+i])																																																						// Replaced NR's 2D array technique with a 1D array of dims x*y, where you index by x + y*size_x
					break_out = 1;																																																																									// Set shared var so all threads can read it, telling them to return
				}
				if (*nfunk >= NMAX)																																																																						// Single thread test to see if iterations exceeds max allowed
				{
					break_out = 1;																																																																									// Set shared var so all threads can read it, telling them to return
				}
			}
			__syncthreads();
			if (break_out == 1)																																																																								// If break_out flag is yes, then make all threads exit for loop
			{
				return;
			}
			
			if (threadIdx.x == 0)																																																																								// Only a single thread to update the global variable.
			{
				*nfunk += NDIM;
			}
			__syncthreads();
			
			ytry=amotry(p,y,psum,ihi,-1.0, u0, phi, b, map_limit, num_points, dim_min, dim_max, TIME_STEPS, pow_2_n, ts, mags, sigs, data_sets, data_type);						// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
			
			__syncthreads();
			
			if (ytry <= y[ilo])
				ytry=amotry(p,y,psum,ihi,2.0, u0, phi, b, map_limit, num_points, dim_min, dim_max, TIME_STEPS, pow_2_n, ts, mags, sigs, data_sets, data_type);						// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
			else if (ytry >= y[inhi])
			{
				ysave=y[ihi];
				ytry=amotry(p,y,psum,ihi,0.5, u0, phi, b, map_limit, num_points, dim_min, dim_max, TIME_STEPS, pow_2_n, ts, mags, sigs, data_sets, data_type);						// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
				
				__syncthreads();
				
				if (ytry >= ysave)
				{
					for (i=1;i<=mpts;i++)
					{
						if (i != ilo)
						{
							if (threadIdx.x == 0)
							{
								for (j=1;j<=NDIM;j++)
								{
									p[(i-1)*NDIM+j]=psum[j]=0.5*(p[(i-1)*NDIM+j]+p[(ilo-1)*NDIM+j]);																																											// Replaced NR's 2D array technique with a 1D array of dims x*y, where you index by x + y*size_x
								}
							}
							__syncthreads();
							ytry=chi2_controller(0, psum, u0, phi, b, map_limit, num_points, dim_min, dim_max, TIME_STEPS, pow_2_n, &temp, &temp, ts, mags, sigs, data_sets, data_type);							// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
							__syncthreads();
							if (threadIdx.x == 0)																																																																					// Only a single thread to write the varaibles to global memory (speed/no errors)
							{
								y[i] = ytry;
							}
							__syncthreads();
						}
					}
					if (threadIdx.x==0)																																																																							// Only a single thread operates on the global function.
					{
						*nfunk += NDIM;
						GET_PSUM
					}
					__syncthreads();
				}
			}
			else
			{
				if (threadIdx.x == 0) --(*nfunk);
			}
			__syncthreads();
		}
	}

//////////////////////// For simplex downhill - end


//////////////////////////////////////////////////////////////	Parameter serach for u0 & phi																																																	// Function to set initial block id values for u0 and phi, then call the NR routines to run a simplex downhill procedure to solve for t0 and tE
	
	__global__ void nested_search(double Al, double map_limit, double b, double lu0, double du0, double lphi, double dphi, double est_t0, double est_tE, int num_points, float dim_min, float dim_max, int *TIME_STEPS, double *ts, float *mags, float *sigs, int data_sets, int *data_type, double *chi2_arr, double *t0_arr, double *tE_arr)
	{
		__shared__ double p[(NDIM*MP)+1], x[NDIM+1], y[MP+1];
		__shared__ int Al_test;
		double u0n, ta, tb, tc, td, s1, s2, u1, u2, pixel_x, pixel_y, u0, phi, ytry;
		int index, i, n, pow_2_n[MAX_DATA_SITES], nfunc;
		
		if (threadIdx.x == 0)
		{
			Al_test	= 0;
		}
		
		u0 = (double)pow((double)10.0, (lu0 + (blockIdx.y) * du0) );
		phi = (double)( lphi + blockIdx.x * dphi );
		
		
		// TEST: Will trajectory ever reach large enough magnification values?

		u0n = (double)( u0 - sin(phi)*(b) );	
		if ( (fabs(cos(phi)) < EPSILON) || ((fabs(sin(phi)) < EPSILON)) )
		{
			s2 = map_limit;
			s1 = -map_limit;
		}
		else
		{
			ta	= (map_limit + u0n*sin(phi) ) / ( cos(phi) );
			tb	= (map_limit - u0n*cos(phi) ) / ( sin(phi) );
			tc	= (-map_limit + u0n*sin(phi) ) / ( cos(phi) );
			td	= (-map_limit - u0n*cos(phi) ) / ( sin(phi) );
			
			s1	=	fabs(ta) > fabs(tb) ? tb : ta;
			s2	=	fabs(ta) <=fabs(tb) ? tb : ta;
			if (fabs(tc)<=fabs(s1))
			{
				s2 = s1;
				s1 = tc;
			}
			else if (fabs(tc)<=fabs(s2))
			{
				s2 = tc;
			}
			
			if (fabs(td)<=fabs(s1))
			{
				s2 = s1;
				s1 = td;
			}
			else if (fabs(td)<=fabs(s2))
			{
				s2 = td;
			}
			
			if (s1 > 0)
			{
				s1=ta;
				s1=s2;
				s2=ta;
			}
		}
		
		
		index = threadIdx.x;
		__syncthreads();
		
		while (index < 2*num_points)
		{
			ta = s1 + index*(s2 - s1)/(2*num_points);
			
			u1 = (double)( ta*cos(phi) - u0n*sin(phi) );																																																				// Determine the u1 coordinates on the mag. map
			u2 = (double)( ta*sin(phi) + u0n*cos(phi) );																																																			// Determine the u2 coordiantes on the mag. map
			
			pixel_x = (float)( (num_points) * ((float)u1 - dim_min) / (dim_max-dim_min) );																																									// 	-	If yes then interpolate mag. from map in texture memory.
			pixel_y = (float)( (num_points) * ((float)u2 - dim_min) / (dim_max-dim_min) );
			
			if (tex2DBicubic(pixel_x, pixel_y) > Al)
			{
				Al_test += 1;
			}
			index += blockDim.x;
		}
		__syncthreads();
		
		
		if (Al_test == 0)
		{
			if (threadIdx.x == 0)
			{
				chi2_arr[blockIdx.x + gridDim.x*blockIdx.y]	= 999999999999;
			}
			return;
		}
		
		
		for (i=0;i<data_sets;i++)
		{
			n = 0;
			n = (int)(log2f(TIME_STEPS[i]-1) + 1);																																																															// Define the smalled value 2^n integer which is >= to the amount of data points
			pow_2_n[i] = (int)pow(2.0,n);
			if (pow_2_n[i] > MAX_DATA_POINTS)
			{
				pow_2_n[i] = MAX_DATA_POINTS;
			}
		}																																																																																			// The value of 2^n
		
		
		if (threadIdx.x == 0)																																																																										// Single thread to write to shared memory
		{
			x[1] = p[(0*NDIM)+1] = est_t0 - 0.5;																																																													// Storing a vertice of the simplex
			x[2] = p[(0*NDIM)+1+1] = est_tE - 0.5;																																																												// Storing a vertice of the simplex
		}
		__syncthreads();
		ytry = chi2_controller(0, x, &u0, &phi, &b, &map_limit, &num_points, &dim_min, &dim_max, TIME_STEPS, pow_2_n, &u0n, &u0n, ts, mags, sigs, &data_sets, data_type);				// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
		
		if (threadIdx.x == 0)
		{
			y[1] = ytry;																																																																															// Single thread write to shared memory
			x[1] = p[(1*NDIM)+1] = est_t0;																																																																			// Storing a vertice of the simplex
			x[2] = p[(1*NDIM)+1+1] = est_tE;																																																																		// Storing a vertice of the simplex
		}
		__syncthreads();
		ytry = chi2_controller(0, x, &u0, &phi, &b, &map_limit, &num_points, &dim_min, &dim_max, TIME_STEPS, pow_2_n, &u0n, &u0n, ts, mags, sigs, &data_sets, data_type);				// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
		
		if (threadIdx.x == 0)
		{
			y[2] = ytry;																																																																													// Single thread write to shared memory
			x[1] = p[(2*NDIM)+1] = est_t0 - 0.5;																																																													// Storing a vertice of the simplex
			x[2] = p[(2*NDIM)+1+1] = est_tE + 0.5;																																																												// Storing a vertice of the simplex
		}
		__syncthreads();
		ytry = chi2_controller(0, x, &u0, &phi, &b, &map_limit, &num_points, &dim_min, &dim_max, TIME_STEPS, pow_2_n, &u0n, &u0n, ts, mags, sigs, &data_sets, data_type);				// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
		
		if (threadIdx.x == 0)
		{
			y[3] = ytry;																																																																													// Single thread write to shared memory
		}
		__syncthreads();
		
		amoeba(p,y,FTOL,&nfunc, &u0, &phi, &b, &map_limit, &num_points, &dim_min, &dim_max, TIME_STEPS, pow_2_n, ts, mags, sigs, &data_sets, data_type);				// Call of NR routine to perform the simplex downhill
		
		__syncthreads();
		
		if(threadIdx.x == 0)																																																																										// Single thread write to global variables to be read outside of the function
		{
			chi2_arr[blockIdx.x + gridDim.x*blockIdx.y]	= y[1];
			t0_arr[blockIdx.x + gridDim.x*blockIdx.y]	= p[1];
			tE_arr[blockIdx.x + gridDim.x*blockIdx.y]	= p[2];
		}
		__syncthreads();
	}

//////////////////////// Parameter serach for for u0 & phi - end


//////////////////////////////////////////////////////////////	Single chi^2 calculation

	__global__ void single_chi2_calc(double t0, double tE, double u0, double phi, double b, double map_limit, int num_points, float dim_min, float dim_max, int *TIME_STEPS, double *ts, float *mags, float *sigs, int data_sets, int *data_type, double *a0, double *a1, double *chi2)
	{
		__shared__ double x[3];
		double ytry, a0try[MAX_DATA_SITES], a1try[MAX_DATA_SITES];
		int i, n, pow_2_n[MAX_DATA_SITES];
		
		for (i=0;i<data_sets;i++)
		{
			n = 0;
			n = (int)(log2f(TIME_STEPS[i]-1) + 1);																																																																		// Define the smalled value 2^n integer which is >= to the amount of data points
			pow_2_n[i] = (int)pow(2.0,n);
			if (pow_2_n[i] > MAX_DATA_POINTS)
			{
				pow_2_n[i] = MAX_DATA_POINTS;
			}
		}
		
		if (threadIdx.x == 0)
		{																																																																																						// Single thread write to shared memory
			x[1] = t0;																																																																																	// Storing a vertice of the simplex
			x[2] = tE;																																																																																	// Storing a vertice of the simplex
		}
		__syncthreads();
		ytry = chi2_controller(1 ,x, &u0, &phi, &b, &map_limit, &num_points, &dim_min, &dim_max, TIME_STEPS, pow_2_n, a0try, a1try, ts, mags, sigs, &data_sets, data_type);							// All threads call the chi^2 function so that each thread can calculate its associated part of the chi^2 value for when it is summed together
		
		__syncthreads();
		if (threadIdx.x == 0)
		{
			chi2[0] = ytry;
//			a0[0] = a0try;
//			a1[0] = a1try;
		}
	}
	
//////////////////////// Single chi^2 calculation - end
	
	#undef EPS
	#undef MAXM
	#undef EPSS
	#undef MR
	#undef MT
	#undef MAXIT
	#undef NRANSI
	
	""")


def read_data(file_id, DATA_LIMIT):																																																										# Routine to read in real data
	partial_ts								=	np.zeros([DATA_LIMIT], np.float64)																																													# Set up the time, mag, error arrays to the correct size
	partial_sigs							=	np.zeros([DATA_LIMIT], np.float32)
	partial_mags						=	np.zeros([DATA_LIMIT], np.float32)
	
	if file_id[25:28] == 'MOA':
		source = 'MOA'
		TIME_STEPS = int(0)
		for line in open(file_id,'r'):																																																															# Open the file & count the number of lines in it
			TIME_STEPS += 1

		start_point = 9
		if TIME_STEPS > DATA_LIMIT:																																																											# If more lines then data limit, store which line is at (end - limit)
			start_point = TIME_STEPS - DATA_LIMIT

		TIME_STEPS = int(0)
		Data_row_count = int(0)
		obs_data  = csv.reader(open(file_id,'r'), delimiter=' ', skipinitialspace=True)																																									# Open the data to be read as a data table with ' ', seperated values
		for line in obs_data:
			if TIME_STEPS + 1 > start_point:																																																										# If line number is bigger then when to start counting
				partial_ts[Data_row_count]		=	np.float64(line[0])-np.float64(2450000)																																								# Store the JD - 2450000
				partial_mags[Data_row_count]	=	np.float32(line[1])																																																		# Store the magnification
				partial_sigs[Data_row_count]	=	np.float32(line[2])																																																		# Store the error
				Data_row_count += 1
				if Data_row_count == DATA_LIMIT:																																																								# Ensure the loop does not store more data then the maximum limit
					break
			TIME_STEPS += 1																																																																	# Count the line numbers
	
	elif file_id[25:28] == 'OGL':
		source='OGLE'
		TIME_STEPS = int(0)
		for line in open(file_id,'r'):																																																															# Open the file & count the number of lines in it
			TIME_STEPS += 1

		m0 = np.float64(0)
		start_point = 0
		
		count = int(0)
		obs_data  = csv.reader(open(file_id,'r'), delimiter=' ', skipinitialspace=True) 																																								# Open the data to be read as a data table with ' ', seperated values
		for line in obs_data:
			m0 += np.float64(line[1])
			count += 1
			if count > 0.1*TIME_STEPS:
				m0 /= count
				break
		
		if TIME_STEPS > DATA_LIMIT:																																																											# If more lines then data limit, store which line is at (end - limit)
			start_point = TIME_STEPS - DATA_LIMIT
		
		TIME_STEPS = int(0)
		Data_row_count = int(0)
		obs_data  = csv.reader(open(file_id,'r'), delimiter=' ', skipinitialspace=True) 
		for line in obs_data:
			if TIME_STEPS + 1 > start_point:																																																										# If line number is bigger then when to start counting
				partial_ts[Data_row_count]		=	np.float64(line[0])-np.float64(2450000)																																								# Store the JD - 2450000
				partial_mags[Data_row_count]	=	10**(-0.4*np.float32(line[1])) - 10**(-0.4*m0)						#converting the magnitude measuremetns of OGLE data into delta flux																																												# Store the magnification
				partial_sigs[Data_row_count]	=	-0.4*np.log(10)*10**(-0.4*np.float32(line[1]))*np.float32(line[2]) #( (np.float32(line[2])*np.float32(line[2])) + (0.01*0.01) )**0.5							# converting the assocaited sigma values of the measured magnitude into sigma of it's delta flux																																											# Store the error
				Data_row_count += 1
				if Data_row_count == DATA_LIMIT:																																																								# Ensure the loop does not store more data then the maximum limit
					break
			TIME_STEPS += 1																																																																	# Count the line numbers
			
	elif file_id[25:28] == 'CTI':
		source='CTIO-I'
		TIME_STEPS = int(0)
		for line in open(file_id,'r'):																																																															# Open the file & count the number of lines in it
			TIME_STEPS += 1

		start_point = 6
		if TIME_STEPS > DATA_LIMIT:																																																											# If more lines then data limit, store which line is at (end - limit)
			start_point = TIME_STEPS - DATA_LIMIT

		TIME_STEPS = int(0)
		Data_row_count = int(0)
		obs_data  = csv.reader(open(file_id,'r'), delimiter=' ', skipinitialspace=True)																																									# Open the data to be read as a data table with ' ', seperated values
		for line in obs_data:
			if TIME_STEPS + 1 > start_point:																																																										# If line number is bigger then when to start counting
				partial_ts[Data_row_count]		=	np.float64(line[0])																																							# Store the JD - 2450000
				partial_mags[Data_row_count]	=	10.0**( -0.4*np.float32(line[1]) )			# Stored as flux, so it can be easily solved on the GPU																																															# Store the magnification
				partial_sigs[Data_row_count]	=	-0.4*np.log(10)*10**(-0.4*np.float32(line[1]))*np.float32(line[2])																																																	# Store the error
				Data_row_count += 1
				if Data_row_count == DATA_LIMIT:																																																								# Ensure the loop does not store more data then the maximum limit
					break
			TIME_STEPS += 1	
	
	return Data_row_count, partial_ts, partial_mags, partial_sigs, source


ts				=	np.zeros([MAX_DATA_SITES*DATA_LIMIT], np.float64)
sigs			=	np.zeros([MAX_DATA_SITES*DATA_LIMIT], np.float32)
mags			=	np.zeros([MAX_DATA_SITES*DATA_LIMIT], np.float32)
DATA_ROWS		=	np.zeros([MAX_DATA_SITES],np.int32)
data_type		=	np.zeros([MAX_DATA_SITES],np.int32)
source			=	['source']*MAX_DATA_SITES

data_sets = 0

listing = os.listdir(file_id)
for infile in listing:
	if infile == 'associated' or infile == 'output' or infile == 'WIP':
		nothing = 0
	else:
		if data_sets == 0:
			if infile[0:3] == 'MOA':
				DATA_ROWS[data_sets],ts[0:DATA_LIMIT],mags[0:DATA_LIMIT],sigs[0:DATA_LIMIT], source[data_sets] = read_data(file_id+infile,DATA_LIMIT)																														# Read in the raw MOA data file
				data_type[data_sets] = 0
			elif infile[0:3] == 'OGL':
				DATA_ROWS[data_sets],ts[0:DATA_LIMIT],mags[0:DATA_LIMIT],sigs[0:DATA_LIMIT], source[data_sets] = read_data(file_id+infile,DATA_LIMIT)																													# Read in the raw OGLE data file
				data_type[data_sets] = 0
			elif infile[0:3] == 'CTI':
				DATA_ROWS[data_sets],ts[0:DATA_LIMIT],mags[0:DATA_LIMIT],sigs[0:DATA_LIMIT], source[data_sets] = read_data(file_id+infile,DATA_LIMIT)																													# Read in the raw OGLE data file
				data_type[data_sets] = 1
			data_sets += 1
		else:
			if infile[0:3] == 'MOA':
				DATA_ROWS[data_sets],ts[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT],mags[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT],sigs[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT], source[data_sets] = read_data(file_id+infile,DATA_LIMIT)			# Read in the raw MOA data file
				data_type[data_sets] = 0
			elif infile[0:3] == 'OGL':
				DATA_ROWS[data_sets],ts[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT],mags[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT],sigs[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT], source[data_sets] = read_data(file_id+infile,DATA_LIMIT)			# Read in the raw OGLE data file
				data_type[data_sets] = 0
			elif infile[0:3] == 'CTI':
				DATA_ROWS[data_sets],ts[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT],mags[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT],sigs[DATA_ROWS[data_sets-1]:DATA_ROWS[data_sets-1]+DATA_LIMIT], source[data_sets] = read_data(file_id+infile,DATA_LIMIT)			# Read in the raw OGLE data file
				data_type[data_sets] = 1
			data_sets += 1

#--- For plotting the Raw Data																																																														# Plot the raw data
plt.clf()
plt.figure(figsize=(20.48,20.48))
if data_sets > 1:
	plt.errorbar(ts[0:DATA_ROWS[0]],mags[0:DATA_ROWS[0]],sigs[0:DATA_ROWS[0]],fmt='-', ls='None', ms=2)
	for i in xrange(1, data_sets):
		plt.errorbar(ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],sigs[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],fmt='-', ls='None', ms=2)
else:
	plt.errorbar(ts[0:DATA_ROWS[0]],mags[0:DATA_ROWS[0]],sigs[0:DATA_ROWS[0]],fmt='o',ms=2,c='k')

title = 'Raw data: '+str(file_id[7:24])
xlabel = 'JD-2450000'
ylabel = 'Delta Flux'
plt.title(title,fontsize='small')
plt.xlabel(xlabel,fontsize='small')
plt.ylabel(ylabel,fontsize='small')
plt.savefig(str(file_id)+'output/raw_data.png')


localtime = time.asctime( time.localtime(time.time()) )																																																			# Get the current local time
print ''
print 'Time started:	'+str(localtime)																																																											# Print current lcoal time (time the code started)
print 'Processing file:  '+str(file_id[7:24])																																																									# Print the file ID of what is being processed
print ''
print 'Code initialised...'
print ''
print ''
print '\t --- INITIAL PARAMETER SPACE SEARCH --- \n'																																																					# Show how large the parameter space that will be searched is
print '	Data sources		:	'+str(source[0:data_sets])
print '	Number of data points	:	'+str(DATA_ROWS[0:data_sets])
print '	Searching over the following parameter space:'
print '	'+str(len(np.arange(ld,ud+dd,dd)))+'	steps of d over the range	:	'+str(10**ld)+' - '+str(10**ud)
print '	'+str(len(np.arange(lq,uq+dq,dq)))+'	steps of q over the range	:	'+str(10**lq)+' - '+str(10**uq)
print '	'+str(int((urho-lrho)/drho +1))+'	steps of rho over the range	:	'+str(lrho)+' - '+str(urho)
print '	'+str( int( len(np.arange(lu0,uu0+du0,du0)) ) )+'	steps of u0 over the range	:	'+str(10**lu0)+' - '+str(10**uu0)
print '	'+str( int( len(np.arange(lphi,uphi+dphi,dphi)) ) )+'	steps of phi over the range	:	'+str(lphi)+' - '+str(uphi)
print '	t0 is found by simplex downhill methods'
print '	tE is found by simplex downhill methods'
print '	a0 is found by linear regression methods'
print '	a1 is found by linear regression methods'
print ''
print '	Total parameter combinations to search:	'+str(int(len(np.arange(ld,ud+dd,dd)))*int(len(np.arange(lq,uq+dq,dq)))*int((urho-lrho)/drho +1)*int((uu0-lu0)/du0 +1)*int((uphi-lphi)/dphi +1))
print ''
print ''

#-- Define dimensions for d, q chi^2 map																																																											# Write to file how big the d/q parameters space is, as this will be used when plotting the chi?^2 map mid way through processing
with open(str(file_id)+"output/d_q_map.txt", "w") as f:
	f.write('%s\n%f, %f, %f, %f, %f, %f\n'% (file_id[7:24], ld,dd,ud, lq,dq,uq))

all_params = str(file_id)+"output/all_fit_parameters_"+str(file_id[7:24])+".txt"
with open(all_params, "w") as f:
	f.write('%s\n%f, %f, %f, %f, %f, %f\n'% (file_id[7:24], ld,dd,ud, lq,dq,uq))
	
best_params = str(file_id)+"output/best_fit_parameters_"+str(file_id[7:24])+".txt"
with open(best_params, "w") as f:
	f.write('No minimum parameters found yet')

	
magnification = gpu_mag_maps.get_function("magnification")																																																			# Define the CUDA function for mag map calculation

remaining_its = its = (len(np.arange(ld,ud+dd,dd))*len(np.arange(uq,lq-dq,-dq)))																																								# Count how many loops in python there will be
secs = 0
previous_time = 0
###### - START TIMER - ######																																																													# Start the timer
start.record()				#
#############################

for log_q in np.arange(uq,lq-dq,-dq):																																																											# Loop over d
	for log_d in np.arange(ld,ud+dd,dd):																																																											# Loop over q
		if secs == 0:
			print ' Estimated time remaining is yet to be determined.\r',																																															# Print out for first loop, the \r prevent a new line, and returns the cursor to the start of the line
			sys.stdout.flush()																																																																		# Force the screen to update
		else:
			current_secs			= format(int(secs - 60*int(secs/60)),'02d')
			current_mins 			= format(int((secs/60)-60*int(secs/3600)),'02d')
			current_hours			= format(int(secs/3600),'02d')
			current_est_secs		= format(int(est_time_left - 60*int(est_time_left/60)),'02d')
			current_est_mins 		= format((int((est_time_left/60)-60*int(est_time_left/3600))), '02d')
			current_est_hours		= format(int(est_time_left/3600),'02d')
			percent_f					= format(int(percent),'02d')
			progress					= current_hours+':'+current_mins+':'+current_secs
			est_remaining			= current_est_hours+':'+current_est_mins+':'+current_est_secs
			print "    Progress:  "+str(percent_f)+"%     |     Running:  "+str(progress)+"s     |     Est. remaining:  "+str(est_remaining)+"s\r",
			sys.stdout.flush()																																																																	# Print the % complete, elapsed time, and est. time remaining, use \r so that it will udpate this line each loop
		
		d = 10.0**(log_d)
		
		q = 10.0**(log_q)																																																																		# Convert log stepping of q into actualy q values
		
		#/////////////////////////////////////////////////////////	Dynamic mag. map sizing
		map_limits = (2.0/3.0)*(math.atan(2.0*d-7.2))+2.46
		### Define dimensions for use in mag. map ###
		dx,x0,dy,y0 = dims(map_limits, num_points)																																																								# Set the dimensions for the magnification map
		#//////////////////////// Dynamic mag. map sizing - end
		
		#/////////////////////////////////////////////////////////	Magnification Map generation
		e1 = 1.0/(1.0+q)
		e2 = q/(1.0+q)
		a = 0.5*d
		b = -(a*(q-1.0)) / (1.0+q)
		
		### Settings for mag. map generation ###																																																								# Block and grid the dims for mag map generation
		blockshape_A = (int(threads_per_block), 1, 1)
		gridshape_A = (int(num_points)/2, int(num_points)/int(threads_per_block))
#																																																																													# Call the mag map CUDA function
		
		magnification(np.float64(e1), np.float64(e2), np.float64(a), np.float64(dx), np.float64(x0), np.float64(dy), np.float64(y0), drv.Out(A), block=blockshape_A, grid=gridshape_A) 
		
		#//////////////////////// Magnification Map generation - end
		
		for rho in np.arange(lrho,urho+drho,drho):
			#/////////////////////////////////////////////////////////	Finite Source Convolution
			### Making the Kernel for use in the map convolution ###
			texref = gpu_finite_source.get_texref("texref")																																																								# Store the mag map (A) into the texture memory, the 'order=C' ensure the array stores as col's by row's in the correct order, this is important to ensure the mag map is not rotated by 90 deg.
			drv.matrix_to_texref(np.float32(A), texref, order="C")
			
			kernel_calc = gpu_finite_source.get_function("kernel_calc")																																														# define the GPU command to call the kernel calculating CUDA code
			
			size = math.floor(math.ceil((2.0*rho)/dx)/2.0) * 2.0 + 1.0																																																			# calculate the size that the kernel needs to be
			
			kernel = np.zeros((int(size),int(size)),np.float64)																																																				# Set up size of the kernel routine
			kernel_quarter = np.zeros((int(size/2+1),int(size/2+1)),np.float64)																																											# Set up array which is only a quarter of the kernel
			
			blockshape_Kernel	= (int(SUB_PIXELS_DIM_X), 1, 1)																																															# Set up block/grid sizes for kernel calc.
			gridshape_Kernel		= (int(size/2+1), int(size/2+1))
			
			kernel_calc(np.float64(dx), np.float64(rho**2), drv.Out(kernel_quarter), block=blockshape_Kernel, grid=gridshape_Kernel)																	# Call kernel routine
			
			kernel[0:int(size/2)+1,0:int(size/2)+1] = kernel_quarter[0:int(size/2)+1,0:int(size/2)+1]																																		# Store the kernel quater in the correct part of the whole kernel map
			kernel[int(size/2):int(size),0:int(size/2)+1] = kernel_quarter[np.arange(int(size/2),-1,-1),0:int(size/2)+1]																										# mirror this quater from bottom left to top left
			kernel[0:int(size),int(size/2):int(size)] = kernel[0:int(size),np.arange(int(size/2),-1,-1)]																																		# mirror the kernel from left to right
			
			kernel = kernel/kernel.sum()																																																													# Normalise the kernel by diviging each pixel of the kernel by the total sum of each pixel
			
			convolve = gpu_finite_source.get_function("convolve")
			
			convolved = np.zeros((num_points,num_points),np.float32)																																															# define the convoloution map array dims
			
			convolve(np.int32(size), drv.In(np.float32(kernel)), drv.Out(convolved), texrefs=[texref], block=blockshape_A, grid=gridshape_A)													# call the GPU convolve map routine
			
			#//////////////////////// Finite Source Convolution - end)
			
			#/////////////////////////////////////////////////////////	Search Parameter t0, tE space
			texref = gpu_trajectories.get_texref("texref")																																																					# store the new convolved map on the GPU texture memory
			drv.matrix_to_texref(np.float32(convolved), texref, order="C")
			
			nested_search = gpu_trajectories.get_function("nested_search")																																												# define the CPU command to call the CUDA search routine
			
			u0_steps	= int( len(np.arange(lu0,uu0+du0,du0)) )																																																								# calc the array of values u0 to be searched
			phi_steps	= int( len(np.arange(lphi,uphi+dphi,dphi)) )																																																										# calc the array of values phi to be searched
			
			if max(DATA_ROWS) <= MAX_THREADS:																																																				# prevents more threads then available to be called
				blockDim_x = 2**(np.floor(math.log(max(DATA_ROWS)-1,2))+1)																																										# if less data then threads store the smallest 2^n value >= data size
			else:
				blockDim_x = MAX_THREADS
				
			blockshape_nested_search = (int(blockDim_x), 1, 1)																																																		# set up block/grid for GPU call to use
			gridshape_nested_search = (int(phi_steps), int(u0_steps))	
			
			chi2 = np.zeros((np.int32(u0_steps),np.int32(phi_steps)),np.float64)																																											# define the CUDA output array dimensions
			t0 = np.zeros((np.int32(u0_steps),np.int32(phi_steps)),np.float64)
			tE = np.zeros((np.int32(u0_steps),np.int32(phi_steps)),np.float64)
			
			# GPU nested loop to search u0, phi, simplex t0, tE																																																			# Call the routine to search over u0 and phi, performing simplex downhill methods to find t0 and tE
			
			nested_search(np.float64(Al), np.float64(map_limits), np.float64(b), np.float64(lu0), np.float64(du0), np.float64(lphi), np.float64(dphi), np.float64(est_t0), np.float64(est_tE), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS), drv.In(ts), drv.In(mags), drv.In(sigs), np.int32(data_sets), drv.In(data_type), drv.Out(chi2), drv.Out(t0), drv.Out(tE),  texrefs=[texref], block=blockshape_nested_search, grid=gridshape_nested_search)
			
			index = (np.isnan(chi2))
			chi2[index] = 999999999999
			
			current_best_chi2_loc	= chi2.argmin()
			
			row_loc = int(current_best_chi2_loc/phi_steps)
			phi = lphi + (current_best_chi2_loc - row_loc*phi_steps)*dphi
			u0 = 10.0**(lu0 + du0*row_loc)
			
			t0_1d					= np.ravel(t0)
			tE_1d					= np.ravel(tE)
			
			if chi2.min() < current_best_chi2 + 2000:																																																							# calc the array of values phi to be searched
				
				u0_steps	=	u0_local_steps
				phi_steps	=	phi_local_steps
				
				lu0_new		= np.log10( u0 ) - u0_local_area*du0
				du0_new		= 2.0*u0_local_area* du0 / u0_local_steps
				
				lphi_new	= phi - phi_local_area*dphi
				dphi_new	= 2.0*phi_local_area*dphi / phi_local_steps
				if lphi_new <= 0:
					dphi_new = (phi + phi_local_area*dphi) / phi_local_steps
					lphi_new = dphi_new
				
				
				gridshape_nested_search = (int(phi_steps), int(u0_steps))	
				
				chi2 = np.zeros((np.int32(u0_local_steps),np.int32(phi_local_steps)),np.float64)																																											# define the CUDA output array dimensions
				t0 = np.zeros((np.int32(u0_local_steps),np.int32(phi_local_steps)),np.float64)
				tE = np.zeros((np.int32(u0_local_steps),np.int32(phi_local_steps)),np.float64)
				
				# GPU nested loop to search u0, phi, simplex t0, tE																																																			# Call the routine to search over u0 and phi, performing simplex downhill methods to find t0 and tE
				
				nested_search(np.float64(Al), np.float64(map_limits), np.float64(b), np.float64(lu0_new), np.float64(du0_new), np.float64(lphi_new), np.float64(dphi_new), np.float64(est_t0), np.float64(est_tE), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS), drv.In(ts), drv.In(mags), drv.In(sigs), np.int32(data_sets), drv.In(data_type), drv.Out(chi2), drv.Out(t0), drv.Out(tE),  texrefs=[texref], block=blockshape_nested_search, grid=gridshape_nested_search)
				
				index = (np.isnan(chi2))
				chi2[index] = 999999999999
				
				current_best_chi2_loc	= chi2.argmin()
			
				row_loc = int(current_best_chi2_loc/phi_local_steps)
				phi = lphi_new + (current_best_chi2_loc - row_loc*phi_local_steps)*dphi_new
				u0 = 10.0**(lu0_new + du0_new*row_loc)
				
				t0_1d					= np.ravel(t0)
				tE_1d					= np.ravel(tE)
			
			#print ''
			#print chi2
			#print ''
			#print chi2.min()
			#print ''
			#print ''
			#print a0
			#print ''
			#print a1
			#print ''
			
			#for i in range (0, data_sets):
				#print ''
				#print a0_1d[i*u0_steps*phi_steps+current_best_chi2_loc]
				#print a1_1d[i*u0_steps*phi_steps+current_best_chi2_loc]
			
			#sys.exit()
			
			# Relys on only a single rho value searched currently
			with open(str(file_id)+"output/d_q_map.txt", "a") as f:																																																							# Write minimised d/q chi^2 values to file
				if math.isnan(chi2.min()):
					f.write('%0.17f, %0.17f, %0.17f\n'% (d, q, 999999999999))
				else:
					f.write('%0.17f, %0.17f, %0.17f\n'% (d, q, chi2.min()))
			
			
			with open(all_params, "a") as f:
				#																																																																								# Write the full set of params to file
				f.write('%0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f'% (d, q, rho, u0, phi, t0_1d[current_best_chi2_loc], tE_1d[current_best_chi2_loc], chi2.min()))
			
			
			if chi2.min() < current_best_chi2:																																																					# test if latest run has a lower chi^2 value
				current_best_chi2		= chi2.min()
				with open(best_params, "w") as f:
					#																																																																								# Write the current best vars to file
					f.write('%0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f'% (d, q, rho, u0, phi, t0_1d[current_best_chi2_loc], tE_1d[current_best_chi2_loc], chi2.min()))
			#//////////////////////// Search Parameter t0, tE space - end
			
			
			############# - END TIMER - #############																																																			# take record of the current time reunning
			end.record()							#
			end.synchronize()						#
			secs = start.time_till(end)*1e-3		#
			#########################################
			
			remaining_its -=  1																																																																		# How many loops of d/q remain
			percent = (100*(its-remaining_its)/its)																																																									# What % is completed
			if percent == 0:																																																								# how long this individual d/q serach take
				est_time_left = (secs/(its-remaining_its)) *remaining_its																																																					# calc estimated time left by multiplying a single iterations by the number left
			else:
				est_time_left = (secs/(its-remaining_its)) *remaining_its																																																					# calc time left by avg. time taken per % * % remaining
			previous_time = secs

#/////////////////////////////////////////////////////////	Outputting Information
best_parameters  = csv.reader(open(best_params,'r'), delimiter=',', skipinitialspace=True)																																				# read in the best stored param values which are ',' seperated

for line in best_parameters:
	current_best_d				=	np.float64(line[0])																																																								# store each paramter from file as a variable
	current_best_q				=	np.float64(line[1])
	current_best_rho			=	np.float64(line[2])
	current_best_u0				=	np.float64(line[3])
	current_best_phi			=	np.float64(line[4])
	current_best_t0				=	np.float64(line[5])
	current_best_tE				=	np.float64(line[6])
	current_best_Chi2			=	np.float64(line[7])

map_limits = (2.0/3.0)*(math.atan(2*current_best_d-7.2))+2.46
with open(str(file_id)+"output/global_vars_"+str(file_id[7:24])+".txt","w") as f:
	f.write('%f, %f, %f, %f, %f, %f'% (map_limits, num_points, threads_per_block, SUB_PIXELS, SUB_PIXELS_DIM_X, DATA_LIMIT))



# INSERT SOLVE FOR a0,a1



print ''
print ''
print '	- Initial grid search complete -'
print ''
print '	Initial search found:'																																																							# Output to screen the best found parameters
print '	d	=	'+str(current_best_d)
print '	q	=	'+str(current_best_q)
print '	rho	=	'+str(current_best_rho)
print '	u0	=	'+str(current_best_u0)
print '	phi	=	'+str(current_best_phi)
print '	t0	=	'+str(current_best_t0)
print '	tE	=	'+str(current_best_tE)
print '	a0	=	'+str(current_best_a0)
print '	a1	=	'+str(current_best_a1)
print '	Chi^2	=	'+str(current_best_Chi2)
print ''	
if secs < 60:																																																																								# output to screen the time taken to run the whole parameter search
	print "Time taken	:	%0.4fs"% secs
elif secs < 3600:
	print "Time taken	:	%im  %0.4fs"% (int(secs/60), secs-60*int(secs/60))
else:
	print "Time taken	:	%ih  %im  %0.4fs"% (int(secs/3600), int((secs/60)-60*int(secs/3600)), secs-60*int(secs/60))

#########################################################################################################################################################################################################

#########################################################################################################################################################################################################

#########################################################################################################################################################################################################

#########################################################################################################################################################################################################

#########################################################################################################################################################################################################

#########################################################################################################################################################################################################

est_t0	=	current_best_t0
est_tE	=	current_best_tE

initial_search_time = secs

ld = 10.0**(np.log10(current_best_d) - d_hr_area*dd)
ud = 10.0**(np.log10(current_best_d) + d_hr_area*dd)
dd = (ud-ld)/d_hr_steps

lq = 10.0**(np.log10(current_best_q) - q_hr_area*dq)
uq = 10.0**(np.log10(current_best_q) + q_hr_area*dq)
dq = (uq-lq)/q_hr_steps

du0	= 2.0*u0_local_area * du0 / u0_local_steps
lu0 = np.log10(current_best_u0) - u0_hr_area*du0
uu0 = np.log10(current_best_u0) + u0_hr_area*du0
du0 = (uu0-lu0)/u0_hr_steps

dphi	= 2.0*phi_local_area * dphi / phi_local_steps
lphi = current_best_phi - phi_hr_area*dphi
uphi = current_best_phi + phi_hr_area*dphi
dphi = (uphi-lphi)/phi_hr_steps


localtime = time.asctime( time.localtime(time.time()) )																																																			# Get the current local time
print ''
print 'Current Time:	'+str(localtime)																																																											# Print current lcoal time (time the code started)
print ''
print ''
print '\t --- LOCALISED PARAMETER SPACE SEARCH --- \n'																																																					# Show how large the parameter space that will be searched is
print '	Searching over the following parameter space:'
print '	'+str(len(np.arange(ld,ud+dd,dd)))+'	steps of d over the range	:	'+str(ld)+' - '+str(ud)
print '	'+str(len(np.arange(uq,lq-dq,-dq)))+'	steps of q over the range	:	'+str(lq)+' - '+str(uq)
print '	1	steps of rho over the range	:	'+str(lrho)+' - '+str(urho)
print '	'+str(len(np.arange(lu0,uu0+du0,du0)))+'	steps of u0 over the range	:	'+str(10**lu0)+' - '+str(10**uu0)
print '	'+str(len(np.arange(lphi,uphi+dphi,dphi)))+'	steps of phi over the range	:	'+str(lphi)+' - '+str(uphi)
print '	t0 is found by simplex downhill methods'
print '	tE is found by simplex downhill methods'
print '	a0 is found by linear regression methods'
print '	a1 is found by linear regression methods'
print ''
print '	Total parameter combinations to search:	'+str( int(d_hr_steps)*int(q_hr_steps)*int(u0_hr_steps)*int(phi_hr_steps) )
print ''
print ''

with open(str(file_id)+"output/ftgs_d_q_map.txt", "w") as f:
	f.write('%s\n%f, %f, %f, %f, %f, %f\n'% (file_id[7:24], ld,dd,ud, lq,dq,uq))

ftgs_all_params = str(file_id)+"output/ftgs_all_fit_parameters_"+str(file_id[7:24])+".txt"
with open(ftgs_all_params, "w") as f:
	f.write('%s\n%f, %f, %f, %f, %f, %f\n'% (file_id[7:24], ld,dd,ud, lq,dq,uq))
	
ftgs_best_params = str(file_id)+"output/ftgs_best_fit_parameters_"+str(file_id[7:24])+".txt"
with open(ftgs_best_params, "w") as f:
	f.write('No minimum parameters found yet')


remaining_its = its = (len(np.arange(ld,ud+dd,dd))*len(np.arange(uq,lq-dq,-dq)))																																								# Count how many loops in python there will be
secs = 0
previous_time = 0
current_best_chi2 = 999999999999
###### - START TIMER - ######																																																													# Start the timer
start.record()				#
#############################
for q in np.arange(uq,lq-dq,-dq):																																																											# Loop over d
	for d in np.arange(ld,ud+dd,dd):																																																											# Loop over q
		if secs == 0:
			print ' Estimated time remaining is yet to be determined.\r',																																															# Print out for first loop, the \r prevent a new line, and returns the cursor to the start of the line
			sys.stdout.flush()																																																																		# Force the screen to update
		else:
			current_secs			= format(int(secs - 60*int(secs/60)),'02d')
			current_mins 			= format(int((secs/60)-60*int(secs/3600)),'02d')
			current_hours			= format(int(secs/3600),'02d')
			current_est_secs		= format(int(est_time_left - 60*int(est_time_left/60)),'02d')
			current_est_mins 		= format((int((est_time_left/60)-60*int(est_time_left/3600))), '02d')
			current_est_hours		= format(int(est_time_left/3600),'02d')
			percent_f					= format(int(percent),'02d')
			progress					= current_hours+':'+current_mins+':'+current_secs
			est_remaining			= current_est_hours+':'+current_est_mins+':'+current_est_secs
			print "    Progress:  "+str(percent_f)+"%     |     Running:  "+str(progress)+"s     |     Est. remaining:  "+str(est_remaining)+"s\r",
			sys.stdout.flush()																																																																	# Print the % complete, elapsed time, and est. time remaining, use \r so that it will udpate this line each loop																																																																	# Convert log stepping of q into actualy q values
		
		
		#/////////////////////////////////////////////////////////	Dynamic mag. map sizing
		map_limits = (2.0/3.0)*(math.atan(2.0*d-7.2))+2.46
		### Define dimensions for use in mag. map ###
		dx,x0,dy,y0 = dims(map_limits, num_points)																																																								# Set the dimensions for the magnification map
		#//////////////////////// Dynamic mag. map sizing - end
		
		#/////////////////////////////////////////////////////////	Magnification Map generation
		e1 = 1.0/(1.0+q)
		e2 = q/(1.0+q)
		a = 0.5*d
		b = -(a*(q-1.0)) / (1.0+q)
		
		### Settings for mag. map generation ###																																																								# Block and grid the dims for mag map generation
		blockshape_A = (int(threads_per_block), 1, 1)
		gridshape_A = (int(num_points)/2, int(num_points)/int(threads_per_block))
#																																																																													# Call the mag map CUDA function
		
		magnification(np.float64(e1), np.float64(e2), np.float64(a), np.float64(dx), np.float64(x0), np.float64(dy), np.float64(y0), drv.Out(A), block=blockshape_A, grid=gridshape_A) 
		
		#//////////////////////// Magnification Map generation - end
		
		for rho in np.arange(lrho,urho+drho,drho):
			#/////////////////////////////////////////////////////////	Finite Source Convolution
			### Making the Kernel for use in the map convolution ###
			texref = gpu_finite_source.get_texref("texref")																																																								# Store the mag map (A) into the texture memory, the 'order=C' ensure the array stores as col's by row's in the correct order, this is important to ensure the mag map is not rotated by 90 deg.
			drv.matrix_to_texref(np.float32(A), texref, order="C")
			
			kernel_calc = gpu_finite_source.get_function("kernel_calc")																																														# define the GPU command to call the kernel calculating CUDA code
			
			size = math.floor(math.ceil((2.0*rho)/dx)/2.0) * 2.0 + 1.0																																																			# calculate the size that the kernel needs to be
			
			kernel = np.zeros((int(size),int(size)),np.float64)																																																				# Set up size of the kernel routine
			kernel_quarter = np.zeros((int(size/2+1),int(size/2+1)),np.float64)																																											# Set up array which is only a quarter of the kernel
			
			blockshape_Kernel	= (int(SUB_PIXELS_DIM_X), 1, 1)																																															# Set up block/grid sizes for kernel calc.
			gridshape_Kernel		= (int(size/2+1), int(size/2+1))
			
			kernel_calc(np.float64(dx), np.float64(rho**2), drv.Out(kernel_quarter), block=blockshape_Kernel, grid=gridshape_Kernel)																	# Call kernel routine
			
			kernel[0:int(size/2)+1,0:int(size/2)+1] = kernel_quarter[0:int(size/2)+1,0:int(size/2)+1]																																		# Store the kernel quater in the correct part of the whole kernel map
			kernel[int(size/2):int(size),0:int(size/2)+1] = kernel_quarter[np.arange(int(size/2),-1,-1),0:int(size/2)+1]																										# mirror this quater from bottom left to top left
			kernel[0:int(size),int(size/2):int(size)] = kernel[0:int(size),np.arange(int(size/2),-1,-1)]																																		# mirror the kernel from left to right
			
			kernel = kernel/kernel.sum()																																																													# Normalise the kernel by diviging each pixel of the kernel by the total sum of each pixel
			
			convolve = gpu_finite_source.get_function("convolve")
			
			convolved = np.zeros((num_points,num_points),np.float32)																																															# define the convoloution map array dims
			
			convolve(np.int32(size), drv.In(np.float32(kernel)), drv.Out(convolved), texrefs=[texref], block=blockshape_A, grid=gridshape_A)													# call the GPU convolve map routine
			
			#//////////////////////// Finite Source Convolution - end)
			
			#/////////////////////////////////////////////////////////	Search Parameter t0, tE space
			texref = gpu_trajectories.get_texref("texref")																																																					# store the new convolved map on the GPU texture memory
			drv.matrix_to_texref(np.float32(convolved), texref, order="C")
			
			nested_search = gpu_trajectories.get_function("nested_search")																																												# define the CPU command to call the CUDA search routine
			
			u0_hr_steps	= int( len(np.arange(lu0,uu0+du0,du0)) )																																																								# calc the array of values u0 to be searched
			phi_hr_steps	= int( len(np.arange(lphi,uphi+dphi,dphi)) )																																																										# calc the array of values phi to be searched
			
			if max(DATA_ROWS) <= MAX_THREADS:																																																				# prevents more threads then available to be called
				blockDim_x = 2**(np.floor(math.log(max(DATA_ROWS)-1,2))+1)																																										# if less data then threads store the smallest 2^n value >= data size
			else:
				blockDim_x = MAX_THREADS
				
			blockshape_nested_search = (int(blockDim_x), 1, 1)																																																		# set up block/grid for GPU call to use
			gridshape_nested_search = (int(phi_hr_steps), int(u0_hr_steps))	
			
			chi2 = np.zeros((np.int32(u0_hr_steps),np.int32(phi_hr_steps)),np.float64)																																											# define the CUDA output array dimensions
			t0 = np.zeros((np.int32(u0_hr_steps),np.int32(phi_hr_steps)),np.float64)
			tE = np.zeros((np.int32(u0_hr_steps),np.int32(phi_hr_steps)),np.float64)
			
			# GPU nested loop to search u0, phi, simplex t0, tE																																																			# Call the routine to search over u0 and phi, performing simplex downhill methods to find t0 and tE
			nested_search(np.float64(Al), np.float64(map_limits), np.float64(b), np.float64(lu0), np.float64(du0), np.float64(lphi), np.float64(dphi), np.float64(est_t0), np.float64(est_tE), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS), drv.In(ts), drv.In(mags), drv.In(sigs), np.int32(data_sets), drv.In(data_type), drv.Out(chi2), drv.Out(t0), drv.Out(tE),  texrefs=[texref], block=blockshape_nested_search, grid=gridshape_nested_search)
			
			index = (np.isnan(chi2))
			chi2[index] = 999999999999
			
			current_best_chi2_loc	= chi2.argmin()
			
			row_loc = int(current_best_chi2_loc/phi_hr_steps)
			phi = lphi + (current_best_chi2_loc - row_loc*phi_hr_steps)*dphi
			u0 = 10.0**(lu0 + du0*row_loc)
			
			t0_1d					= np.ravel(t0)
			tE_1d					= np.ravel(tE)
			
			# Relys on only a single rho value searched currently
			with open(str(file_id)+"output/ftgs_d_q_map.txt", "a") as f:																																																							# Write minimised d/q chi^2 values to file
				if math.isnan(chi2.min()):
					f.write('%0.17f, %0.17f, %0.17f\n'% (d, q, 999999999999))
				else:
					f.write('%0.17f, %0.17f, %0.17f\n'% (d, q, chi2.min()))
			
			
			with open(ftgs_all_params, "a") as f:
				#																																																																								# Write the full set of params to file
				f.write('%0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f'% (d, q, rho, u0, phi, t0_1d[current_best_chi2_loc], tE_1d[current_best_chi2_loc], chi2.min()))
			
			if chi2.min() < current_best_chi2:																																																					# test if latest run has a lower chi^2 value
				current_best_chi2		= chi2.min()
				with open(ftgs_best_params, "w") as f:
					#																																																																								# Write the full set of params to file
					f.write('%0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f, %0.17f'% (d, q, rho, u0, phi, t0_1d[current_best_chi2_loc], tE_1d[current_best_chi2_loc], chi2.min()))
			#//////////////////////// Search Parameter t0, tE space - end
			
			
			############# - END TIMER - #############																																																			# take record of the current time reunning
			end.record()							#
			end.synchronize()						#
			secs = start.time_till(end)*1e-3		#
			#########################################
			
			remaining_its -=  1																																																																		# How many loops of d/q remain
			percent = (100*(its-remaining_its)/its)																																																									# What % is completed
			if percent == 0:																																																								# how long this individual d/q serach take
				est_time_left = (secs/(its-remaining_its)) *remaining_its																																																					# calc estimated time left by multiplying a single iterations by the number left
			else:
				est_time_left = (secs/(its-remaining_its)) *remaining_its																																																					# calc time left by avg. time taken per % * % remaining
			previous_time = secs


best_parameters  = csv.reader(open(ftgs_best_params,'r'), delimiter=',', skipinitialspace=True)																																				# read in the best stored param values which are ',' seperated
for line in best_parameters:
	current_best_d				=	np.float64(line[0])																																																								# store each paramter from file as a variable
	current_best_q				=	np.float64(line[1])
	current_best_rho			=	np.float64(line[2])
	current_best_u0				=	np.float64(line[3])
	current_best_phi			=	np.float64(line[4])
	current_best_t0				=	np.float64(line[5])
	current_best_tE				=	np.float64(line[6])
	current_best_Chi2			=	np.float64(line[7])

map_limits = (2.0/3.0)*(math.atan(2*current_best_d-7.2))+2.46
with open(str(file_id)+"output/global_vars_"+str(file_id[7:24])+".txt","w") as f:
	f.write('%f, %f, %f, %f, %f, %f'% (map_limits, num_points, threads_per_block, SUB_PIXELS, SUB_PIXELS_DIM_X, DATA_LIMIT))

print ''
print ''
print '	- Localised grid search complete -'
print ''
print '	Initial search found:'																																																							# Output to screen the best found parameters
print '	d	=	'+str(current_best_d)
print '	q	=	'+str(current_best_q)
print '	rho	=	'+str(current_best_rho)
print '	u0	=	'+str(current_best_u0)
print '	phi	=	'+str(current_best_phi)
print '	t0	=	'+str(current_best_t0)
print '	tE	=	'+str(current_best_tE)
print '	a0	=	'+str(current_best_a0)
print '	a1	=	'+str(current_best_a1)
print '	Chi^2	=	'+str(current_best_Chi2)
print ''	
if secs < 60:																																																																								# output to screen the time taken to run the whole parameter search
	print "Time taken	:	%0.4fs"% secs
elif secs < 3600:
	print "Time taken	:	%im  %0.4fs"% (int(secs/60), secs-60*int(secs/60))
else:
	print "Time taken	:	%ih  %im  %0.4fs"% (int(secs/3600), int((secs/60)-60*int(secs/3600)), secs-60*int(secs/60))

print ''
secs += initial_search_time
if secs < 60:																																																																								# output to screen the time taken to run the whole parameter search
	print "Total time taken	:	%0.4fs"% secs
elif secs < 3600:
	print "Total time taken	:	%im  %0.4fs"% (int(secs/60), secs-60*int(secs/60))
else:
	print "Total time taken	:	%ih  %im  %0.4fs"% (int(secs/3600), int((secs/60)-60*int(secs/3600)), secs-60*int(secs/60))



### Plotting Section ###

####----- Plot of mag map

map_limits = (2.0/3.0)*(math.atan(2*current_best_d-7.2))+2.46
### Define dimensions for use in mag. map ###
dx,x0,dy,y0 = dims(map_limits, num_points)

e1 = 1.0/(1.0+current_best_q)																																																																	# set up the required parameter values to calc the mag map associated with the best parameter values
e2 = current_best_q/(1+current_best_q)
a = 0.5*current_best_d
b = -(a*(current_best_q-1.0)) / (1.0+current_best_q)
#																																																																													# call the mag map generation code for the best fit parameter values

magnification(np.float64(e1), np.float64(e2), np.float64(a), np.float64(dx), np.float64(x0), np.float64(dy), np.float64(y0), drv.Out(A), block=blockshape_A, grid=gridshape_A)

texref = gpu_trajectories.get_texref("texref")																																																											# store the mag map array (A) into texture memory
drv.matrix_to_texref(np.float32(A), texref, order="C")

plt.clf()
plt.figure(figsize=(20.48,20.48))

plt.plot(0,0,'o', mec='y', mfc='none', mew=0.5, ms=3)																																																					# Plot on the graph the mid point between the two lens masses
plt.plot(b,0,'o', mec='r', mfc='none', mew=1, ms=4)																																																						# plot the centre of mass coordinate position
plt.plot([-a,a],[0,0],'rx',ms=8)																																																																# Plot the location of the two lens masses

#--- For plotting the best trajectory and mass locations
u0n = current_best_u0 - np.sin(current_best_phi)*b																																																					# calc the new converted to mid way between masses u0 position
t0n = current_best_t0 - current_best_tE*(current_best_u0/np.tan(current_best_phi) - u0n/np.tan(current_best_phi))																							# calc the new converted to mid way between masses t0 position

tbot = t0n-10.0*current_best_tE																																																																# pre t0 vertice location of the source trajectory
u1bot = ((tbot - t0n)/current_best_tE)*np.cos(current_best_phi) - u0n*np.sin(current_best_phi)																																	# u1 (x) coordinate of this vertice
u2bot = ((tbot - t0n)/current_best_tE)*np.sin(current_best_phi) + u0n*np.cos(current_best_phi)																																	# u2 (y) coordinate of this vertice

ttop = t0n+10.0*current_best_tE																																																															# post t0 vertice location of the source trajectory
u1top = ((ttop - t0n)/current_best_tE)*np.cos(current_best_phi) - u0n*np.sin(current_best_phi)																																	# u1 (x) coordinate of this vertice
u2top = ((ttop - t0n)/current_best_tE)*np.sin(current_best_phi) + u0n*np.cos(current_best_phi)																																	# u2 (y) coordinate of this vertice
xbot = [u1bot,u1top]
ybot = [u2bot,u2top]

plt.plot(xbot,ybot,'m-',lw=1)																																																																# Plot a line for the trajectory
plt.arrow(xbot[0],ybot[0],(xbot[1]-xbot[0])/2,(ybot[1]-ybot[0])/2, color='m', shape='full', lw=0.25 ,length_includes_head=True, head_width=0.075)									# plot half the line with an arrow on the end to show the trajectory direction

view_range = [-map_limits,map_limits,-map_limits,map_limits]
plt.imshow(np.log10(A), cmap = 'jet', extent=view_range, aspect='auto', interpolation='bicubic')																																			# plot the mag map as an image, and define the axis range as +- the mag map limits
plt.colorbar()

title = 'Magnification map (d='+str(current_best_d)+', q='+str(current_best_q)+') of best grid search binary model parameters\n'+str(file_id[7:24])																																						# labels for the plot
xlabel = 'u1'
ylabel = 'u2'
plt.title(title,fontsize='small')
plt.xlabel(xlabel,fontsize='small')
plt.ylabel(ylabel,fontsize='small')
plt.savefig(str(file_id)+'output/grid_search_mag_map.png')																																																													# save the figure

#--- For plotting d, q parameter map.																																																													# plot the minimised d,q chi^2 map
d_q_map  = csv.reader(open(str(file_id)+'output/ftgs_d_q_map.txt','r'), delimiter=',', skipinitialspace=True)
vmax_calc = 0
line_number = -2
for line in d_q_map:
	if line_number == -2:
		line_number += 1
	elif line_number == -1:
		ld	=	np.float64(line[0])
		dd	=	np.float64(line[1])
		ud	=	np.float64(line[2])
		lq	=	np.float64(line[3])
		dq	=	np.float64(line[4])
		uq	=	np.float64(line[5])
		
		size_y = len(np.arange(ld,ud+dd,dd))
		size_x = len(np.arange(uq,lq-dq,-dq))
		
		d_arr	= np.zeros((size_x+1, size_y+1),np.float64)
		q_arr	= np.zeros((size_y+1, size_x+1),np.float64)
		chi2_arr	= np.zeros((size_x+1, size_y+1),np.float64)
		
		line_number += 1
	else:
		if np.float64(line[2]) == 9999999999:
			chi2_arr[(size_x-1-int(line_number/(size_y)),line_number-int(line_number/(size_y))*(size_y))]		= 0
		else:
			chi2_arr[(size_x-1-int(line_number/(size_y)),line_number-int(line_number/(size_y))*(size_y))]		= np.float64(line[2])
		
		line_number += 1

d_arr[0,:]	=	np.arange(ld,ud+2*dd,dd)
d_arr[1:,:]	=	d_arr[0,:]
q_arr[0,:]	=	10**(np.arange(lq,uq+2*dq,dq))
q_arr[1:,:]	=	q_arr[0,:]
q_arr = q_arr.transpose()																																																																		# method above is used to create repeating the in the y-axis, so transpose to make it repeat in the x-axis

max_chi2 = chi2_arr.max()
index = (chi2_arr == 0)
chi2_arr[index] = max_chi2

#vmax_calc = np.sum(chi2_arr) + np.min(chi2_arr)

plt.clf()
fig = plt.figure(figsize=(20.48,20.48))
ax = fig.add_subplot(1,1,1)
ax.set_yscale('log')
#ax.set_xscale('log')
plt.pcolor(d_arr,q_arr,chi2_arr, cmap='jet_r', edgecolors='w', linewidths=1)
plt.colorbar()

plt.xticks(np.arange(ld,(ud+dd),(dd)) )
plt.xlim(ld,ud+dd)
plt.ylim(10.0**lq,10.0**(uq+dq))

title = 'chi^2 map of minimised d/q values\n'+str(file_id[7:24])																																																		# set labels for the figure
xlabel = 'd values'
ylabel = 'q values'
plt.title(title,fontsize='small')
plt.xlabel(xlabel,fontsize='small')
plt.ylabel(ylabel,fontsize='small')
plt.savefig(str(file_id)+'output/complete_d_q_chi^2_map.png')																																																						# save figure


##--- For plotting the best fitted LC

single_chi2_calc = gpu_trajectories.get_function("single_chi2_calc")

for i in range(len(data_type)):
	if i == 0:
		plt.clf()
		plt.figure(figsize=(20.48,20.48))
		plt.errorbar(ts[0:DATA_ROWS[0]],mags[0:DATA_ROWS[0]],sigs[0:DATA_ROWS[0]],fmt='o',ms=2,c='k')
		if data_type[i] == 0:											# MOA
		
			if DATA_ROWS[0] <= MAX_THREADS:																																																																	# prevents more threads then available to be called
				blockDim_x = 2**(np.floor(math.log(DATA_ROWS[0]-1,2))+1)																																																								# if less data then threads store the smallest 2^n value >= data size
			else:
				blockDim_x = MAX_THREADS
			
			a0 = np.zeros([MAX_DATA_SETS],np.float64)
			a1 = np.zeros([MAX_DATA_SETS],np.float64)
			chi2 = np.zeros([1],np.float64)

			blockshape_chi2 = (int(blockDim_x), 1, 1)
			gridshape_chi2 = (1,1)
			
			single_chi2_calc(np.float64(current_best_t0), np.float64(current_best_tE), np.float64(current_best_u0), np.float64(current_best_phi), np.float64(b), np.float64(map_limits), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS[0]), drv.In(ts[0:DATA_ROWS[0]]), drv.In(mags[0:DATA_ROWS[0]]), drv.In(sigs[0:DATA_ROWS[0]]), np.int32(data_sets), drv.In(data_type[0]), drv.Out(a0), drv.Out(a1), drv.Out(chi2), texrefs=[texref], block=blockshape_chi2, grid=gridshape_chi2)
			
			LC = gpu_trajectories.get_function("LC")																																																													# define the GPU function to return the interpolated mag values for given time steps (ts)
			
			new_ts = np.zeros([t],np.float32)
			new_ts = np.arange(ts[0], ts[DATA_ROWS[i]-1], (ts[DATA_ROWS[i]-1]-ts[0])/t)
			
			u1, u2, tau = rd_zeta(u0n, current_best_phi, new_ts, t0n, current_best_tE, map_limits)																																											# calc the u1,u2 coordinates on the mag map for the time steps (ts)
			A_interp = np.zeros(len(new_ts),np.float32)																																																							# set up array size for GPU mag values output
			blockshape_LC = (100, 1, 1)																																																																# define block size for GPU
			gridshape_LC = ((int(len(new_ts))/100)+1, 1)																																																						# define grid size for GPU
			
			LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(t), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			
			A_interp = (A_interp-1)*Fo																																																										# Calc the flux of these mag values from the solved Fo value outputed from the CUDA func.
			
			plt.plot(new_ts,A_interp,'b-')
			
			#u1, u2, tau = rd_zeta(u0n, current_best_phi, ts, t0n, current_best_tE, map_limits)																																											# calc the u1,u2 coordinates on the mag map for the time steps (ts)
			#A_interp = np.zeros(DATA_ROWS[0],np.float32)																																																							# set up array size for GPU mag values output
			#blockshape_LC = (100, 1, 1)																																																																# define block size for GPU
			#gridshape_LC = ((int(DATA_ROWS[0])/100)+1, 1)																																																						# define grid size for GPU
			#LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(DATA_ROWS[0]), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			#A_interp = (A_interp-1)*Fo																																																										# Calc the flux of these mag values from the solved Fo value outputed from the CUDA func.

			#plt.plot(ts[0:DATA_ROWS[0]],A_interp,'b-')																																																																# Plot the LC from the calculated flux values
			
			minx = ts[0:DATA_ROWS[0]].min()																																																																				# Set limits for the plots dimensions
			maxx = ts[0:DATA_ROWS[0]].max()
			miny = mags[0:DATA_ROWS[0]].min()
			maxy = mags[0:DATA_ROWS[0]].max()+5000
			plt.axis([minx, maxx, miny, maxy])
			title = 'LC model of best fitted binary model parameters with MOA data\n'+str(file_id[7:24])																																											# Labels for the figures
			xlabel = 'JD - 2450000'
			ylabel = 'Delta flux'
			plt.title(title,fontsize='small')
			plt.xlabel(xlabel,fontsize='small')
			plt.ylabel(ylabel,fontsize='small')
			plt.savefig(str(file_id)+'output/Grid_search_MOA_best_parameters_LC.png')

		elif data_type[i] == 1:										# OGLE
			
			if DATA_ROWS[0] <= MAX_THREADS:																																																																	# prevents more threads then available to be called
				blockDim_x = 2**(np.floor(math.log(DATA_ROWS[0]-1,2))+1)																																																								# if less data then threads store the smallest 2^n value >= data size
			else:
				blockDim_x = MAX_THREADS
			
			a0 = np.zeros([MAX_DATA_SETS],np.float64)
			a1 = np.zeros([MAX_DATA_SETS],np.float64)
			chi2 = np.zeros([1],np.float64)

			blockshape_chi2 = (int(blockDim_x), 1, 1)
			gridshape_chi2 = (1,1)
			
			single_chi2_calc(np.float64(current_best_t0), np.float64(current_best_tE), np.float64(current_best_u0), np.float64(current_best_phi), np.float64(b), np.float64(map_limits), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS[0]), drv.In(ts[0:DATA_ROWS[0]]), drv.In(mags[0:DATA_ROWS[0]]), drv.In(sigs[0:DATA_ROWS[0]]), np.int32(data_sets), drv.In(data_type[0]), drv.Out(a0), drv.Out(a1), drv.Out(chi2), texrefs=[texref], block=blockshape_chi2, grid=gridshape_chi2)
			
			LC = gpu_trajectories.get_function("LC")
			
			new_ts = np.zeros([t],np.float32)
			new_ts = np.arange(ts[0], ts[DATA_ROWS[i]-1], (ts[DATA_ROWS[i]-1]-ts[0])/t)
			
			u1, u2, tau = rd_zeta(u0n, current_best_phi, new_ts, t0n, current_best_tE, map_limits)																																											# calc the u1,u2 coordinates on the mag map for the time steps (ts)
			A_interp = np.zeros(len(new_ts),np.float32)																																																							# set up array size for GPU mag values output
			blockshape_LC = (100, 1, 1)																																																																# define block size for GPU
			gridshape_LC = ((int(len(new_ts))/100)+1, 1)																																																						# define grid size for GPU
			
			LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(t), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			
			A_interp = m0[i] - 2.5*np.log10(A_interp*K+1-K)																																																												# Calc the flux of these mag values from the solved Fo value outputed from the CUDA func.
			
			plt.plot(new_ts,A_interp,'b-')
			
			#u1, u2, tau = rd_zeta(u0n, current_best_phi, ts[0:DATA_ROWS[0]], t0n, current_best_tE, map_limits)
			#A_interp = np.zeros(DATA_ROWS[0],np.float32)
			#blockshape_LC = (100, 1, 1)
			#gridshape_LC = ((int(DATA_ROWS[0])/100)+1, 1)			
			
			#LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(DATA_ROWS[0]), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			
			#A_interp = m0[i] - 2.5*np.log10(A_interp*K+1-K)	
			
			#plt.plot(ts[0:DATA_ROWS[0]],A_interp,'b-')
			minx = ts[0:DATA_ROWS[0]].min()
			maxx = ts[0:DATA_ROWS[0]].max()
			miny = mags[0:DATA_ROWS[0]].min()+-2
			maxy = mags[0:DATA_ROWS[0]].max()+1
			plt.axis([minx, maxx, maxy, miny])
			title = 'LC model of best fitted binary model parameters with OGLE data\n'+str(file_id[7:24])
			xlabel = 'JD - 2450000'
			ylabel = 'Delta flux'
			plt.title(title,fontsize='small')
			plt.xlabel(xlabel,fontsize='small')
			plt.ylabel(ylabel,fontsize='small')
			plt.savefig(str(file_id)+'output/Grid_search_OGLE_best_parameters_LC.png')
			
	else:
		if DATA_ROWS[i] == 0:
			break
		plt.clf()
		plt.errorbar(ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],sigs[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],fmt='o',ms=2,c='k')
		if data_type[i] == 0:											# MOA
			
			if DATA_ROWS[i] <= MAX_THREADS:																																																																	# prevents more threads then available to be called
				blockDim_x = 2**(np.floor(math.log(DATA_ROWS[i]-1,2))+1)																																																								# if less data then threads store the smallest 2^n value >= data size
			else:
				blockDim_x = MAX_THREADS
			
			a0 = np.zeros([MAX_DATA_SETS],np.float64)
			a1 = np.zeros([MAX_DATA_SETS],np.float64)
			chi2 = np.zeros([1],np.float64)

			blockshape_chi2 = (int(blockDim_x), 1, 1)
			gridshape_chi2 = (1,1)
			single_chi2_calc(np.float64(current_best_t0), np.float64(current_best_tE), np.float64(current_best_u0), np.float64(current_best_phi), np.float64(b), np.float64(map_limits), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS[i]), drv.In(ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])]), drv.In(mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])]), drv.In(sigs[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])]), np.int32(data_sets), drv.In(data_type[i]), drv.Out(a0), drv.Out(a1), drv.Out(chi2), texrefs=[texref], block=blockshape_chi2, grid=gridshape_chi2)

			LC = gpu_trajectories.get_function("LC")																																																													# define the GPU function to return the interpolated mag values for given time steps (ts)
			
			new_ts = np.zeros([t],np.float32)
			new_ts = np.arange(ts[np.sum(DATA_ROWS[0:i])], ts[np.sum(DATA_ROWS[0:i+1])-1], (ts[np.sum(DATA_ROWS[0:i+1])-1]-ts[np.sum(DATA_ROWS[0:i])])/t)
			
			u1, u2, tau = rd_zeta(u0n, current_best_phi, new_ts, t0n, current_best_tE, map_limits)																																											# calc the u1,u2 coordinates on the mag map for the time steps (ts)
			A_interp = np.zeros(len(new_ts),np.float32)																																																							# set up array size for GPU mag values output
			blockshape_LC = (100, 1, 1)																																																																# define block size for GPU
			gridshape_LC = ((int(len(new_ts))/100)+1, 1)																																																						# define grid size for GPU
			
			LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(t), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			
			A_interp = (A_interp-1)*Fo																																																										# Calc the flux of these mag values from the solved Fo value outputed from the CUDA func.
			
			plt.plot(new_ts,A_interp,'b-')
			
			#u1, u2, tau = rd_zeta(u0n, current_best_phi, ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])], t0n, current_best_tE, map_limits)																																											# calc the u1,u2 coordinates on the mag map for the time steps (ts)
			#A_interp = np.zeros(DATA_ROWS[i],np.float32)																																																							# set up array size for GPU mag values output
			#blockshape_LC = (100, 1, 1)																																																																# define block size for GPU
			#gridshape_LC = ((int(DATA_ROWS[i])/100)+1, 1)																																																						# define grid size for GPU
			
			#LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(DATA_ROWS[i]), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			#A_interp = (A_interp-1)*Fo																																																										# Calc the flux of these mag values from the solved Fo value outputed from the CUDA func.
			
			#plt.plot(ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],A_interp,'b-')																																																																# Plot the LC from the calculated flux values
			minx = ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].min()																																																																				# Set limits for the plots dimensions
			maxx = ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].max()
			miny = mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].min()
			maxy = mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].max()+5000
			plt.axis([minx, maxx, miny, maxy])
			title = 'LC model of best fitted binary model parameters with MOA data\n'+str(file_id[7:24])																																											# Labels for the figures
			xlabel = 'JD - 2450000'
			ylabel = 'Delta flux'
			plt.title(title,fontsize='small')
			plt.xlabel(xlabel,fontsize='small')
			plt.ylabel(ylabel,fontsize='small')
			plt.savefig(str(file_id)+'output/Grid_search_MOA_best_parameters_LC.png')
			
		elif data_type[i] == 1:										# OGLE
			
			if DATA_ROWS[i] <= MAX_THREADS:																																																																	# prevents more threads then available to be called
				blockDim_x = 2**(np.floor(math.log(DATA_ROWS[i]-1,2))+1)																																																								# if less data then threads store the smallest 2^n value >= data size
			else:
				blockDim_x = MAX_THREADS
			
			a0 = np.zeros([MAX_DATA_SETS],np.float64)
			a1 = np.zeros([MAX_DATA_SETS],np.float64)
			chi2 = np.zeros([1],np.float64)

			blockshape_chi2 = (int(blockDim_x), 1, 1)
			gridshape_chi2 = (1,1)
			
			single_chi2_calc(np.float64(current_best_t0), np.float64(current_best_tE), np.float64(current_best_u0), np.float64(current_best_phi), np.float64(b), np.float64(map_limits), np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), drv.In(DATA_ROWS[i]), drv.In(ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])]), drv.In(mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])]), drv.In(sigs[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])]), np.int32(data_sets), drv.In(data_type[i]), drv.Out(a0), drv.Out(a1), drv.Out(chi2), texrefs=[texref], block=blockshape_chi2, grid=gridshape_chi2)

			LC = gpu_trajectories.get_function("LC")
			
			new_ts = np.zeros([t],np.float32)
			new_ts = np.arange(ts[np.sum(DATA_ROWS[0:i])], ts[np.sum(DATA_ROWS[0:i+1])-1], (ts[np.sum(DATA_ROWS[0:i+1])-1]-ts[np.sum(DATA_ROWS[0:i])])/t)
			
			u1, u2, tau = rd_zeta(u0n, current_best_phi, new_ts, t0n, current_best_tE, map_limits)																																											# calc the u1,u2 coordinates on the mag map for the time steps (ts)
			A_interp = np.zeros(len(new_ts),np.float32)																																																							# set up array size for GPU mag values output
			blockshape_LC = (100, 1, 1)																																																																# define block size for GPU
			gridshape_LC = ((int(len(new_ts))/100)+1, 1)																																																						# define grid size for GPU
			
			LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(t), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			
			A_interp = m0[i] - 2.5*np.log10(A_interp*K+1-K)																																																										# Calc the flux of these mag values from the solved Fo value outputed from the CUDA func.
			
			plt.plot(new_ts,A_interp,'b-')
			
			#u1, u2, tau = rd_zeta(u0n, current_best_phi, ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])], t0n, current_best_tE, map_limits)
			#A_interp = np.zeros(DATA_ROWS[i],np.float32)
			#blockshape_LC = (100, 1, 1)
			#gridshape_LC = ((int(DATA_ROWS[i])/100)+1, 1)
			#LC(np.int32(num_points), np.float32(x0), np.float32(x0+dx*num_points-dx), np.int32(DATA_ROWS[i]), np.float32(map_limits), np.float64(b), drv.In(u1), drv.In(u2), drv.Out(A_interp), texrefs=[texref], block=blockshape_LC, grid=gridshape_LC)
			#A_interp = m0[i] - 2.5*np.log10(A_interp*K+1-K)	
			
			#plt.plot(ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])],A_interp,'b-')
			minx = ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].min()
			maxx = ts[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].max()
			miny = mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].min()+-2
			maxy = mags[DATA_ROWS[i-1]:np.sum(DATA_ROWS[0:i+1])].max()+1
			plt.axis([minx, maxx, maxy, miny])
			title = 'LC model of best fitted binary model parameters with OGLE data\n'+str(file_id[7:24])
			xlabel = 'JD - 2450000'
			ylabel = 'Delta flux'
			plt.title(title,fontsize='small')
			plt.xlabel(xlabel,fontsize='small')
			plt.ylabel(ylabel,fontsize='small')
			plt.savefig(str(file_id)+'output/Grid_search_OGLE_best_parameters_LC.png')
			
localtime = time.asctime( time.localtime(time.time()) )	
print ''
print 'Code finished ('+str(localtime)+')'
print 'Exiting...'
print '\a'
print '\a'
print '\a'
