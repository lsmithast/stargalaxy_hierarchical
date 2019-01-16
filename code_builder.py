
from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source("_fitter",
   r""" // passed to the real C compiler
#include <sys/types.h>
#include <pwd.h>
#include<math.h>
//#include "lbfgs.h"

#define MAX(x,y) ((x)>(y)?(x):(y))
static float logaddexp(float last, float new)
{
   float max,min;
   if (last>new) 
   {max=last;min=new;}
   else{max=new;min=last;}
   return max+log1pf(exp(min-max));
   //float max = MAX(last,new);
   //return log(exp(last-max)+exp(new-max))+max;
}

void procobj(
   float *ima,
   float *var,
   float *grid,
   int npix,
   int ngrid,
   float *retlike)
   {
      int i,j,k;
      float mult;
      float accum0=0;
      for(j=0;j<npix;j++)
      { 
          for(k=0;k<npix;k++)
          {
              int curx = j*npix+k;
              accum0 += -0.5*log(var[curx]*2*M_PI);
          }
      }  


      for (i=0;i<ngrid;i++)
      {
          float accum1=0, accum2=0;

          for(j=0;j<npix;j++)
          {
              for (k=0;k<npix;k++)
              {
                  int curx = j*npix+k;
                  int curx1 = npix*npix*i+j*npix+k ;
                  accum1 += ima[curx]/var[curx] * grid[curx1];
                  accum2 += grid[curx1]*grid[curx1]/var[curx] ;
                  //fprintf(stderr,"X %d %d %d %f %f %f\n",i,curx, curx1,ima[curx],var[curx], grid[curx1]);
              }
          }
          mult  = accum1/accum2;
          //fprintf(stderr,"%f %f %f\n",accum1,accum2,mult);
          accum1 = accum0;//-INFINITY;
          for(j=0;j<npix;j++)
          {
              for(k=0;k<npix;k++)
              {
                  int curx = j*npix+k;
                  int curx1 = npix*npix*i+j*npix+k ;
                  float diff = ima[curx] - mult * grid[curx1];
                  //accum1 = logaddexp(accum1, -0.5*log(var[curx]) -0.5 * diff*diff/var[curx]);
                  accum1 += -0.5 * diff*diff/var[curx];
              }
          }
          retlike[i] = accum1;
      }
      //return 0;
   }



    """,
    libraries=['m'],#
#	include_dirs=['liblbfgs-1.10/include/'],#
#	library_dirs=['liblbfgs-1.10/lib/.libs/'],
	)   # or a list of libraries to link with
    # (more arguments like setup.py's Extension class:
    # include_dirs=[..], extra_objects=[..], and so on)

ffibuilder.cdef("""     // some declarations from the man page
    struct passwd {
        char *pw_name;
        ...;     // literally dot-dot-dot
    };
    struct passwd *getpwuid(int uid);

void procobj(
   float *ima,
   float *var,
   float *grid,
   int npix,
   int ngrid,
   float *retlike);
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
