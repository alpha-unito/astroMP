#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cmath>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <iomanip>
#include <fstream>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include <omp.h>

// π
#define CONST_PI      3.14159265358979
// Universal gravitational constant taken as 1
#define G      1.0
// Universal gravitational constant in (kpc/Msun)*(km/s)^2
#define Gtrue      4.3011e-6
#define Ydfitt      1.0
// MOND acceleration scale in (kpc^-1)*(km/s)^2
#define a0true      3600.0
// Ending condition of the Poisson solver
#define tol 1.e-9



// In this example all the following quantites are defined in the following units of measure
// Lenghts in unit of the galaxy disk scale lenghts (hR)
// Luminosities in unit of the galaxy total luminosity (Ld)
// Masses in unit of Md = (1 Msun/Lsun)*Ld
// Surface brithnesses in unit of Iunit = Ld/(hR^2)
// Surface mass densities in unit of (1 Msun/Lsun)*Iunit
// 3D luminositu densities in unit of Ld/(hR^3)
// 3D mass densities in unit of rhounit = (1 Msun/Lsun)*(Ld/(hR^3))
// Gravitational potential in unit of phiunit = Gtrue*Mdunit/hR
// Gravitational field in unit of gunit = Gtrue*Mdunit/(hR^2)
// Velocities in unit of vunit = sqrt(Gtrue*Mdunit/hR)



// Funtion used in the Poisson solver
double Error(double **V, double **V_New, int A1, int B1);

using namespace std;



int main()
{
//    DECOMMENT THE BELOW PART OF CODE IF YOU WANT TO MEASURE THE TIME FOR EVERY Monte Carlo Markov Chain (MCMC) ITERATION
//    struct timeval start[Niters + 1], end[Niters + 1];
    
//    double time_taken[Niters + 1];
    
    
    
//    Number of galaxies (NGal), threads on which perform the parallel computation (Nthreads), free parameters of the model to estimate with the MCMC (Npar) and iterations of the MCMC (Niters)
    int NGal, Nthreads, Npar, Niters;
    
    ifstream infileA ("NGal.txt");
    {
        infileA>>NGal;
    }
    
    infileA.close();
    
//    Each quantity is related to every galaxy in the considered sample
    struct Galaxies {
//        Number of grid points in the radial direction
        int NR;
//        Number of grid points in the vertical direction
        int NZ;
//        Number of points of the measured rotation curve
        int Nptv;
//        Number of points of the measured vertical velocity dispersion
        int Nptsigz;
//        Disk scale length
        double hR;
//        Central surface brightness of the disk
        double Id0;
//        Observed disk scale height: it is useful to define the grid in this example
        double hzformula;
//        Ending point of the radial dimension of the grid
        double xend;
//        Disk scale height
        double hzMCMC;
//        Mass-to-light ratio
        double YMCMC;
//        Number of maximum iterations of the Poisson solver
        int NitersSORbreak;
    } params[NGal];
    
    ifstream infileB ("Nthreads.txt");
    {
        infileB>>Nthreads;
    }
    
    infileB.close();
    
    ifstream infileC ("Npar.txt");
    {
        infileC>>Npar;
    }
    
    infileC.close();
    
    ifstream infileD ("Niters.txt");
    {
        infileD>>Niters;
    }
    
    infileD.close();
    
//    Conversion factor from g/cm^3 to Msum/kpc^3
    double goncm3inMsunonkpc3 = 1.47759e31;
    
    
    
//    Units for the various quantities involved in the code (see the above comments), starting point of the grid in the radial dimension (xbeg), starting and ending points of the grid in the vertical dimentsion (zbeg, zend), grid steps in the radial and vertical dimensions (dx = dR, dz), squared grid steps in the radial and vertial dimensions (dx2 = dR2, dz2), quantity proportional to the difference between the gravitational potential at the next and at the previous iterations in the Poisson solver (err), parameter that regulates the Poisson solver convergence speed (wsor) and total number of degrees of freedom of the model (Ndoftot)
    double Ld[NGal], Mdunit[NGal], Iunit[NGal], rhounit[NGal], gunit[NGal], a0[NGal], vunit[NGal], xbeg[NGal], zbeg[NGal], zend[NGal], dx[NGal], dz[NGal], dx2[NGal], dz2[NGal], dR[NGal], dR2[NGal], err[NGal], wsor[NGal], Ndoftot[NGal];
    
//    Import the number of grid points in the radial direction for every galaxy in the sample
    ifstream infile1 ("NR_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile1>>params[i].NR;
    }
    
    infile1.close();
    
//    Import the number of grid points in the vertical direction for every galaxy in the sample
    ifstream infile2 ("NZ_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile2>>params[i].NZ;
    }
    
    infile2.close();
    
    double **x, **R, **sigz;
    x = new double*[NGal];
    R = new double*[NGal];
    sigz = new double*[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
        x[i] = new double[params[i].NR + 1];
        R[i] = new double[params[i].NR + 1];
        sigz[i] = new double[params[i].NR + 1];
    }
    
    double **z;
    z = new double*[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
        z[i] = new double[params[i].NZ + 1];
    }
    
    double **Rhalf, **vbar, **sigzbar;
    Rhalf = new double*[NGal];
    vbar = new double*[NGal];
    sigzbar = new double*[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
        Rhalf[i] = new double[(params[i].NR - 1)/2 + 1];
        vbar[i] = new double[(params[i].NR - 1)/2 + 1];
        sigzbar[i] = new double[(params[i].NR - 1)/2 + 1];
    }
    
    double ***rho, ***S, ***DRphiN, ***phi0, ***phi1, ***DRphi1, ***Dzphi1, ***eps, ***DReps, ***Dzeps, ***integrandf1, ***Atrap1, ***integralf12D, ***Atrap2;
    rho = new double**[NGal];
    S = new double**[NGal];
    DRphiN = new double**[NGal];
    phi0 = new double**[NGal];
    phi1 = new double**[NGal];
    DRphi1 = new double**[NGal];
    Dzphi1 = new double**[NGal];
    eps = new double**[NGal];
    DReps = new double**[NGal];
    Dzeps = new double**[NGal];
    integrandf1 = new double**[NGal];
    Atrap1 = new double**[NGal];
    integralf12D = new double**[NGal];
    Atrap2 = new double**[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
        rho[i] = new double*[params[i].NR + 1];
        S[i] = new double*[params[i].NR + 1];
        DRphiN[i] = new double*[params[i].NR + 1];
        phi0[i] = new double*[params[i].NR + 1];
        phi1[i] = new double*[params[i].NR + 1];
        DRphi1[i] = new double*[params[i].NR + 1];
        Dzphi1[i] = new double*[params[i].NR + 1];
        eps[i] = new double*[params[i].NR + 1];
        DReps[i] = new double*[params[i].NR + 1];
        Dzeps[i] = new double*[params[i].NR + 1];
        integrandf1[i] = new double*[params[i].NR + 1];
        Atrap1[i] = new double*[params[i].NR + 1];
        integralf12D[i] = new double*[params[i].NR + 1];
        Atrap2[i] = new double*[params[i].NR + 1];
        
        for(int j = 0; j <= params[i].NR; j++)
        {
            rho[i][j] = new double[params[i].NZ + 1];
            S[i][j] = new double[params[i].NZ + 1];
            DRphiN[i][j] = new double[params[i].NZ + 1];
            phi0[i][j] = new double[params[i].NZ + 1];
            phi1[i][j] = new double[params[i].NZ + 1];
            DRphi1[i][j] = new double[params[i].NZ + 1];
            Dzphi1[i][j] = new double[params[i].NZ + 1];
            eps[i][j] = new double[params[i].NZ + 1];
            DReps[i][j] = new double[params[i].NZ + 1];
            Dzeps[i][j] = new double[params[i].NZ + 1];
            integrandf1[i][j] = new double[params[i].NZ + 1];
            Atrap1[i][j] = new double[params[i].NZ + 1];
            integralf12D[i][j] = new double[params[i].NZ + 1];
            Atrap2[i][j] = new double[params[i].NZ + 1];
        }
    }
    
    
//    Import the number of points of the measured rotation curve for every galaxy in the sample
    ifstream infile3 ("Nptv_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile3>>params[i].Nptv;
    }
    
    infile3.close();
    
//    Import the number of points of the measured vertical velocity dispersion for every galaxy in the sample
    ifstream infile4 ("Nptsigz_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile4>>params[i].Nptsigz;
    }
    
    infile4.close();
    
    double **rvvalue, **vvalue, **devstvvalue, **vbarinterp_X, **vbarinterp_Y, **argchi2v_X, **argchi2v_Y;
    rvvalue = new double*[NGal];
    vvalue = new double*[NGal];
    devstvvalue = new double*[NGal];
    vbarinterp_X = new double*[NGal];
    vbarinterp_Y = new double*[NGal];
    argchi2v_X = new double*[NGal];
    argchi2v_Y = new double*[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
        rvvalue[i] = new double[params[i].Nptv + 1];
        vvalue[i] = new double[params[i].Nptv + 1];
        devstvvalue[i] = new double[params[i].Nptv + 1];
        vbarinterp_X[i] = new double[params[i].Nptv + 1];
        vbarinterp_Y[i] = new double[params[i].Nptv + 1];
        argchi2v_X[i] = new double[params[i].Nptv + 1];
        argchi2v_Y[i] = new double[params[i].Nptv + 1];
    }
    
    
    
    double **rsigzvalue, **sigzvalue, **devstsigzvalue, **sigzbarinterp_X, **sigzbarinterp_Y, **argchi2sigz_X, **argchi2sigz_Y;
    rsigzvalue = new double*[NGal];
    sigzvalue = new double*[NGal];
    devstsigzvalue = new double*[NGal];
    sigzbarinterp_X = new double*[NGal];
    sigzbarinterp_Y = new double*[NGal];
    argchi2sigz_X = new double*[NGal];
    argchi2sigz_Y = new double*[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
        rsigzvalue[i] = new double[params[i].Nptsigz + 1];
        sigzvalue[i] = new double[params[i].Nptsigz + 1];
        devstsigzvalue[i] = new double[params[i].Nptsigz + 1];
        sigzbarinterp_X[i] = new double[params[i].Nptsigz + 1];
        sigzbarinterp_Y[i] = new double[params[i].Nptsigz + 1];
        argchi2sigz_X[i] = new double[params[i].Nptsigz + 1];
        argchi2sigz_Y[i] = new double[params[i].Nptsigz + 1];
    }
    
//    chi square of the model rotation curve for the first part of the MCMC (chi2v_X)
//    chi square of the model rotation curve for the second part of the MCMC (chi2v_Y)
//    chi square of the model vertical velocity dispersion for the first part of the MCMC (chi2sigz_X)
//    chi square of the model vertical velocity dispersion for the second part of the MCMC (chi2sigz_X)
//    chi square for one single galaxy = (rotation curve chi2 + vertical velocity dispersion chi2)/(total number of degrees of freedom) for the first part of the MCMC (chi2red_X)
//    chi square for one single galaxy = (rotation curve chi2 + vertical velocity dispersion chi2)/(total number of degrees of freedom) for the second part of the MCMC (chi2red_Y)
    double chi2v_X[NGal], chi2v_Y[NGal], chi2sigz_X[NGal], chi2sigz_Y[NGal], chi2red_X[NGal], chi2red_Y[NGal];
    
//    chi square for all the galaxies considered at the same time given by the sum of the chi squares for every galaxy for the first part of the MCMC (chi2redtot_X)
//    chi square for all the galaxies considered at the same time given by the sum of the chi squares for every galaxy for the second part of the MCMC (chi2redtot_Y)
    double chi2redtot_X, chi2redtot_Y;
    
//    Free parameters of the model for the first (_X) and the second (_Y) part of the MCMC
    double eps0_X[Niters + 1], Q_X[Niters + 1], logrhoc_X[Niters + 1], rhoc_X[Niters + 1];
    double eps0_Y, Q_Y, logrhoc_Y, rhoc_Y;
    
//    Parameters useful to define the boundary conditions of the potential
    double b[NGal], M1onMd[NGal], M2onMd[NGal], M3onMd[NGal], a1[NGal], a2[NGal], a3[NGal];
    

//    Import the disk scale length for every galaxy in the sample
    ifstream infile5 ("hR_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile5>>params[i].hR;
    }
    
    infile5.close();
    
//    Import the central surface brightness for every galaxy in the sample
    ifstream infile6 ("Id0_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile6>>params[i].Id0;
    }
    
    infile6.close();
    
//    Unit of measures of the various quantities involved (see the comments above)
    for(int i = 0; i < NGal; i++)
    {
        Ld[i] = 2.0*CONST_PI*params[i].Id0*params[i].hR*params[i].hR;
        Mdunit[i] = Ydfitt*Ld[i];
        Iunit[i] = Ld[i]/(params[i].hR*params[i].hR);
        rhounit[i] = Mdunit[i]/(params[i].hR*params[i].hR*params[i].hR);
        gunit[i] = Gtrue*Mdunit[i]/(params[i].hR*params[i].hR);
        a0[i] = a0true/gunit[i];
        vunit[i] = sqrt(Gtrue*Mdunit[i]/params[i].hR);
    }
    
    for(int i = 0; i < NGal; i++) params[i].Id0 = params[i].Id0/Iunit[i];
    
//    Import the observed disk scale height for every galaxy in the sample (useful to define the grid in this example)
    ifstream infile7 ("hzformula_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile7>>params[i].hzformula;
    }
    
    infile7.close();
    
    for(int i = 0; i < NGal; i++) params[i].hzformula = params[i].hzformula/params[i].hR;
    
//    Import the  ending point of the radial dimension of the grid for every galaxy in the sample
    ifstream infile8 ("xend_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile8>>params[i].xend;
    }
    
    infile8.close();
    
//    Definition of the starting point of the radial dimension of the grid (xbeg) and of the starding and ending points of the vertical dimension of the grid (zbeg, zend)
    for(int i = 0; i < NGal; i++)
    {
        xbeg[i] = -params[i].xend;
        zbeg[i] = -100.0*params[i].hzformula;
        zend[i] = 100.0*params[i].hzformula;
    }
    
//    Grid steps in the radial (dx) and vertical (dz) dimensions of the grid and squared grid steps in the radial (dx2) and vertical (dz2) dimensions of the grid dimensions
    for(int i = 0; i < NGal; i++)
    {
        dx[i] = (params[i].xend - xbeg[i])/double(params[i].NR);
        dz[i] = (zend[i] - zbeg[i])/double(params[i].NZ);
        dx2[i] = dx[i]*dx[i];
        dz2[i] = dz[i]*dz[i];
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j <= params[i].NR; j++)
        {
//            Grid definition in the radial dimension
            x[i][j] = xbeg[i] + j*dx[i];
            R[i][j] = x[i][j];
        }
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for(int k = 0; k <= params[i].NZ; k++)
        {
//            Grid definition in the vertical dimension
            z[i][k] = zbeg[i] + k*dz[i];
        }
    }
    
    for(int i = 0; i < NGal; i++)
    {
        dR[i] = dx[i];
        dR2[i] = dx2[i];
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < (params[i].NR - 1)/2 + 1; j++)
        {
//            Radius in the right half of the grid
            Rhalf[i][j] = R[i][j + (params[i].NR - 1)/2 + 1];
        }
    }
    
//    Import the mass density for each galaxy in the sample
    ifstream infile9 ("rho_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j <= params[i].NR; j++)
            {
                for(int k = 0; k <= params[i].NZ; k++)
                {
                    infile9>>rho[i][j][k];
                }
            }
        }
    }
    
    infile9.close();
    
//    Initialization of the gravitational potential to 0 in every point of the grid
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j <= params[i].NR; j++)
        {
            for(int k = 0; k <= params[i].NZ; k++)
            {
                phi0[i][j][k] = 0.0;
            }
        }
    }
    
    
//    Defintion of the linear interpolators used to interplolate the modelled rotation curves and vertical velocity dispersion profiles in the points of the measured data to compare the modelled and measured profiles and to compute their chi squares
    gsl_interp **interp1, **interp2;
    interp1 = new gsl_interp*[NGal];
    interp2 = new gsl_interp*[NGal];
    
    for(int i = 0; i < NGal; i++)
    {
//        Linear interpolator for the rotation curves
        interp1[i] = gsl_interp_alloc(gsl_interp_linear, (params[i].NR - 1)/2 + 1);
        gsl_interp_init (interp1[i], Rhalf[i], vbar[i], (params[i].NR - 1)/2 + 1);
        
//        Linear interpolator for the vertical velocity dispersion profiles
        interp2[i] = gsl_interp_alloc(gsl_interp_linear, (params[i].NR - 1)/2 + 1);
        gsl_interp_init (interp2[i], Rhalf[i], sigzbar[i], (params[i].NR - 1)/2 + 1);
    }
    
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    
//    Import the radii correspondent to the measured rotation curves for every galaxy in the sample
    ifstream infile10 ("rvvalues_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j < params[i].Nptv; j++)
            {
                infile10>>rvvalue[i][j];
            }
        }
    }
    
    infile10.close();
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < params[i].Nptv; j++)
        {
            rvvalue[i][j] = rvvalue[i][j]/params[i].hR;
        }
    }
    
//    Import the the measured rotation curves for every galaxy in the sample
    ifstream infile11 ("vvalues_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j < params[i].Nptv; j++)
            {
                infile11>>vvalue[i][j];
            }
        }
    }
    
    infile11.close();
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < params[i].Nptv; j++)
        {
            vvalue[i][j] = vvalue[i][j]/vunit[i];
        }
    }
    
//    Import the uncertainties of the measured rotation curves for every galaxy in the sample
    ifstream infile12 ("devstvvalues_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j < params[i].Nptv; j++)
            {
                infile12>>devstvvalue[i][j];
            }
        }
    }
    
    infile12.close();
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < params[i].Nptv; j++)
        {
            devstvvalue[i][j] = devstvvalue[i][j]/vunit[i];
        }
    }
    
    
    
//    Import the radii correspondent to the measured vertical velocity dispersion profiles for every galaxy in the sample
    ifstream infile13 ("rsigzvalues_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j < params[i].Nptsigz; j++)
            {
                infile13>>rsigzvalue[i][j];
            }
        }
    }
    
    infile13.close();
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < params[i].Nptsigz; j++)
        {
            rsigzvalue[i][j] = rsigzvalue[i][j]/params[i].hR;
        }
    }
    
//    Import the the measured vertical velocity dispersion profiles for every galaxy in the sample
    ifstream infile14 ("sigzvalues_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j < params[i].Nptsigz; j++)
            {
                infile14>>sigzvalue[i][j];
            }
        }
    }
    
    infile14.close();
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < params[i].Nptsigz; j++)
        {
            sigzvalue[i][j] = sigzvalue[i][j]/vunit[i];
        }
    }
    
//    Import the uncertainties of the measured vertical velocity dispersion profiles for every galaxy in the sample
    ifstream infile15 ("devstsigzvalues_DMS_Ppak_Galaxies.txt");
    {
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j < params[i].Nptsigz; j++)
            {
                infile15>>devstsigzvalue[i][j];
            }
        }
    }
    
    infile15.close();
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j < params[i].Nptsigz; j++)
        {
            devstsigzvalue[i][j] = devstsigzvalue[i][j]/vunit[i];
        }
    }
    
    
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j <= params[i].NR; j++)
        {
            sigz[i][j] = 0.0;
        }
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j <= params[i].NR; j++)
        {
            for(int k = 0; k <= params[i].NZ; k++)
            {
                integralf12D[i][j][k] = 0.0;
            }
        }
    }



    for(int i = 0; i < NGal; i++) Ndoftot[i] = params[i].Nptv + params[i].Nptsigz - Npar;
    
    for(int i = 0; i < NGal; i++)
    {
        chi2v_X[i] = 0.0;
        chi2v_Y[i] = 0.0;
        chi2sigz_X[i] = 0.0;
        chi2sigz_Y[i] = 0.0;
        chi2red_X[i] = 0.0;
        chi2red_Y[i] = 0.0;
    }
    
    chi2redtot_X = 0.0;
    chi2redtot_X = 0.0;
    
    
    
//    Import the disk scale heights for every galaxy in the sample
    ifstream infile16 ("hzMCMC_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile16>>params[i].hzMCMC;
    }
    
    infile16.close();
    
    for(int i = 0; i < NGal; i++) params[i].hzMCMC = params[i].hzMCMC/params[i].hR;
    
//    Quantities useful to the definition of the gravitational potential boundary conditions
    for(int i = 0; i < NGal; i++)
    {
        b[i] = -0.269*params[i].hzMCMC*params[i].hzMCMC*params[i].hzMCMC + 1.080*params[i].hzMCMC*params[i].hzMCMC + 1.092*params[i].hzMCMC;
        
        M1onMd[i] = -0.0090*b[i]*b[i]*b[i]*b[i] + 0.0640*b[i]*b[i]*b[i] -0.1653*b[i]*b[i] + 0.1164*b[i] + 1.9487;
        M2onMd[i] = 0.0173*b[i]*b[i]*b[i]*b[i] - 0.0903*b[i]*b[i]*b[i] + 0.0877*b[i]*b[i] + 0.2029*b[i] - 1.3077;
        M3onMd[i] = -0.0051*b[i]*b[i]*b[i]*b[i] + 0.0287*b[i]*b[i]*b[i] - 0.0361*b[i]*b[i] - 0.0544*b[i] + 0.2242;
        
        a1[i] = -0.0358*b[i]*b[i]*b[i]*b[i] + 0.2610*b[i]*b[i]*b[i] - 0.6987*b[i]*b[i] - 0.1193*b[i] + 2.0074;
        a2[i] = -0.0830*b[i]*b[i]*b[i]*b[i] + 0.4992*b[i]*b[i]*b[i] - 0.7967*b[i]*b[i] - 1.2966*b[i] + 4.4441;
        a3[i] = -0.0247*b[i]*b[i]*b[i]*b[i] + 0.1718*b[i]*b[i]*b[i] - 0.4124*b[i]*b[i] - 0.5944*b[i] + 0.7333;
    }
    
//    Import the mass-to-light ratios for every galaxy in the sample
    ifstream infile17 ("YMCMC_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile17>>params[i].YMCMC;
    }
    
    infile17.close();
    
    
//    Quantity useful to the definition of the gravitational potential boundary conditions
    for(int i = 0; i < NGal; i++)
    {
        for (int j = 0; j <= params[i].NR; j++)
        {
            for (int k = 0; k <= params[i].NZ; k++)
            {
                DRphiN[i][j][k] = G*M1onMd[i]*params[i].YMCMC*R[i][j]/sqrt((R[i][j]*R[i][j] + (a1[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a1[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))*(R[i][j]*R[i][j] + (a1[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a1[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))*(R[i][j]*R[i][j] + (a1[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a1[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))) + G*M2onMd[i]*params[i].YMCMC*R[i][j]/sqrt((R[i][j]*R[i][j] + (a2[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a2[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))*(R[i][j]*R[i][j] + (a2[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a2[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))*(R[i][j]*R[i][j] + (a2[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a2[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))) + G*M3onMd[i]*params[i].YMCMC*R[i][j]/sqrt((R[i][j]*R[i][j] + (a3[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a3[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))*(R[i][j]*R[i][j] + (a3[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a3[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k])))*(R[i][j]*R[i][j] + (a3[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))*(a3[i] + sqrt(b[i]*b[i] + z[i][k]*z[i][k]))));
            }
        }
    }
    
    
//    Boundary Conditions for the radial and vertical derivatives of the gravitational potential (Neumann boundary conditions)
//    IF YOU WANT, YOU CAN DEFINE BOUNDARY CONSITIONS FOR THE GRAVITATIONAL POTENTIAL (DIRICHLET BOUNDARY CONDITIONS)
    
    //        BOTTOM
    
    for(int i = 0; i < NGal; i++)
    {
        for (int j = 0; j < (params[i].NR - 1)/2; j++) phi0[i][j+1][0] = phi0[i][j][0] - dR[i]*sqrt(a0[i]*DRphiN[i][j][0]);
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for (int j = (params[i].NR - 1)/2 + 1; j <= params[i].NR; j++) phi0[i][j][0] = phi0[i][params[i].NR - j][0];
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for (int j = 0; j <= params[i].NR; j++) phi0[i][j][1] = phi0[i][j][0];
    }
    
    

    //        TOP
    
    for(int i = 0; i < NGal; i++)
    {
        for (int j = 0; j < (params[i].NR - 1)/2; j++) phi0[i][j+1][params[i].NZ] = phi0[i][j][params[i].NZ] - dR[i]*sqrt(a0[i]*DRphiN[i][j][params[i].NZ]);
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for (int j = (params[i].NR - 1)/2 + 1; j <= params[i].NR; j++) phi0[i][j][params[i].NZ] = phi0[i][params[i].NR - j][params[i].NZ];
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for (int j = 0; j <= params[i].NR; j++) phi0[i][j][params[i].NZ - 1] = phi0[i][j][params[i].NZ];
    }
    
    
    
    //        LEFT
    
    for(int i = 0; i < NGal; i++)
    {
        for (int k = 0; k < params[i].NZ; k++) phi0[i][0][k+1] = phi0[i][0][k];
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for (int k = 0; k <= params[i].NZ; k++) phi0[i][1][k] = phi0[i][0][k] - dR[i]*sqrt(a0[i]*DRphiN[i][1][k]);
    }
    
    
    
    //        RIGHT
    
    for(int i = 0; i < NGal; i++)
    {
        for (int k = 0; k < params[i].NZ; k++) phi0[i][params[i].NR][k+1] = phi0[i][params[i].NR][k];
    }
    
    for(int i = 0; i < NGal; i++)
    {
        for (int k = 0; k <= params[i].NZ; k++) phi0[i][params[i].NR - 1][k] = phi0[i][params[i].NR][k] - dR[i]*sqrt(a0[i]*DRphiN[i][params[i].NR - 1][k]);
    }
    
    
    
//    Parameter inizialization for the MCMC. In this case three free parameters are considered
    
    eps0_X[0] = 0.70;
    Q_X[0] = 0.80;
    logrhoc_X[0] = -25.0;
    
   
    
//    Definition of the uniform pseudo random number generators used in the MCMC with the <boost> library
    
    unsigned long int random_seed;
    
//    Import the random seed used in the pseudo uniform random number generators from an external file
    ifstream infileSEED ("random_seed.txt");
    {
        infileSEED>>random_seed;
    }
    
    infileSEED.close();
    
    boost::random::mt19937 generator(random_seed);
    boost::random::mt19937 generator1(random_seed);
    
//    Definition of a uniform distribution between 0 and 1
    boost::random::uniform_real_distribution<double> distributionU(0.0,1.0);
//    Definition of a normal distribution with mean 0 and standard deviation 1
    boost::random::normal_distribution<double> UnitGaussian(0.0,1.0);
    

    
//    Defintion of the priors for the model parameters. In this case flat priors are used
    
    double eps0beg = 0.10;
    double eps0end = 1.00;
    
    double Qbeg = 0.01;
    double Qend = 2.00;
    
    double logrhocbeg = -27.0;
    double logrhocend = -23.0;
    
    double percentage = 0.1;
    
//    Standard deviations of the normal distributions used to made the "jump" from the previous to the next combination of parameters in the MCMC
    double stepsizeeps0 = percentage*(eps0end - eps0beg);
    double stepsizeQ = percentage*(Qend - Qbeg);
    double stepsizelogrhoc = percentage*(logrhocend - logrhocbeg);
    
//    Natural logarithm of Metropolis - Hastings ratio (lnrHastings) and a random number that follows the uniform distribution between 0 and 1 (U)
    double lnrHastings, U;
    
//    wsor parameter defined for the 18th galaxy in the sample
    wsor[17] = 2.0/(1.0 + CONST_PI/200.0);
    
//    Import the number of maximum iterations of the Poisson solver for every galaxies
    ifstream infile18 ("NitersSORbreak_List.txt");
    {
        for(int i = 0; i < NGal; i++) infile18>>params[i].NitersSORbreak;
    }
    
    infile18.close();
    
    int NitersSORbreak[NGal];
    
    for(int i = 0; i < NGal; i++) NitersSORbreak[i] = params[i].NitersSORbreak;
    
    int n[NGal];
    
    for(int i = 0; i < NGal; i++) n[i] = 0;
    
    for(int i = 0; i < NGal; i++) err[i] = 1.e30;
    
    
    
    
//    DECOMMENT THE BELOW PART OF CODE IF THE CODE WAS KILLED AND YOU WANT TO RESTART THE MCMC FROM A CERTAIN POINT
//    EXAMPLE WITH A CODE KILLED AT A MCMC ITERATION BETWEEN 7000 AND 8000
    
////////////////////////////////////////////////////////////////////////
//    int Iter_truncation = 7000;
//    
//    ifstream infile3g ("eps0_X_7000_mega_run_new_1_P.txt");
//    {
//        for(int t1 = 0; t1 <= Iter_truncation; t1++) infile3g>>eps0_X[t1];
//    }
//    
//    infile3g.close();
//    
//    ifstream infile4g ("Q_X_7000_mega_run_new_1_P.txt");
//    {
//        for(int t1 = 0; t1 <= Iter_truncation; t1++) infile4g>>Q_X[t1];
//    }
//    
//    infile4g.close();
//    
//    ifstream infile5g ("logrhoc_X_7000_mega_run_new_1_P.txt");
//    {
//        for(int t1 = 0; t1 <= Iter_truncation; t1++) infile5g>>logrhoc_X[t1];
//    }
//    
//    infile5g.close();
//    
//    
//    
//    ifstream infile1G ("generator_7000.txt");
//    {
//        infile1G>>generator;
//    }
//    
//    infile1G.close();
//    
//    ifstream infile2G ("generator1_7000.txt");
//    {
//        infile2G>>generator1;
//    }
//    
//    infile2G.close();
////////////////////////////////////////////////////////////////////////
    
    
    
    
    
//   Monte Carlo Markov Chain (MCMC): start
//    COMMENT THE BELOW PART OF CODE IF THE CODE WAS KILLED AND YOU WANT TO RESTART THE MCMC FROM A CERTAIN POINT
    for(int t1 = 0; t1 < Niters; t1++)
    {
        
//    DECOMMENT THE BELOW PART OF CODE IF THE CODE WAS KILLED AND YOU WANT TO RESTART THE MCMC FROM A CERTAIN POINT
//    for(int t1 = Iter_truncation; t1 < Niters; t1++)
//    {
        
        
        
//        DECOMMENT THE BELOW PART OF CODE IF YOU WANT TO MEASURE THE TIME FOR EVERY MCMC ITERATION
//        gettimeofday(&start[t1], NULL);
//        
//        ios_base::sync_with_stdio(false);
        
        
        
//        Print to external files the states of the uniform pseudo random number generators every 1000 MCMC iterations: it is useful if the code is killed and you do not want to restart the MCMC from the beginning.
//        The below example is done for a total (burn-in + effective) number of MCMC iterations equal to 20000.
        if(t1 == 1000)
        {
            ofstream outfile1A ("generator_1000.txt");
            {
                outfile1A << generator << endl;
            }
            outfile1A.close();
            
            ofstream outfile2A ("generator1_1000.txt");
            {
                outfile2A << generator1 << endl;
            }
            outfile2A.close();
        }
        
        if(t1 == 2000)
        {
            ofstream outfile1B ("generator_2000.txt");
            {
                outfile1B << generator << endl;
            }
            outfile1B.close();
            
            ofstream outfile2B ("generator1_2000.txt");
            {
                outfile2B << generator1 << endl;
            }
            outfile2B.close();
        }
        
        if(t1 == 3000)
        {
            ofstream outfile1C ("generator_3000.txt");
            {
                outfile1C << generator << endl;
            }
            outfile1C.close();
            
            ofstream outfile2C ("generator1_3000.txt");
            {
                outfile2C << generator1 << endl;
            }
            outfile2C.close();
        }
        
        if(t1 == 4000)
        {
            ofstream outfile1D ("generator_4000.txt");
            {
                outfile1D << generator << endl;
            }
            outfile1D.close();
            
            ofstream outfile2D ("generator1_4000.txt");
            {
                outfile2D << generator1 << endl;
            }
            outfile2D.close();
        }
        
        if(t1 == 5000)
        {
            ofstream outfile1E ("generator_5000.txt");
            {
                outfile1E << generator << endl;
            }
            outfile1E.close();
            
            ofstream outfile2E ("generator1_5000.txt");
            {
                outfile2E << generator1 << endl;
            }
            outfile2E.close();
        }
        
        if(t1 == 6000)
        {
            ofstream outfile1F ("generator_6000.txt");
            {
                outfile1F << generator << endl;
            }
            outfile1F.close();
            
            ofstream outfile2F ("generator1_6000.txt");
            {
                outfile2F << generator1 << endl;
            }
            outfile2F.close();
        }
        
        if(t1 == 7000)
        {
            ofstream outfile1G ("generator_7000.txt");
            {
                outfile1G << generator << endl;
            }
            outfile1G.close();
            
            ofstream outfile2G ("generator1_7000.txt");
            {
                outfile2G << generator1 << endl;
            }
            outfile2G.close();
        }
        
        if(t1 == 8000)
        {
            ofstream outfile1H ("generator_8000.txt");
            {
                outfile1H << generator << endl;
            }
            outfile1H.close();
            
            ofstream outfile2H ("generator1_8000.txt");
            {
                outfile2H << generator1 << endl;
            }
            outfile2H.close();
        }
        
        if(t1 == 9000)
        {
            ofstream outfile1I ("generator_9000.txt");
            {
                outfile1I << generator << endl;
            }
            outfile1I.close();
            
            ofstream outfile2I ("generator1_9000.txt");
            {
                outfile2I << generator1 << endl;
            }
            outfile2I.close();
        }
        
        if(t1 == 10000)
        {
            ofstream outfile1J ("generator_10000.txt");
            {
                outfile1J << generator << endl;
            }
            outfile1J.close();
            
            ofstream outfile2J ("generator1_10000.txt");
            {
                outfile2J << generator1 << endl;
            }
            outfile2J.close();
        }
        
        if(t1 == 11000)
        {
            ofstream outfile1K ("generator_11000.txt");
            {
                outfile1K << generator << endl;
            }
            outfile1K.close();
            
            ofstream outfile2K ("generator1_11000.txt");
            {
                outfile2K << generator1 << endl;
            }
            outfile2K.close();
        }
        
        if(t1 == 12000)
        {
            ofstream outfile1L ("generator_12000.txt");
            {
                outfile1L << generator << endl;
            }
            outfile1L.close();
            
            ofstream outfile2L ("generator1_12000.txt");
            {
                outfile2L << generator1 << endl;
            }
            outfile2L.close();
        }
        
        if(t1 == 13000)
        {
            ofstream outfile1M ("generator_13000.txt");
            {
                outfile1M << generator << endl;
            }
            outfile1M.close();
            
            ofstream outfile2M ("generator1_13000.txt");
            {
                outfile2M << generator1 << endl;
            }
            outfile2M.close();
        }
        
        if(t1 == 14000)
        {
            ofstream outfile1N ("generator_14000.txt");
            {
                outfile1N << generator << endl;
            }
            outfile1N.close();
            
            ofstream outfile2N ("generator1_14000.txt");
            {
                outfile2N << generator1 << endl;
            }
            outfile2N.close();
        }
        
        if(t1 == 15000)
        {
            ofstream outfile1O ("generator_15000.txt");
            {
                outfile1O << generator << endl;
            }
            outfile1O.close();
            
            ofstream outfile2O ("generator1_15000.txt");
            {
                outfile2O << generator1 << endl;
            }
            outfile2O.close();
        }
        
        if(t1 == 16000)
        {
            ofstream outfile1P ("generator_16000.txt");
            {
                outfile1P << generator << endl;
            }
            outfile1P.close();
            
            ofstream outfile2P ("generator1_16000.txt");
            {
                outfile2P << generator1 << endl;
            }
            outfile2P.close();
        }
        
        if(t1 == 17000)
        {
            ofstream outfile1Q ("generator_17000.txt");
            {
                outfile1Q << generator << endl;
            }
            outfile1Q.close();
            
            ofstream outfile2Q ("generator1_17000.txt");
            {
                outfile2Q << generator1 << endl;
            }
            outfile2Q.close();
        }
        
        if(t1 == 18000)
        {
            ofstream outfile1R ("generator_18000.txt");
            {
                outfile1R << generator << endl;
            }
            outfile1R.close();
            
            ofstream outfile2R ("generator1_18000.txt");
            {
                outfile2R << generator1 << endl;
            }
            outfile2R.close();
        }
        
        if(t1 == 19000)
        {
            ofstream outfile1S ("generator_19000.txt");
            {
                outfile1S << generator << endl;
            }
            outfile1S.close();
            
            ofstream outfile2S ("generator1_19000.txt");
            {
                outfile2S << generator1 << endl;
            }
            outfile2S.close();
        }
        
//        Random number that follows the uniform distribution between 0 and 1
        U = distributionU(generator);
        
        cout<<"U = "<<U<<endl;
        cout<<"t1 = "<<t1<<endl;
        cout<<"eps0_X["<<t1<<"] = "<<eps0_X[t1]<<"     "<<"Q_X["<<t1<<"] = "<<Q_X[t1]<<"     "<<"logrhoc_X["<<t1<<"] = "<<logrhoc_X[t1]<<endl;
        
        rhoc_X[t1] = goncm3inMsunonkpc3*pow(10.0,logrhoc_X[t1]);
        
//        wsor parameters defined for the other 29 galaxies (in this example we consider a sample of 30 galaxies)
//        The Poisson solver convergence is guaranteeed if 0 < wsor < 2.
//        Successive Over Relaxation (SOR) Poisson solver (the one used in this code) is faster than Jacobi and Gauss-Seidel Poisson solvers if 1 < wsor < 2.
//        The fastest convergence is usually achieved for wsor = 2/(1 + π/min(NR, NZ)).
//        In this example the wsor parameters is set individually for each galaxy in the sample and for each galaxy assumes different values according to the value of the free parameter eps0, to guardantee the fastest convergence in every case
//        You can set it differently according to your specific case
        if (eps0_X[t1] >= 0.51)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.45 && eps0_X[t1] < 0.51)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.38 && eps0_X[t1] < 0.45)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.31 && eps0_X[t1] < 0.38)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.24 && eps0_X[t1] < 0.31)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.17 && eps0_X[t1] < 0.24)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.12 && eps0_X[t1] < 0.17)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/60.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.12)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[0] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.47)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.47)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.41)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.29 && eps0_X[t1] < 0.35)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.22 && eps0_X[t1] < 0.29)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.22)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.32 && eps0_X[t1] < 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.32)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.26)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.20)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/75.0);
        }
        if (eps0_X[t1] >= 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.32 && eps0_X[t1] < 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.32)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.26)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.20)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/75.0);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.53)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.47 && eps0_X[t1] < 0.53)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.40 && eps0_X[t1] < 0.47)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.40)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.33)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.26)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.12 && eps0_X[t1] < 0.19)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[3] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.49)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.42 && eps0_X[t1] < 0.49)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.36 && eps0_X[t1] < 0.42)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.29 && eps0_X[t1] < 0.36)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.22 && eps0_X[t1] < 0.29)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.22)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.12 && eps0_X[t1] < 0.15)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/65.0);
        }
        else
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/60.0);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.44)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.39 && eps0_X[t1] < 0.44)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.39)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.27 && eps0_X[t1] < 0.33)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.27)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.13 && eps0_X[t1] < 0.20)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/65.0);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.19)
        {
            wsor[6] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.19)
        {
            wsor[6] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else
        {
            wsor[6] = 2.0/(1.0 + CONST_PI/150.0);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.57)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.48 && eps0_X[t1] < 0.57)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.40 && eps0_X[t1] < 0.48)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.40)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.25 && eps0_X[t1] < 0.33)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.18 && eps0_X[t1] < 0.25)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.18)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[7] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.55)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.47 && eps0_X[t1] < 0.55)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.47)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.41)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.28 && eps0_X[t1] < 0.35)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.28)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.14 && eps0_X[t1] < 0.20)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[8] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.48)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.48)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.41)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.28 && eps0_X[t1] < 0.35)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.21 && eps0_X[t1] < 0.28)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.14 && eps0_X[t1] < 0.21)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.12 && eps0_X[t1] < 0.14)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/65.0);
        }
        else
        {
            wsor[9] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        
        if (eps0_X[t1] >= 0.47)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.42 && eps0_X[t1] < 0.47)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.38 && eps0_X[t1] < 0.42)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.32 && eps0_X[t1] < 0.38)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.32)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.26)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.12 && eps0_X[t1] < 0.19)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[10] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.50)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.44 && eps0_X[t1] < 0.50)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.39 && eps0_X[t1] < 0.44)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.39)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.33)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.26)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.19)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/60.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.15)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/45.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.44)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.39 && eps0_X[t1] < 0.44)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.39)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.27 && eps0_X[t1] < 0.33)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.27)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.13 && eps0_X[t1] < 0.20)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[12] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.44)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.39 && eps0_X[t1] < 0.44)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.39)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.27 && eps0_X[t1] < 0.33)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.27)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.17 && eps0_X[t1] < 0.20)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/65.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.48)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.48)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.34 && eps0_X[t1] < 0.41)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.27 && eps0_X[t1] < 0.34)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.27)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.20)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.12 && eps0_X[t1] < 0.15)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.12)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/45.0);
        }
        else
        {
            wsor[14] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.49)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.43 && eps0_X[t1] < 0.49)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.38 && eps0_X[t1] < 0.43)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.32 && eps0_X[t1] < 0.38)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.25 && eps0_X[t1] < 0.32)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.18 && eps0_X[t1] < 0.25)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.18)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[15] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.51)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.44 && eps0_X[t1] < 0.51)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.38 && eps0_X[t1] < 0.44)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.32 && eps0_X[t1] < 0.38)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.32)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.26)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.19)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[16] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.51)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.44 && eps0_X[t1] < 0.51)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.37 && eps0_X[t1] < 0.44)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.30 && eps0_X[t1] < 0.37)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.23 && eps0_X[t1] < 0.30)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.16 && eps0_X[t1] < 0.23)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.49)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.43 && eps0_X[t1] < 0.49)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.37 && eps0_X[t1] < 0.43)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.31 && eps0_X[t1] < 0.37)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.24 && eps0_X[t1] < 0.31)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.17 && eps0_X[t1] < 0.24)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.50)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.45 && eps0_X[t1] < 0.50)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.40 && eps0_X[t1] < 0.45)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.40)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.28 && eps0_X[t1] < 0.35)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.21 && eps0_X[t1] < 0.28)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.13 && eps0_X[t1] < 0.21)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/25.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.42)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.36 && eps0_X[t1] < 0.42)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.31 && eps0_X[t1] < 0.36)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.25 && eps0_X[t1] < 0.31)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.25)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.16 && eps0_X[t1] < 0.19)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[21] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.54)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.47 && eps0_X[t1] < 0.54)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.47)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.34 && eps0_X[t1] < 0.41)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.28 && eps0_X[t1] < 0.34)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.20 && eps0_X[t1] < 0.28)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.13 && eps0_X[t1] < 0.20)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[22] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_X[t1] >= 0.49)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.44 && eps0_X[t1] < 0.49)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.38 && eps0_X[t1] < 0.44)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.33 && eps0_X[t1] < 0.38)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.26 && eps0_X[t1] < 0.33)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.26)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.16 && eps0_X[t1] < 0.19)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/65.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.16)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/45.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.35)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.28 && eps0_X[t1] < 0.35)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.22 && eps0_X[t1] < 0.28)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.16 && eps0_X[t1] < 0.22)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.16)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/75.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.51)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.46 && eps0_X[t1] < 0.51)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.46)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.41)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.29 && eps0_X[t1] < 0.35)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.22 && eps0_X[t1] < 0.29)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.13 && eps0_X[t1] < 0.22)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/40.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.47)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.47)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.41)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.29 && eps0_X[t1] < 0.35)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.22 && eps0_X[t1] < 0.29)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.14 && eps0_X[t1] < 0.22)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.52)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.43 && eps0_X[t1] < 0.52)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.43)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.27 && eps0_X[t1] < 0.35)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.19 && eps0_X[t1] < 0.27)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.14 && eps0_X[t1] < 0.19)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/60.0);
        }
        else if (eps0_X[t1] >= 0.11 && eps0_X[t1] < 0.14)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/40.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.47)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.38 && eps0_X[t1] < 0.47)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.30 && eps0_X[t1] < 0.38)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.23 && eps0_X[t1] < 0.30)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.23)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/75.0);
        }
        
        
        
        if (eps0_X[t1] >= 0.47)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_X[t1] >= 0.41 && eps0_X[t1] < 0.47)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_X[t1] >= 0.35 && eps0_X[t1] < 0.41)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_X[t1] >= 0.30 && eps0_X[t1] < 0.35)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_X[t1] >= 0.23 && eps0_X[t1] < 0.30)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_X[t1] >= 0.16 && eps0_X[t1] < 0.23)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_X[t1] >= 0.15 && eps0_X[t1] < 0.16)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[29] = 2.0/(1.0 + CONST_PI);
        }
        
        
//        Setting the number of threads used for parallel computation
        omp_set_dynamic(0);
        omp_set_num_threads(Nthreads);
        
//        OpenMP directive defined for the parallelized for loop. "shared" meand that the quantity "chi2red_X" is shared in every iteration of the for loop and "reduction(+: chi2redtot_X)" means that "chi2redtot_X" is the quantiy involved in the reduce operation which in this case is a sum (+).
//        The parallelized for loop iterates on the number of galaxies in the considered sample since the same operations are performed for every galaxy and every galaxy is completely independent from each other.
        #pragma omp parallel for shared(chi2red_X) reduction(+: chi2redtot_X)
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j <= params[i].NR; j++)
            {
                for(int k = 0; k <= params[i].NZ; k++)
                {
                    eps[i][j][k] = eps0_X[t1]+(1.0 - eps0_X[t1])*0.5*(1.0 + ((-1.0 + pow(rho[i][j][k]/(rhoc_X[t1]/rhounit[i]),2.0*Q_X[t1]))/(1.0 + pow(rho[i][j][k]/(rhoc_X[t1]/rhounit[i]),2.0*Q_X[t1]))));
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int k = 1; k < params[i].NZ; k++)
                {
                    DReps[i][j][k] = (eps[i][j+1][k] - eps[i][j-1][k])/(2.0*dR[i]);
                    Dzeps[i][j][k] = (eps[i][j][k+1] - eps[i][j][k-1])/(2.0*dz[i]);
                }
            }
            
//            Successive Over Relaxation (SOR) Poisson solver, run to find the gravitational potential
            while (err[i] > tol){
                for(int j = 0; j <= params[i].NR; j++)
                {
                    for(int k = 0; k <= params[i].NZ; k++)
                    {
                        phi1[i][j][k] = phi0[i][j][k];
                    }
                }
                for(int j = 2; j < params[i].NR - 1; j++)
                {
                    for(int k = 2; k < params[i].NZ - 1; k++)
                    {
                        S[i][j][k] = -((4.0*CONST_PI*G*rho[i][j][k]-(DReps[i][j][k]*((phi0[i][j+1][k] - phi0[i][j-1][k])/(2.0*dR[i])) + Dzeps[i][j][k]*((phi0[i][j][k+1] - phi0[i][j][k-1])/(2.0*dz[i]))))/eps[i][j][k]);
                        phi1[i][j][k] = phi0[i][j][k]*(1.0 - wsor[i]) + wsor[i]/(2.0*(dz2[i] + dR2[i])*R[i][j])*(dR2[i]*dz2[i]*R[i][j]*S[i][j][k] + phi1[i][j+1][k]*(R[i][j] + 0.5*dR[i])*dz2[i] + phi1[i][j-1][k]*(R[i][j] - 0.5*dR[i])*dz2[i] + (phi1[i][j][k+1] + phi1[i][j][k-1])*dR2[i]*R[i][j]);
                    }
                }
                
                // Compute error
                
                err[i] = Error(phi0[i], phi1[i], params[i].NR, params[i].NZ);
                
                for(int j = 2; j < params[i].NR - 1; j++)
                {
                    for(int k = 2; k < params[i].NZ - 1; k++)
                    {
                        phi0[i][j][k] = phi1[i][j][k];
                    }
                }
                n[i]++;
                // Print the error
//                cout<<i<<"     "<<"err["<<i<<"] = "<< setprecision(8) << err[i] <<"     "<<"n["<<i<<"] = "<<n[i]<<endl;
                if(n[i] == NitersSORbreak[i])
                {
                    break;
                }
            }
            
//            Radial and vertical derivatives of the gravitational potential (= radial and vertical components of the gravitational field)
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int k = 1; k < params[i].NZ; k++)
                {
                    DRphi1[i][j][k] = (phi1[i][j+1][k] - phi1[i][j-1][k])/(2.0*dR[i]);
                    Dzphi1[i][j][k] = (phi1[i][j][k+1] - phi1[i][j][k-1])/(2.0*dz[i]);
                }
            }
            
            
            
            for(int j = 0; j <= params[i].NR; j++)
            {
                for(int k = 0; k <= params[i].NZ; k++)
                {
                    integrandf1[i][j][k] = Dzphi1[i][j][k]*exp(-abs(z[i][k])/params[i].hzMCMC);
                }
            }
            
            for(int t = params[i].NZ/2; t < params[i].NZ; t++)
            {
                for(int j = 1; j < params[i].NR; j++)
                {
                    for(int k = t; k < params[i].NZ; k++)
                    {
                        Atrap1[i][j][k] = (integrandf1[i][j][k+1] + integrandf1[i][j][k])*dz[i]/2.0;
                        integralf12D[i][j][t] = integralf12D[i][j][t] + Atrap1[i][j][k];
                    }
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int t = 1; t < params[i].NZ/2; t++)
                {
                    integralf12D[i][j][t] = integralf12D[i][j][params[i].NZ - t];
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int k = params[i].NZ/2; k < params[i].NZ; k++)
                {
                    Atrap2[i][j][k] = (integralf12D[i][j][k+1] + integralf12D[i][j][k])*dz[i]/2.0;
                    sigz[i][j] = sigz[i][j] + Atrap2[i][j][k];
                }
            }
            
//            Model vertical velocity dispersion profile
            for(int j = 1; j < params[i].NR; j++) sigz[i][j] = sqrt(sigz[i][j]/params[i].hzMCMC);
            
//            Model vertical velocity dispersion profile defined in the right part of the radial grid
            for(int j = 0; j < (params[i].NR - 1)/2+1; j++) sigzbar[i][j] = sigz[i][j + (params[i].NR - 1)/2 + 1];
            
//            Model rotation curve defined in the right part of the radial grid and in the plane z = 0
            for(int j = 0; j < (params[i].NR - 1)/2 + 1; j++) vbar[i][j] = sqrt(Rhalf[i][j]*DRphi1[i][j + (params[i].NR - 1)/2 + 1][params[i].NZ/2]);
            
            for(int j = 0; j < params[i].Nptv; j++)
            {
//                Interpolation of the model rotation curve in the radii of the measured rotation curve
                vbarinterp_X[i][j] = gsl_interp_eval (interp1[i], Rhalf[i], vbar[i], rvvalue[i][j], acc);
                argchi2v_X[i][j] = (vvalue[i][j] - vbarinterp_X[i][j])*(vvalue[i][j] - vbarinterp_X[i][j])/(devstvvalue[i][j]*devstvvalue[i][j]);
//                chi square of the rotation curve for each galaxy in the sample
                chi2v_X[i] = chi2v_X[i] + argchi2v_X[i][j];
            }
            
            for(int j = 0; j < params[i].Nptsigz; j++)
            {
//                Interpolation of the model vertical velocity dispersion profile in the radii of the measured vertical velocity dispersion profile
                sigzbarinterp_X[i][j] = gsl_interp_eval (interp2[i], Rhalf[i], sigzbar[i], rsigzvalue[i][j], acc);
                argchi2sigz_X[i][j] = (sigzvalue[i][j] - sigzbarinterp_X[i][j])*(sigzvalue[i][j] - sigzbarinterp_X[i][j])/(devstsigzvalue[i][j]*devstsigzvalue[i][j]);
//                chi square of the vertical velocity dispersion profile for each galaxy in the sample
                chi2sigz_X[i] = chi2sigz_X[i] + argchi2sigz_X[i][j];
            }
            
//            Total reduced chi square for every galaxy in the sample
            chi2red_X[i] = (chi2v_X[i] + chi2sigz_X[i])/double(Ndoftot[i]);
            
            
            
//            Reduce operation (sum):
//            Total reduced chi square, obtained by summing the total reduced chi squares for every galaxy in the sample
            chi2redtot_X = chi2redtot_X + chi2red_X[i];
            
            
            
//            Reinitialization of the various quantities
            
            n[i] = 0;
            
            for (int j = 2; j < params[i].NR - 1; j++)
            {
                for (int k = 2; k < params[i].NZ - 1; k++)
                {
                    phi0[i][j][k] = 0.0;
                }
            }
            
            for (int j = 1; j < params[i].NR; j++)
            {
                for (int k = 1; k < params[i].NZ; k++)
                {
                    integralf12D[i][j][k] = 0.0;
                }
            }
            
            for (int j = 1; j < params[i].NR; j++) sigz[i][j] = 0.0;
            
            err[i] = 1.e30;
        }
        
        cout<<endl;
        
        cout<<"chi2redtot_X = "<<chi2redtot_X<<endl;
        
        cout<<endl;
        
//        Next combination of free parameters (_Y) generated from the previous one (_X) with normal distributions peaked at the (_X) parameters and with standard deviations equal to "stepsizeNAMEOFTHEPARAMETER" that in this example are qual to the 10% of the prior ranges defined for the free parameters
        
        eps0_Y = eps0_X[t1] + stepsizeeps0*UnitGaussian(generator1);
        Q_Y = Q_X[t1] + stepsizeQ*UnitGaussian(generator1);
        logrhoc_Y = logrhoc_X[t1] + stepsizelogrhoc*UnitGaussian(generator1);
        
//        The quantities defined in the second part of the MCMC are the same as in the first part (see the comments above)
        
        cout<<"eps0_Y = "<<eps0_Y<<"     "<<"Q_Y = "<<Q_Y<<"     "<<"logrhoc_Y = "<<logrhoc_Y<<endl;
        
        rhoc_Y = goncm3inMsunonkpc3*pow(10.0,logrhoc_Y);
        
        if (eps0_Y >= 0.51)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.45 && eps0_Y < 0.51)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.38 && eps0_Y < 0.45)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.31 && eps0_Y < 0.38)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.24 && eps0_Y < 0.31)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.17 && eps0_Y < 0.24)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.12 && eps0_Y < 0.17)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/60.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.12)
        {
            wsor[0] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[0] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.47)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.47)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.41)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.29 && eps0_Y < 0.35)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.22 && eps0_Y < 0.29)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.22)
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[1] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_Y >= 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.32 && eps0_Y < 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.32)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.26)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.20)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/75.0);
        }
        if (eps0_Y >= 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.32 && eps0_Y < 0.38)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.32)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.26)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.20)
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[2] = 2.0/(1.0 + CONST_PI/75.0);
        }
        
        
        
        
        if (eps0_Y >= 0.53)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.47 && eps0_Y < 0.53)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.40 && eps0_Y < 0.47)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.40)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.33)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.26)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.12 && eps0_Y < 0.19)
        {
            wsor[3] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[3] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.49)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.42 && eps0_Y < 0.49)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.36 && eps0_Y < 0.42)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.29 && eps0_Y < 0.36)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.22 && eps0_Y < 0.29)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.22)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.12 && eps0_Y < 0.15)
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/65.0);
        }
        else
        {
            wsor[4] = 2.0/(1.0 + CONST_PI/60.0);
        }
        
        
        
        
        if (eps0_Y >= 0.44)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.39 && eps0_Y < 0.44)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.39)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.27 && eps0_Y < 0.33)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.27)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.13 && eps0_Y < 0.20)
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[5] = 2.0/(1.0 + CONST_PI/65.0);
        }
        
        
        
        
        if (eps0_Y >= 0.19)
        {
            wsor[6] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.19)
        {
            wsor[6] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else
        {
            wsor[6] = 2.0/(1.0 + CONST_PI/150.0);
        }
        
        
        
        
        if (eps0_Y >= 0.57)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.48 && eps0_Y < 0.57)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.40 && eps0_Y < 0.48)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.40)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.25 && eps0_Y < 0.33)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.18 && eps0_Y < 0.25)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.18)
        {
            wsor[7] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[7] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        
        if (eps0_Y >= 0.55)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.47 && eps0_Y < 0.55)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.47)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.41)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.28 && eps0_Y < 0.35)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.28)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.14 && eps0_Y < 0.20)
        {
            wsor[8] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[8] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        
        if (eps0_Y >= 0.48)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.48)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.41)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.28 && eps0_Y < 0.35)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.21 && eps0_Y < 0.28)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.14 && eps0_Y < 0.21)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.12 && eps0_Y < 0.14)
        {
            wsor[9] = 2.0/(1.0 + CONST_PI/65.0);
        }
        else
        {
            wsor[9] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        
        if (eps0_Y >= 0.47)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.42 && eps0_Y < 0.47)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.38 && eps0_Y < 0.42)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.32 && eps0_Y < 0.38)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.32)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.26)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.12 && eps0_Y < 0.19)
        {
            wsor[10] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[10] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.50)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.44 && eps0_Y < 0.50)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.39 && eps0_Y < 0.44)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.39)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.33)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.26)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.19)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/60.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.15)
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[11] = 2.0/(1.0 + CONST_PI/45.0);
        }
        
        
        
        if (eps0_Y >= 0.44)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.39 && eps0_Y < 0.44)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.39)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.27 && eps0_Y < 0.33)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.27)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.13 && eps0_Y < 0.20)
        {
            wsor[12] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[12] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.44)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.39 && eps0_Y < 0.44)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.39)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.27 && eps0_Y < 0.33)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.27)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.17 && eps0_Y < 0.20)
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[13] = 2.0/(1.0 + CONST_PI/65.0);
        }
        
        
        
        if (eps0_Y >= 0.48)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.48)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.34 && eps0_Y < 0.41)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.27 && eps0_Y < 0.34)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.27)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.20)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.12 && eps0_Y < 0.15)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.12)
        {
            wsor[14] = 2.0/(1.0 + CONST_PI/45.0);
        }
        else
        {
            wsor[14] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.49)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.43 && eps0_Y < 0.49)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.38 && eps0_Y < 0.43)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.32 && eps0_Y < 0.38)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.25 && eps0_Y < 0.32)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.18 && eps0_Y < 0.25)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.18)
        {
            wsor[15] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[15] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.51)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.44 && eps0_Y < 0.51)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.38 && eps0_Y < 0.44)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.32 && eps0_Y < 0.38)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.32)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.26)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.19)
        {
            wsor[16] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[16] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.51)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.44 && eps0_Y < 0.51)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.37 && eps0_Y < 0.44)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.30 && eps0_Y < 0.37)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.23 && eps0_Y < 0.30)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.16 && eps0_Y < 0.23)
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[18] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_Y >= 0.49)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.43 && eps0_Y < 0.49)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.37 && eps0_Y < 0.43)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.31 && eps0_Y < 0.37)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.24 && eps0_Y < 0.31)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.17 && eps0_Y < 0.24)
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[19] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_Y >= 0.50)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.45 && eps0_Y < 0.50)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.40 && eps0_Y < 0.45)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.40)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.28 && eps0_Y < 0.35)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.21 && eps0_Y < 0.28)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.13 && eps0_Y < 0.21)
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[20] = 2.0/(1.0 + CONST_PI/25.0);
        }
        
        
        
        if (eps0_Y >= 0.42)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.36 && eps0_Y < 0.42)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.31 && eps0_Y < 0.36)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.25 && eps0_Y < 0.31)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.25)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.16 && eps0_Y < 0.19)
        {
            wsor[21] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[21] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.54)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.47 && eps0_Y < 0.54)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.47)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.34 && eps0_Y < 0.41)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.28 && eps0_Y < 0.34)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.20 && eps0_Y < 0.28)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.13 && eps0_Y < 0.20)
        {
            wsor[22] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[22] = 2.0/(1.0 + CONST_PI);
        }
        
        
        
        if (eps0_Y >= 0.49)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.44 && eps0_Y < 0.49)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.38 && eps0_Y < 0.44)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.33 && eps0_Y < 0.38)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.26 && eps0_Y < 0.33)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.26)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.16 && eps0_Y < 0.19)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/65.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.16)
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[23] = 2.0/(1.0 + CONST_PI/45.0);
        }
        
        
        
        if (eps0_Y >= 0.35)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.28 && eps0_Y < 0.35)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.22 && eps0_Y < 0.28)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.16 && eps0_Y < 0.22)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.16)
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[24] = 2.0/(1.0 + CONST_PI/75.0);
        }
        
        
        
        if (eps0_Y >= 0.51)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.46 && eps0_Y < 0.51)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.46)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.41)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.29 && eps0_Y < 0.35)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.22 && eps0_Y < 0.29)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.13 && eps0_Y < 0.22)
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[25] = 2.0/(1.0 + CONST_PI/40.0);
        }
        
        
        
        if (eps0_Y >= 0.47)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.47)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.41)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.29 && eps0_Y < 0.35)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.22 && eps0_Y < 0.29)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.14 && eps0_Y < 0.22)
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else
        {
            wsor[26] = 2.0/(1.0 + CONST_PI/50.0);
        }
        
        
        
        if (eps0_Y >= 0.52)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.43 && eps0_Y < 0.52)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.43)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.27 && eps0_Y < 0.35)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.19 && eps0_Y < 0.27)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.14 && eps0_Y < 0.19)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/60.0);
        }
        else if (eps0_Y >= 0.11 && eps0_Y < 0.14)
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[27] = 2.0/(1.0 + CONST_PI/40.0);
        }
        
        
        
        if (eps0_Y >= 0.47)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.38 && eps0_Y < 0.47)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.30 && eps0_Y < 0.38)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.23 && eps0_Y < 0.30)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.23)
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else
        {
            wsor[28] = 2.0/(1.0 + CONST_PI/75.0);
        }
        
        
        
        if (eps0_Y >= 0.47)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/200.0);
        }
        else if (eps0_Y >= 0.41 && eps0_Y < 0.47)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/175.0);
        }
        else if (eps0_Y >= 0.35 && eps0_Y < 0.41)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/150.0);
        }
        else if (eps0_Y >= 0.30 && eps0_Y < 0.35)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/125.0);
        }
        else if (eps0_Y >= 0.23 && eps0_Y < 0.30)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/100.0);
        }
        else if (eps0_Y >= 0.16 && eps0_Y < 0.23)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/75.0);
        }
        else if (eps0_Y >= 0.15 && eps0_Y < 0.16)
        {
            wsor[29] = 2.0/(1.0 + CONST_PI/50.0);
        }
        else
        {
            wsor[29] = 2.0/(1.0 + CONST_PI);
        }
        
        
        omp_set_dynamic(0);
        omp_set_num_threads(Nthreads);
        
//        #pragma omp parallel for shared(chi2redtot_Y, chi2red_Y) reduction(+: chi2redtot_Y)
        #pragma omp parallel for shared(chi2red_Y) reduction(+: chi2redtot_Y)
        for(int i = 0; i < NGal; i++)
        {
            for(int j = 0; j <= params[i].NR; j++)
            {
                for(int k = 0; k <= params[i].NZ; k++)
                {
                    eps[i][j][k] = eps0_Y+(1.0 - eps0_Y)*0.5*(1.0 + ((-1.0 + pow(rho[i][j][k]/(rhoc_Y/rhounit[i]),2.0*Q_Y))/(1.0 + pow(rho[i][j][k]/(rhoc_Y/rhounit[i]),2.0*Q_Y))));
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int k = 1; k < params[i].NZ; k++)
                {
                    DReps[i][j][k] = (eps[i][j+1][k] - eps[i][j-1][k])/(2.0*dR[i]);
                    Dzeps[i][j][k] = (eps[i][j][k+1] - eps[i][j][k-1])/(2.0*dz[i]);
                }
            }
            
            if (eps0_Y < eps0beg || eps0_Y > eps0end)
            {
                NitersSORbreak[i] = 2;
            }
            else if (Q_Y < Qbeg || Q_Y > Qend)
            {
                NitersSORbreak[i] = 2;
            }
            else if (logrhoc_Y < logrhocbeg || logrhoc_Y > logrhocend)
            {
                NitersSORbreak[i] = 2;
            }
            else
            {
                NitersSORbreak[i] = params[i].NitersSORbreak;
            }
            
            
            
            while (err[i] > tol){
                for(int j = 0; j <= params[i].NR; j++)
                {
                    for(int k = 0; k <= params[i].NZ; k++)
                    {
                        phi1[i][j][k] = phi0[i][j][k];
                    }
                }
                for(int j = 2; j < params[i].NR - 1; j++)
                {
                    for(int k = 2; k < params[i].NZ - 1; k++)
                    {
                        S[i][j][k] = -((4.0*CONST_PI*G*rho[i][j][k]-(DReps[i][j][k]*((phi0[i][j+1][k] - phi0[i][j-1][k])/(2.0*dR[i])) + Dzeps[i][j][k]*((phi0[i][j][k+1] - phi0[i][j][k-1])/(2.0*dz[i]))))/eps[i][j][k]);
                        phi1[i][j][k] = phi0[i][j][k]*(1.0 - wsor[i]) + wsor[i]/(2.0*(dz2[i] + dR2[i])*R[i][j])*(dR2[i]*dz2[i]*R[i][j]*S[i][j][k] + phi1[i][j+1][k]*(R[i][j] + 0.5*dR[i])*dz2[i] + phi1[i][j-1][k]*(R[i][j] - 0.5*dR[i])*dz2[i] + (phi1[i][j][k+1] + phi1[i][j][k-1])*dR2[i]*R[i][j]);
                    }
                }
                
                // Compute error
                
                err[i] = Error(phi0[i], phi1[i], params[i].NR, params[i].NZ);
                
                for(int j = 2; j < params[i].NR - 1; j++)
                {
                    for(int k = 2; k < params[i].NZ - 1; k++)
                    {
                        phi0[i][j][k] = phi1[i][j][k];
                    }
                }
                n[i]++;
                // Print the error
//                cout<<i<<"     "<<"err["<<i<<"] = "<< setprecision(8) << err[i] <<"     "<<"n["<<i<<"] = "<<n[i]<<endl;
                if(n[i] == NitersSORbreak[i])
                {
                    break;
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int k = 1; k < params[i].NZ; k++)
                {
                    DRphi1[i][j][k] = (phi1[i][j+1][k] - phi1[i][j-1][k])/(2.0*dR[i]);
                    Dzphi1[i][j][k] = (phi1[i][j][k+1] - phi1[i][j][k-1])/(2.0*dz[i]);
                }
            }
            
            for(int j = 0; j <= params[i].NR; j++)
            {
                for(int k = 0; k <= params[i].NZ; k++)
                {
                    integrandf1[i][j][k] = Dzphi1[i][j][k]*exp(-abs(z[i][k])/params[i].hzMCMC);
                }
            }
            
            for(int t = params[i].NZ/2; t < params[i].NZ; t++)
            {
                for(int j = 1; j < params[i].NR; j++)
                {
                    for(int k = t; k < params[i].NZ; k++)
                    {
                        Atrap1[i][j][k] = (integrandf1[i][j][k+1] + integrandf1[i][j][k])*dz[i]/2.0;
                        integralf12D[i][j][t] = integralf12D[i][j][t] + Atrap1[i][j][k];
                    }
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int t = 1; t < params[i].NZ/2; t++)
                {
                    integralf12D[i][j][t] = integralf12D[i][j][params[i].NZ - t];
                }
            }
            
            for(int j = 1; j < params[i].NR; j++)
            {
                for(int k = params[i].NZ/2; k < params[i].NZ; k++)
                {
                    Atrap2[i][j][k] = (integralf12D[i][j][k+1] + integralf12D[i][j][k])*dz[i]/2.0;
                    sigz[i][j] = sigz[i][j] + Atrap2[i][j][k];
                }
            }
            
            for(int j = 1; j < params[i].NR; j++) sigz[i][j] = sqrt(sigz[i][j]/params[i].hzMCMC);
            
            for(int j = 0; j < (params[i].NR - 1)/2+1; j++) sigzbar[i][j] = sigz[i][j + (params[i].NR - 1)/2 + 1];
            
            for(int j = 0; j < (params[i].NR - 1)/2 + 1; j++) vbar[i][j] = sqrt(Rhalf[i][j]*DRphi1[i][j + (params[i].NR - 1)/2 + 1][params[i].NZ/2]);
            
            for(int j = 0; j < params[i].Nptv; j++)
            {
                vbarinterp_Y[i][j] = gsl_interp_eval (interp1[i], Rhalf[i], vbar[i], rvvalue[i][j], acc);
                argchi2v_Y[i][j] = (vvalue[i][j] - vbarinterp_Y[i][j])*(vvalue[i][j] - vbarinterp_Y[i][j])/(devstvvalue[i][j]*devstvvalue[i][j]);
                chi2v_Y[i] = chi2v_Y[i] + argchi2v_Y[i][j];
            }
            
            for(int j = 0; j < params[i].Nptsigz; j++)
            {
                sigzbarinterp_Y[i][j] = gsl_interp_eval (interp2[i], Rhalf[i], sigzbar[i], rsigzvalue[i][j], acc);
                argchi2sigz_Y[i][j] = (sigzvalue[i][j] - sigzbarinterp_Y[i][j])*(sigzvalue[i][j] - sigzbarinterp_Y[i][j])/(devstsigzvalue[i][j]*devstsigzvalue[i][j]);
                chi2sigz_Y[i] = chi2sigz_Y[i] + argchi2sigz_Y[i][j];
            }
            
            chi2red_Y[i] = (chi2v_Y[i] + chi2sigz_Y[i])/double(Ndoftot[i]);
            
            
            
//            Reduce (somma)
            
            chi2redtot_Y = chi2redtot_Y + chi2red_Y[i];
            
            
            
//            Reinitialization of the various quantities
            
            n[i] = 0;
            
            for (int j = 2; j < params[i].NR - 1; j++)
            {
                for (int k = 2; k < params[i].NZ - 1; k++)
                {
                    phi0[i][j][k] = 0.0;
                }
            }
            
            for (int j = 1; j < params[i].NR; j++)
            {
                for (int k = 1; k < params[i].NZ; k++)
                {
                    integralf12D[i][j][k] = 0.0;
                }
            }
            
            for (int j = 1; j < params[i].NR; j++) sigz[i][j] = 0.0;
            
            err[i] = 1.e30;
            
            NitersSORbreak[i] = params[i].NitersSORbreak;
        }
        
        cout<<endl;
        
        cout<<"chi2redtot_Y = "<<chi2redtot_Y<<endl;
        
        cout<<endl;

        
        
//        Metropolis-Hastings acceptance criterion
        
//        The combination of free parameter generated from the previous one in the second part of the MCMC is rejected if the free parameters exceed the prior ranges limits
        if (eps0_Y < eps0beg || eps0_Y > eps0end)
        {
            eps0_X[t1+1] = eps0_X[t1];
            Q_X[t1+1] = Q_X[t1];
            logrhoc_X[t1+1] = logrhoc_X[t1];
        }
        else if (Q_Y < Qbeg || Q_Y > Qend)
        {
            eps0_X[t1+1] = eps0_X[t1];
            Q_X[t1+1] = Q_X[t1];
            logrhoc_X[t1+1] = logrhoc_X[t1];
        }
        else if (logrhoc_Y < logrhocbeg || logrhoc_Y > logrhocend)
        {
            eps0_X[t1+1] = eps0_X[t1];
            Q_X[t1+1] = Q_X[t1];
            logrhoc_X[t1+1] = logrhoc_X[t1];
        }
        else
        {
//            Natural logarithm of Metropolis-Hastings ratio
            lnrHastings = -chi2redtot_Y + chi2redtot_X;
            
            if (U == 0.0) {
                eps0_X[t1+1] = eps0_Y;
                Q_X[t1+1] = Q_Y;
                logrhoc_X[t1+1] = logrhoc_Y;
            } else {
                if (lnrHastings >= 2.0*log(U)) {
                    eps0_X[t1+1] = eps0_Y;
                    Q_X[t1+1] = Q_Y;
                    logrhoc_X[t1+1] = logrhoc_Y;
                } else {
                    eps0_X[t1+1] = eps0_X[t1];
                    Q_X[t1+1] = Q_X[t1];
                    logrhoc_X[t1+1] = logrhoc_X[t1];
                }
            }
        }
        cout<<"lnrHastings = "<<lnrHastings<<endl;
        cout<<"2.0*log(U) = "<<2.0*log(U)<<endl;
        
        for(int i = 0; i < NGal; i++)
        {
            chi2v_X[i] = 0.0;
            chi2sigz_X[i] = 0.0;
            chi2red_X[i] = 0.0;
            
            chi2v_Y[i] = 0.0;
            chi2sigz_Y[i] = 0.0;
            chi2red_Y[i] = 0.0;
        }
        
        chi2redtot_X = 0.0;
        chi2redtot_Y = 0.0;
        
        
        
//        DECOMMENT THE BELOW PART OF CODE IF YOU WANT TO MEASURE THE TIME FOR EVERY MCMC ITERATION
//        gettimeofday(&end[t1], NULL);
//        
//        time_taken[t1] = (end[t1].tv_sec - start[t1].tv_sec) * 1e6;
//        time_taken[t1] = (time_taken[t1] + (end[t1].tv_usec -
//                                            start[t1].tv_usec)) * 1e-6;
//        
//        ofstream outfile16 ("Time_taken_for_every_iteration.txt");
//        {
//            for(int t1 = 0; t1 < Niters; t1++)
//            {
//                outfile16 << time_taken[t1] << setprecision(6) << endl;
//            }
//        }
//        outfile16.close();
    
        
//        Print to external files the chains of the free parameters of the model every 1000 MCMC iterations: it is useful if the code is killed and you do not want to restart the MCMC from the beginning and if you want to monitor the chains convergence.
//        The below example is done for a total (burn-in + effective) number of MCMC iterations equal to 20000.
        if(t1 == 1000)
        {
            ofstream outfile3a ("eps0_X_1000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 1000; i++) outfile3a<<eps0_X[i]<<endl;
            }
            outfile3a.close();
            
            ofstream outfile4a ("Q_X_1000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 1000; i++) outfile4a<<Q_X[i]<<endl;
            }
            outfile4a.close();
            
            ofstream outfile5a ("logrhoc_X_1000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 1000; i++) outfile5a<<logrhoc_X[i]<<endl;
            }
            outfile5a.close();
        }
        
        if(t1 == 2000)
        {
            ofstream outfile3b ("eps0_X_2000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 2000; i++) outfile3b<<eps0_X[i]<<endl;
            }
            outfile3b.close();
            
            ofstream outfile4b ("Q_X_2000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 2000; i++) outfile4b<<Q_X[i]<<endl;
            }
            outfile4b.close();
            
            ofstream outfile5b ("logrhoc_X_2000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 2000; i++) outfile5b<<logrhoc_X[i]<<endl;
            }
            outfile5b.close();
        }
        
        if(t1 == 3000)
        {
            ofstream outfile3c ("eps0_X_3000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 3000; i++) outfile3c<<eps0_X[i]<<endl;
            }
            outfile3c.close();
            
            ofstream outfile4c ("Q_X_3000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 3000; i++) outfile4c<<Q_X[i]<<endl;
            }
            outfile4c.close();
            
            ofstream outfile5c ("logrhoc_X_3000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 3000; i++) outfile5c<<logrhoc_X[i]<<endl;
            }
            outfile5c.close();
        }
        
        if(t1 == 4000)
        {
            ofstream outfile3d ("eps0_X_4000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 4000; i++) outfile3d<<eps0_X[i]<<endl;
            }
            outfile3d.close();
            
            ofstream outfile4d ("Q_X_4000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 4000; i++) outfile4d<<Q_X[i]<<endl;
            }
            outfile4d.close();
            
            ofstream outfile5d ("logrhoc_X_4000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 4000; i++) outfile5d<<logrhoc_X[i]<<endl;
            }
            outfile5d.close();
        }
        
        if(t1 == 5000)
        {
            ofstream outfile3e ("eps0_X_5000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 5000; i++) outfile3e<<eps0_X[i]<<endl;
            }
            outfile3e.close();
            
            ofstream outfile4e ("Q_X_5000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 5000; i++) outfile4e<<Q_X[i]<<endl;
            }
            outfile4e.close();
            
            ofstream outfile5e ("logrhoc_X_5000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 5000; i++) outfile5e<<logrhoc_X[i]<<endl;
            }
            outfile5e.close();
        }
        
        if(t1 == 6000)
        {
            ofstream outfile3f ("eps0_X_6000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 6000; i++) outfile3f<<eps0_X[i]<<endl;
            }
            outfile3f.close();
            
            ofstream outfile4f ("Q_X_6000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 6000; i++) outfile4f<<Q_X[i]<<endl;
            }
            outfile4f.close();
            
            ofstream outfile5f ("logrhoc_X_6000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 6000; i++) outfile5f<<logrhoc_X[i]<<endl;
            }
            outfile5f.close();
        }
        
        if(t1 == 7000)
        {
            ofstream outfile3g ("eps0_X_7000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 7000; i++) outfile3g<<eps0_X[i]<<endl;
            }
            outfile3g.close();
            
            ofstream outfile4g ("Q_X_7000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 7000; i++) outfile4g<<Q_X[i]<<endl;
            }
            outfile4g.close();
            
            ofstream outfile5g ("logrhoc_X_7000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 7000; i++) outfile5g<<logrhoc_X[i]<<endl;
            }
            outfile5g.close();
        }
        
        if(t1 == 8000)
        {
            ofstream outfile3h ("eps0_X_8000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 8000; i++) outfile3h<<eps0_X[i]<<endl;
            }
            outfile3h.close();
            
            ofstream outfile4h ("Q_X_8000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 8000; i++) outfile4h<<Q_X[i]<<endl;
            }
            outfile4h.close();
            
            ofstream outfile5h ("logrhoc_X_8000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 8000; i++) outfile5h<<logrhoc_X[i]<<endl;
            }
            outfile5h.close();
        }
        if(t1 == 9000)
        {
            ofstream outfile3i ("eps0_X_9000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 9000; i++) outfile3i<<eps0_X[i]<<endl;
            }
            outfile3i.close();
            
            ofstream outfile4i ("Q_X_9000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 9000; i++) outfile4i<<Q_X[i]<<endl;
            }
            outfile4i.close();
            
            ofstream outfile5i ("logrhoc_X_9000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 9000; i++) outfile5i<<logrhoc_X[i]<<endl;
            }
            outfile5i.close();
        }
        if(t1 == 10000)
        {
            ofstream outfile3j ("eps0_X_10000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 10000; i++) outfile3j<<eps0_X[i]<<endl;
            }
            outfile3j.close();
            
            ofstream outfile4j ("Q_X_10000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 10000; i++) outfile4j<<Q_X[i]<<endl;
            }
            outfile4j.close();
            
            ofstream outfile5j ("logrhoc_X_10000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 10000; i++) outfile5j<<logrhoc_X[i]<<endl;
            }
            outfile5j.close();
        }
        if(t1 == 11000)
        {
            ofstream outfile3k ("eps0_X_11000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 11000; i++) outfile3k<<eps0_X[i]<<endl;
            }
            outfile3k.close();
            
            ofstream outfile4k ("Q_X_11000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 11000; i++) outfile4k<<Q_X[i]<<endl;
            }
            outfile4k.close();
            
            ofstream outfile5k ("logrhoc_X_11000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 11000; i++) outfile5k<<logrhoc_X[i]<<endl;
            }
            outfile5k.close();
        }
        if(t1 == 12000)
        {
            ofstream outfile3l ("eps0_X_12000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 12000; i++) outfile3l<<eps0_X[i]<<endl;
            }
            outfile3l.close();
            
            ofstream outfile4l ("Q_X_12000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 12000; i++) outfile4l<<Q_X[i]<<endl;
            }
            outfile4l.close();
            
            ofstream outfile5l ("logrhoc_X_12000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 12000; i++) outfile5l<<logrhoc_X[i]<<endl;
            }
            outfile5l.close();
        }
        if(t1 == 13000)
        {
            ofstream outfile3m ("eps0_X_13000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 13000; i++) outfile3m<<eps0_X[i]<<endl;
            }
            outfile3m.close();
            
            ofstream outfile4m ("Q_X_13000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 13000; i++) outfile4m<<Q_X[i]<<endl;
            }
            outfile4m.close();
            
            ofstream outfile5m ("logrhoc_X_13000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 13000; i++) outfile5m<<logrhoc_X[i]<<endl;
            }
            outfile5m.close();
        }
        if(t1 == 14000)
        {
            ofstream outfile3n ("eps0_X_14000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 14000; i++) outfile3n<<eps0_X[i]<<endl;
            }
            outfile3n.close();
            
            ofstream outfile4n ("Q_X_14000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 14000; i++) outfile4n<<Q_X[i]<<endl;
            }
            outfile4n.close();
            
            ofstream outfile5n ("logrhoc_X_14000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 14000; i++) outfile5n<<logrhoc_X[i]<<endl;
            }
            outfile5n.close();
        }
        if(t1 == 15000)
        {
            ofstream outfile3o ("eps0_X_15000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 15000; i++) outfile3o<<eps0_X[i]<<endl;
            }
            outfile3o.close();
            
            ofstream outfile4o ("Q_X_15000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 15000; i++) outfile4o<<Q_X[i]<<endl;
            }
            outfile4o.close();
            
            ofstream outfile5o ("logrhoc_X_15000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 15000; i++) outfile5o<<logrhoc_X[i]<<endl;
            }
            outfile5o.close();
        }
        if(t1 == 16000)
        {
            ofstream outfile3p ("eps0_X_16000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 16000; i++) outfile3p<<eps0_X[i]<<endl;
            }
            outfile3p.close();
            
            ofstream outfile4p ("Q_X_16000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 16000; i++) outfile4p<<Q_X[i]<<endl;
            }
            outfile4p.close();
            
            ofstream outfile5p ("logrhoc_X_16000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 16000; i++) outfile5p<<logrhoc_X[i]<<endl;
            }
            outfile5p.close();
        }
        if(t1 == 17000)
        {
            ofstream outfile3q ("eps0_X_17000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 17000; i++) outfile3q<<eps0_X[i]<<endl;
            }
            outfile3q.close();
            
            ofstream outfile4q ("Q_X_17000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 17000; i++) outfile4q<<Q_X[i]<<endl;
            }
            outfile4q.close();
            
            ofstream outfile5q ("logrhoc_X_17000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 17000; i++) outfile5q<<logrhoc_X[i]<<endl;
            }
            outfile5q.close();
        }
        if(t1 == 18000)
        {
            ofstream outfile3r ("eps0_X_18000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 18000; i++) outfile3r<<eps0_X[i]<<endl;
            }
            outfile3r.close();
            
            ofstream outfile4r ("Q_X_18000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 18000; i++) outfile4r<<Q_X[i]<<endl;
            }
            outfile4r.close();
            
            ofstream outfile5r ("logrhoc_X_18000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 18000; i++) outfile5r<<logrhoc_X[i]<<endl;
            }
            outfile5r.close();
        }
        if(t1 == 19000)
        {
            ofstream outfile3s ("eps0_X_19000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 19000; i++) outfile3s<<eps0_X[i]<<endl;
            }
            outfile3s.close();
            
            ofstream outfile4s ("Q_X_19000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 19000; i++) outfile4s<<Q_X[i]<<endl;
            }
            outfile4s.close();
            
            ofstream outfile5s ("logrhoc_X_19000_mega_run_new_1_P.txt");
            {
                for(int i = 0; i <= 19000; i++) outfile5s<<logrhoc_X[i]<<endl;
            }
            outfile5s.close();
        }
    }
    
    vector<double> sorted_eps0_X;
    vector<double> sorted_Q_X;
    vector<double> sorted_logrhoc_X;
    
    sorted_eps0_X.assign(eps0_X,eps0_X+(Niters + 1));
    sorted_Q_X.assign(Q_X,Q_X+(Niters + 1));
    sorted_logrhoc_X.assign(logrhoc_X,logrhoc_X+(Niters + 1));
    
    stable_sort (sorted_eps0_X.begin(), sorted_eps0_X.end());
    stable_sort (sorted_Q_X.begin(), sorted_Q_X.end());
    stable_sort (sorted_logrhoc_X.begin(), sorted_logrhoc_X.end());
    
    
    
//    Print to external files the complete chains of the free parameters of the model.
//    The below example is done for a total (burn-in + effective) number of MCMC iterations equal to 20000.
    ofstream outfile3 ("eps0_X_20000_mega_run_new_1_P.txt");
    {
        for(int i = 0; i <= Niters; i++) outfile3<<eps0_X[i]<<endl;
    }
    outfile3.close();
    
    ofstream outfile4 ("Q_X_20000_mega_run_new_1_P.txt");
    {
        for(int i = 0; i <= Niters; i++) outfile4<<Q_X[i]<<endl;
    }
    outfile4.close();
    
    ofstream outfile5 ("logrhoc_X_20000_mega_run_new_1_P.txt");
    {
        for(int i = 0; i <= Niters; i++) outfile5<<logrhoc_X[i]<<endl;
    }
    outfile5.close();
    
    
    
//    Indexes of the MCMC iterations and sorted complete chains of the free parameters of the model.
    ofstream outfile8 ("eps0_X_sorted_20000_mega_run_new_1_P.txt");
    {
        int count = 0;
        for (vector<double>::iterator it=sorted_eps0_X.begin(); it!=sorted_eps0_X.end(); ++it)
        {
            outfile8<<count<<"    "<<*it<<endl;
            count++;
        }
    }
    outfile8.close();
    
    ofstream outfile9 ("Q_X_sorted_20000_mega_run_new_1_P.txt");
    {
        int count = 0;
        for (vector<double>::iterator it=sorted_Q_X.begin(); it!=sorted_Q_X.end(); ++it)
        {
            outfile9<<count<<"    "<<*it<<endl;
            count++;
        }
    }
    outfile9.close();
    
    ofstream outfile10 ("logrhoc_X_sorted_20000_mega_run_new_1_P.txt");
    {
        int count = 0;
        for (vector<double>::iterator it=sorted_logrhoc_X.begin(); it!=sorted_logrhoc_X.end(); ++it)
        {
            outfile10<<count<<"    "<<*it<<endl;
            count++;
        }
    }
    outfile10.close();
    
    
    
//    Sorted complete chains of the free parameters of the model.
    ofstream outfile13 ("eps0_X_sorted_no_indexes_20000_mega_run_new_1_P.txt");
    {
        for (vector<double>::iterator it=sorted_eps0_X.begin(); it!=sorted_eps0_X.end(); ++it) outfile13<<*it<<endl;
    }
    outfile13.close();
    
    ofstream outfile14 ("Q_X_sorted_no_indexes_20000_mega_run_new_1_P.txt");
    {
        for (vector<double>::iterator it=sorted_Q_X.begin(); it!=sorted_Q_X.end(); ++it) outfile14<<*it<<endl;
    }
    outfile14.close();
    
    ofstream outfile15 ("logrhoc_X_sorted_no_indexes_20000_mega_run_new_1_P.txt");
    {
        for (vector<double>::iterator it=sorted_logrhoc_X.begin(); it!=sorted_logrhoc_X.end(); ++it) outfile15<<*it<<endl;
    }
    outfile15.close();
    
    
    
    
//    Deallocation of memory

    for(int i = 0; i < NGal; i++)
    {
        delete[] x[i];
        delete[] R[i];
        delete[] sigz[i];
    }
    delete[] x;
    delete[] R;
    delete[] sigz;
    
    for(int i = 0; i < NGal; i++)
    {
        delete[] z[i];
    }
    delete[] z;
    
    for(int i = 0; i < NGal; i++)
    {
        delete[] Rhalf[i];
        delete[] vbar[i];
        delete[] sigzbar[i];
    }
    delete[] Rhalf;
    delete[] vbar;
    delete[] sigzbar;
    


    for(int i = 0; i < NGal; i++)
    {
        for(int j = 0; j <= params[i].NR; j++)
        {
            delete[] rho[i][j];
            delete[] S[i][j];
            delete[] DRphiN[i][j];
            delete[] phi0[i][j];
            delete[] phi1[i][j];
            delete[] DRphi1[i][j];
            delete[] Dzphi1[i][j];
            delete[] eps[i][j];
            delete[] DReps[i][j];
            delete[] Dzeps[i][j];
            delete[] integrandf1[i][j];
            delete[] Atrap1[i][j];
            delete[] integralf12D[i][j];
            delete[] Atrap2[i][j];
        }
    }
    delete[] rho;
    delete[] S;
    delete[] DRphiN;
    delete[] phi0;
    delete[] phi1;
    delete[] DRphi1;
    delete[] Dzphi1;
    delete[] eps;
    delete[] DReps;
    delete[] Dzeps;
    delete[] integrandf1;
    delete[] Atrap1;
    delete[] integralf12D;
    delete[] Atrap2;
    
    
    
    for(int i = 0; i < NGal; i++)
    {
        delete[] rvvalue[i];
        delete[] vvalue[i];
        delete[] devstvvalue[i];
        delete[] vbarinterp_X[i];
        delete[] vbarinterp_Y[i];
        delete[] argchi2v_X[i];
        delete[] argchi2v_Y[i];
    }
    delete[] rvvalue;
    delete[] vvalue;
    delete[] devstvvalue;
    delete[] vbarinterp_X;
    delete[] vbarinterp_Y;
    delete[] argchi2v_X;
    delete[] argchi2v_Y;
    
    
    
    for(int i = 0; i < NGal; i++)
    {
        delete[] rsigzvalue[i];
        delete[] sigzvalue[i];
        delete[] devstsigzvalue[i];
    }
    delete[] rsigzvalue;
    delete[] sigzvalue;
    delete[] devstsigzvalue;
    
    
    
    for(int i = 0; i < NGal; i++)
    {
        gsl_interp_free (interp1[i]);
        gsl_interp_free (interp2[i]);
    }
    
    
    
    return 0;
}



// Definition of function used in the SOR Poisson solver
////////////////////////////////////////////////////////////////////////
double Error(double **V0, double **V1, int A, int B)
//
// Compute error
////////////////////////////////////////////////////////////////////////
{
//    int  i,k,n;
    double err;
    double err_L1  = 0.0;   // average relative error per lattice point
    double err_max = 0.0;
    
    for (int i = 0; i <= A; i++)
    {
        for (int k = 0; k <= B; k++)
        {
            err      = fabs(V1[i][k] - V0[i][k]); //(fabs(V0[i][j]) + 1.e-40);
            err_L1  += err;
            err_max  = (err > err_max ? err:err_max);
        }
    }
    
    err_L1 /= (double)(A-1)*(B-1);
    return err_L1;
}
////////////////////////////////////////////////////////////////////////
