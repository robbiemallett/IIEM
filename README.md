# IIEM
The Improved Integral Equation Model for scattering from a rough surface.

This is derived from the modifications to IEM by Hsieh et al. (1997).

This code is translated more-or-less function for function from the MATLAB code published by Ulaby & Long, 2014.

Key differences are that vectors are no longer passed to the double integration during calculation of cross-pol backscatter, 
and the double integration of the co-pol is split into two (real and complex parts), as scipy leverages a fortran library that can't do complex double integrations. 
