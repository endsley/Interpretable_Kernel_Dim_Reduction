# InterpretableKernelDimReduction
Interpretable Kernel Dimensionality Reduction 

***

## Description
Interpretable Kernel Dimensionality Reduction is a supervised technique similar to linear discriminant analysis. Instead of working with the feature space, the data is first projected to a low dimension subspace. Since this projection is linear, it is interpretable. Once projected into the lower dimension subspace, it is then projected again into the Reproducing Kernel Hilbert Space (RKHS).  The objective function here is solved by ISM+.

## Input/Output
This code takes data and its labels as input. The files should be in csv file format
This code outputs a projection matrix W, as well as the reduced dimension data XW. 


### Code Instructions
You will need to install numpy, pytorch, sklearn. 
Example of how to run this code is in SDR_examply.py



### Citation
Please cite this work if you use it in your research.  
@Misc{wu2019,  
author =   {Chieh Wu},  
title =    {{InterpretableKernelDimReduction}: Interpretable Kernel Dimension Reduction library},  
howpublished = {\url{https://github.com/neu-spiral/InterpretableKernelDimReduction}},  
year = {2019--2019}  
}  

