import numpy as np
from Load_BE import loadBladeElement
from Single_ring_vel import Single_ring_contribution
def SolveLL(Niterations, LL_sys,N ,Omega, Nb, wind, R, polar_alpha, polar_cl, polar_cd):

    GammaNew= np.ones((Nb,N)); 
    a= np.zeros((Nb,N)); 
    aline= np.zeros((Nb,N)); 
    fnorm= np.zeros((Nb,N)); 
    ftan= np.zeros((Nb,N)); 
    
    Niterations = 20; 
    errorlimit = 0.01;
    W =0.1; 
    for kiter in range(Niterations):
        
        Gamma = GammaNew;
        
        for icp in range(N):
            results =np.zeros([len(LL_sys.r_R)-1,10]) 
            u=0; v=0; w=0           #initialize velocity at each control point

            for blade in range(Nb):
                            
                for jring in range(N):
                    Ui= Single_ring_contribution(blade,icp,jring)             
                    u = u + Ui[0]*Gamma[blade][jring];
                    v = v + Ui[1]*Gamma[blade][jring];
                    w = w + Ui[2]*Gamma[blade][jring];
                    
                # calculate total perceived velocity
                vrot =  -Omega*LL_sys.control_p[blade,1,icp]                                # tangential velocity of at a given cp
                vel = [wind[0] + u, wind[1] + v, wind[2] + w]                               # axial
                azimdir = np.cross([-1/LL_sys.r_R[icp], 0 , 0]  , LL_sys.control_p[icp]);   # rotational direction
                vazim = np.dot(azimdir , vel);                                              # azimuthal direction
                vaxial =  np.dot([1, 0, 0] , vel);                                          # axial velocity
                
                #calculate loads using blade element theory
                results = loadBladeElement(vaxial, vazim, LL_sys.r_R[icp], LL_sys.chord[icp], LL_sys.twist[icp], polar_alpha, polar_cl, polar_cd);
                
                #new point of new estimate of circulation for the blade section
                GammaNew[blade][icp] = results[:,2]
                a[blade][icp] = (-(u + vrot[0])/wind[0]);
                aline[blade][icp] =(vazim/(LL_sys.r_R[icp]*R*Omega)-1);
                fnorm[blade][icp] = results[:,0];
                ftan[blade][icp] = results[:,1];
                Gamma[blade][icp] = results[:,2];

        #check convergence of solution
        refererror =np.max(np.abs(GammaNew));
        refererror =np.max(refererror,0.001);                           # define scale of bound circulation
        error =np.max(np.abs(GammaNew - Gamma));                        # difference betweeen iterations
        error= error/refererror;                                        # relative error
        
        if error < errorlimit:
          # if error smaller than limit, stop iteration cycle
          break
     
        # set new estimate of bound circulation
        GammaNew = (1-W)*Gamma + W*GammaNew;

    return [a , aline, fnorm , ftan, Gamma]