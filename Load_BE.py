import numpy as np
# define function to determine load in the blade element
def loadBladeElement(vnorm, vtan, LL_sys, N_t):
    """
    calculates the load in the blade element
    """
    vmag2 = vnorm**2 + vtan**2;
    inflowangle = np.arctan2(vnorm, vtan);
    twist = np.tile(LL_sys.twist[0:-1], LL_sys.Nb*N_t);
    chord = np.tile(LL_sys.chord[0:-1], LL_sys.Nb*N_t);
    alpha = twist + inflowangle*180/np.pi;
    cl = np.interp(alpha, LL_sys.polar_alpha, LL_sys.polar_cl);
    cd = np.interp(alpha, LL_sys.polar_alpha, LL_sys.polar_cd);
    lift = 0.5*vmag2*cl*chord;
    drag = 0.5*vmag2*cd*chord;
    fnorm = lift*np.cos(inflowangle) + drag*np.sin(inflowangle);
    ftan = lift*np.sin(inflowangle) - drag*np.cos(inflowangle);
    gamma = 0.5*np.sqrt(vmag2)*cl*chord;
    return fnorm, ftan, gamma, alpha, np.degrees(inflowangle);

