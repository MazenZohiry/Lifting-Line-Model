import numpy as np


def Tot_CT_CQ_CP(Turbines, fnorm, ftan, Omega, Uinf):
	CT = np.zeros(len(Turbines));
	CQ = np.zeros(len(Turbines));
	CP = np.zeros(len(Turbines));
	for i in range(len(Turbines)):
		#CT caluclation
		idx = range(i*(Turbines[i].N*Turbines[i].Nb), (i + 1)*(Turbines[i].N*Turbines[i].Nb));
		# CT = np.zeros(np.shape(fnorm[idx]));
		dr_blade = (Turbines[i].r[1:] - Turbines[i].r[0:-1]);
		dr = np.tile(dr_blade, Turbines[i].Nb);
		CT[i] = np.sum(fnorm[idx]*dr/(np.pi*Turbines[i].R));

		#CQ calcualtion
		# CQ = np.zeros(np.shape(ftan));
		r_R_mid_blade = (Turbines[i].r_R[0:-1] + Turbines[i].r_R[1:])/2;
		r_R_mid = np.tile(r_R_mid_blade, Turbines[i].Nb);
		CQ[i] = np.sum(ftan[idx]*dr*r_R_mid*Turbines[i].R/(np.pi*Turbines[i].R**2));

		#CP calcualtion
		CP[i] = np.sum(ftan[idx]*dr*r_R_mid*Turbines[i].R*Omega/(Uinf*np.pi*Turbines[i].R));
	return CT, CQ, CP;