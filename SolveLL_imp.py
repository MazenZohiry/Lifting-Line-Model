import numpy as np
from Load_BE import loadBladeElement
from Single_ring_vel import Single_ring_contribution
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CT_a import Tot_CT_CQ_CP

def Induced_velocity_matrix(Turbines, wakes, plot = False, save_plot = False):
	control_p = np.hstack([np.hstack(T.control_p) for T in Turbines]);
	N_cp = control_p.shape[1];
	MatrixU = np.zeros((N_cp, N_cp));
	MatrixV = np.zeros((N_cp, N_cp));
	MatrixW = np.zeros((N_cp, N_cp));
	for i_cp in range(control_p.shape[1]):
		cont_t = control_p[:, i_cp].reshape(3, 1);
		for j_ring in range(Turbines[0].N):
			for T_idx in range(len(Turbines)):
				U, V, W = Single_ring_contribution(cont_t, j_ring, Turbines[T_idx], wakes[T_idx]);
				idx_j = T_idx*Turbines[0].N*Turbines[0].Nb + j_ring + np.arange(0, Turbines[0].Nb, 1, dtype = int)*Turbines[0].N;
				MatrixU[i_cp, idx_j] = U;
				MatrixV[i_cp, idx_j] = V;
				MatrixW[i_cp, idx_j] = W;
	if (plot):
		if len(Turbines) == 1:
			MU = MatrixU.reshape(Turbines[0].Nb*Turbines[0].N, Turbines[0].Nb, Turbines[0].N).sum(axis = 1)[0:Turbines[0].N, :];
			MV = MatrixV.reshape(Turbines[0].Nb*Turbines[0].N, Turbines[0].Nb, Turbines[0].N).sum(axis = 1)[0:Turbines[0].N, :];
			MW = MatrixW.reshape(Turbines[0].Nb*Turbines[0].N, Turbines[0].Nb, Turbines[0].N).sum(axis = 1)[0:Turbines[0].N, :];
		else:
			MU = MatrixU;
			MV = MatrixV;
			MW = MatrixW;
		h, w = plt.figaspect(2.85/1.2);
		plt.figure(figsize = (w, h)); 
		plt.subplot(1, 3, 1);
		plt.imshow(MU);
		plt.colorbar(orientation = 'horizontal');
		plt.title(r'$u$');
		plt.subplot(1, 3, 2);
		plt.imshow(MV);
		plt.colorbar(orientation = 'horizontal');
		plt.title(r'$v$');
		plt.subplot(1, 3, 3);
		plt.imshow(MW);
		plt.colorbar(orientation = 'horizontal');
		plt.title(r'$w$');
		plt.tight_layout();
		if (save_plot):
			plt.savefig('./Figures/Induction_matrix_Nt%i_Nb_%i_N_%i.png'%(len(Turbines), Turbines[0].Nb, Turbines[0].N), dpi = 300, bbox_inches = "tight");
	return MatrixU, MatrixV, MatrixW;

def SolveLL(Niterations, errorlimit, des_variables, Turbines, wakes, Omega, wind, W = 0.5, iter_wake_skip = 4, disp_iter = False):
	N_t = len(Turbines);
	GammaNew = np.ones(Turbines[0].Nb*Turbines[0].N*N_t);
	control_points = np.hstack([np.hstack(T.control_p) - np.array([[0.0], [T.L], [0.0]]) for T in Turbines]);
	radial_station = (Turbines[0].r_R[0:-1] + Turbines[0].r_R[1:])/2;
	radial_position = np.linalg.norm(control_points, axis = 0);
	for kiter in range(Niterations):
		if kiter >= 1 and disp_iter:
			text = ('Iteration = {},   residual = {},   relax-factor = {}'.format(kiter, error, W));
			sys.stdout.write('\r' + text);
		if kiter % iter_wake_skip == 0:
			MatrixU, MatrixV, MatrixW = Induced_velocity_matrix(Turbines, wakes);
		Gamma = GammaNew;
		u = MatrixU.dot(Gamma);
		v = MatrixV.dot(Gamma);
		w = MatrixW.dot(Gamma);
		U_induced = np.vstack((u, v, w));
		# calculate total perceived velocity
		vrot = np.cross(np.array([[-Omega], [0], [0]]), control_points, axisa = 0, axisb = 0).T;                              # tangential velocity of at a given cp
		vel = wind + vrot + U_induced;                               						# axial
		azimdir = np.cross(np.vstack((-1/radial_position, np.zeros_like(radial_position), np.zeros_like(radial_position))), control_points, axisa = 0, axisb = 0).T;			# rotational direction
		vazim = np.sum(azimdir*vel, axis = 0);                                              # azimuthal direction
		vaxial = vel[0, :];                                          						# axial velocity

		#calculate loads using blade element theory
		fnorm, ftan, gamma, alpha, phi = loadBladeElement(vaxial, vazim, Turbines[0], N_t);
		#new point of new estimate of circulation for the blade section
		GammaNew = gamma;
		a = (-u/wind[0]);
		aline = (vazim/(radial_position*Omega) - 1);
		#check convergence of solution
		refererror = np.max(np.abs(GammaNew));
		refererror = np.max((refererror, 0.001));                        # define scale of bound circulation
		error = np.max(np.abs(GammaNew - Gamma));                        # difference betweeen iterations
		error = error/refererror;                                        # relative error
		if error < errorlimit:
			# if error smaller than limit, stop iteration cycle
			break
	 
		# Aitken method
		rk = GammaNew - Gamma;
		if kiter == 0:
			GammaNew = Gamma + W*(GammaNew - Gamma);
			rkm1 = rk.copy();
		else:
			W *= -np.dot(rkm1, rk - rkm1)/np.dot(rk - rkm1, rk - rkm1);
			GammaNew = Gamma + W*(GammaNew - Gamma);
			rkm1 = rk.copy();
		# Update wake
		if kiter % iter_wake_skip == 0:
			a_per_turbine = a.reshape(N_t, Turbines[0].Nb*Turbines[0].N);
			a_ave = np.mean(a_per_turbine, axis = 1);
			for i in range(N_t):
				wakes[i].aw = a_ave[i]; 
				wakes[i].compute_wake();

	fnorm /= (0.5*wind[0]**2*Turbines[0].R);
	ftan /= (0.5*wind[0]**2*Turbines[0].R);
	CT, CQ, CP = Tot_CT_CQ_CP(Turbines, fnorm, ftan, Omega, wind[0]);
	Gamma /= ((np.pi*wind[0]**2)/(Omega*Turbines[0].Nb));
	sys.stdout.write('\nCT = {},   CQ = {},   CP = {}\n'.format(CT, CQ, CP));
	results = dict();
	for variable in des_variables:
		results[variable] = eval(variable);
	return results;

