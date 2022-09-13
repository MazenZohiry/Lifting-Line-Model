import numpy as np
from Biot_savart import induced_vel

def Single_ring_contribution(cont_p, j_ring, LL_sys, wake):
	U = np.zeros(LL_sys.Nb); V = np.zeros(LL_sys.Nb); W = np.zeros(LL_sys.Nb);
	for blade_idx in range(LL_sys.Nb):
		bound_p1 = LL_sys.bound_vort_p[blade_idx, :, j_ring].reshape(3, 1);
		bound_p2 = LL_sys.bound_vort_p[blade_idx, :, j_ring + 1].reshape(3, 1);

		wake_p_bottom = np.hstack((bound_p1, wake.X_wake[blade_idx, :, :, j_ring].T))[:, ::-1];
		wake_p_top = np.hstack((bound_p2, wake.X_wake[blade_idx, :, :, j_ring + 1].T));

		points = np.hstack((wake_p_bottom, wake_p_top));
		u, v, w = induced_vel(points[:, 0:-1], points[:, 1:], cont_p);
		U[blade_idx] = np.sum(u);
		V[blade_idx] = np.sum(v);
		W[blade_idx] = np.sum(w);
		# u_bound, v_bound, w_bound = induced_vel(bound_p1, bound_p2, cont_p);
		# # u_wake_bottom, v_wake_bottom, w_wake_bottom = induced_vel(wake_p_bottom[:, 1:], wake_p_bottom[:, 0:-1], cont_p) 
		# u_wake_bottom, v_wake_bottom, w_wake_bottom = induced_vel(wake_p_bottom[:, 0:-1], wake_p_bottom[:, 1:], cont_p) 
		# u_wake_top, v_wake_top, w_wake_top = induced_vel(wake_p_top[:, 0:-1], wake_p_top[:, 1:], cont_p);
		# U[blade_idx] = (u_bound + np.sum(u_wake_bottom + u_wake_top));
		# V[blade_idx] = (v_bound + np.sum(v_wake_bottom + v_wake_top));
		# W[blade_idx] = (w_bound + np.sum(w_wake_bottom + w_wake_top));
		
		# print(bound_p1, '\n', bound_p2)
		# print('cp\n', cont_p, '\nwake bottom\n', wake_p_bottom, '\nwake top\n', wake_p_top, '\n')
		# print('shifted wake bott\n', wake_p_bottom[:,0:-1], '\n', wake_p_bottom[:,1:], '\n')
		# print('shifted wake top\n', wake_p_top[:,0:-1], '\n', wake_p_top[:,1:], '\n')

		# print('u_b, v_b, w_b = {}, {}, {}'.format(u_bound, v_bound, w_bound));
		# print('u, v, w = {}, {}, {}'.format(U, V, W));
		# print('\n\n');
	return U, V, W;

