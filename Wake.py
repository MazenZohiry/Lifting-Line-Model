import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as ani
# import numba as nub
# @nub.jit(nopython = True, cache = True)
class Gen_Wake():
	def __init__(self, L_w, N_w_per_rot, aw, f_w, U_inf, wake_offset, Omega, LL_sys):
		self.aw = aw;
		self.f_w = f_w;
		self.U_inf = U_inf;
		self.U_wake = None;										# Wake convection speed
		self.L_w = L_w;											# Length of wake
		self.t_max = None;										# Max convection time
		self.N_w = None;											# Total number of wake points
		self.N_w_per_rot = N_w_per_rot;							# Number of segments per rotation
		self.dt = None;											# Step size
		self.Omega = Omega;										# Rotational frequency
		self.t_rot = 2*np.pi/Omega;								# Time for 1 rotation
		self.LL_sys = LL_sys;									# LL_sys object handle
		self.wake_offset = wake_offset;							# Wake start point behind the chord as a fraction of the chord length
		self.X_wake = None;										# Matrix of all the points on the wake, shape = (Nb, N_w, 3, N). (N = number of blade elements)

	def wake_points(self, t):
		x_w = t*self.U_wake;
		y_w = self.LL_sys.r*np.sin(self.Omega*t);
		z_w = self.LL_sys.r*np.cos(self.Omega*t);
		X_wake_i = np.vstack((x_w*np.ones_like(self.LL_sys.r), y_w, z_w));
		return X_wake_i;

	def compute_wake(self):
		self.U_wake = self.f_w*self.U_inf*(1 - self.aw);
		self.rot_max = self.L_w/(self.U_wake*self.t_rot);
		self.dt = self.t_rot/self.N_w_per_rot;
		self.N_w = int(self.N_w_per_rot*self.rot_max);
		dim = np.shape(self.LL_sys.bound_vort_p[0, :, :]);
		self.X_wake = np.zeros((self.LL_sys.Nb, self.N_w, dim[0], dim[1]));
		wake_offset = self.LL_sys.Transformation_Mat(rot_angle = self.LL_sys.phi).dot(np.vstack([np.zeros(dim[1]), (1 - 0.25 + self.wake_offset)*self.LL_sys.chord, np.zeros(dim[1])]));
		wake_start0 = self.LL_sys.bound_vort_p[0, :, :] + wake_offset - np.array([[0.0], [self.LL_sys.L], [0]]);
		wake_start = np.zeros(np.shape(self.X_wake[:, 0, :, :]));
		self.LL_sys.Blade_points(wake_start0, wake_start);
		self.X_wake[:, 0, :, :] = wake_start;
		rot_mat = self.LL_sys.Transformation_Mat();
		for t in range(1, self.N_w):
			self.X_wake[0, t, :, :] = wake_offset + self.LL_sys.Transformation_Mat(rot_angle = self.LL_sys.phi).dot(self.wake_points(t*self.dt));
			for nb in range(1, self.LL_sys.Nb):
				self.X_wake[nb, t, :, :] = rot_mat.dot(self.X_wake[nb - 1, t, :, :]);
		self.X_wake += np.array([[0.0], [self.LL_sys.L], [0.0]]);
		return self.X_wake;

	def plot_wake(self, fig = None, ax = None, truncate = None):
		if fig == None and ax == None:
			fig = plt.figure(figsize = (10, 8));
			ax = fig.add_subplot(111, projection = '3d');
			ax.set_xlabel(r'$x$');		ax.set_ylabel(r'$y$');		ax.set_zlabel(r'$z$');
			ax.view_init(elev = 20, azim = -140);
			ax.set_box_aspect(aspect = (2, 1, 1));
		if truncate == None:
			idx = range(self.N_w);
		else:
			idx = np.where(self.X_wake[0, :, 0, 0] <= truncate)[0];
		colour = ['r', 'g', 'b'];
		for nb in range(np.shape(self.X_wake)[0]):
			for i in range(np.shape(self.X_wake)[-1]):
				ax.plot3D(self.X_wake[nb, idx, 0, i], self.X_wake[nb, idx, 1, i], self.X_wake[nb, idx, 2, i], colour[nb], linewidth = 0.6);
		self.LL_sys.Plot_Turbine(fig = fig, ax = ax);
		return 0;
#%%
if __name__ == '__main__':
	from LL_system import LL_sys
	U_inf = 1;
	aw = 0.25;
	chord_distribution = lambda r_R: -27.32*(r_R)**5 + 98.97*(r_R)**4 - \
									  144.5*(r_R)**3 + 108.9*(r_R)**2 - \
									  45.06*(r_R) + 9.941;
	twist_distribution = lambda r_R: 191.4*(r_R)**5 - 686*(r_R)**4 + \
									 986.7*(r_R)**3 - 728.2*(r_R)**2 + \
									 291.9*(r_R) - 52.56; 
	Omega = 1.6;
	save_gif = False;
	N = 3;
	Nb = 3;
	R = 50;
	r_start = 0.2;
	r_end = 1;
	L_w = 100;
	N_w = 30;
	LL_SYS = LL_sys(N, Nb, 'polarDU95W180.txt', 0, 0, R, r_start, r_end, 'uniform');
	LL_SYS.Blade_Geometry(twist_distribution, chord_distribution);
	wake = Gen_Wake(L_w, N_w, aw, 1, U_inf, 0.25, Omega, LL_SYS);
	X_wake = wake.compute_wake();
	wake.plot_wake(truncate = 5);
	plt.show();

	fig = plt.figure(figsize = (10, 5));
	ax = fig.add_subplot(111, projection = '3d');
	ax.view_init(elev = 20, azim = -140);
	ax.set_xlabel(r'$x$');		ax.set_ylabel(r'$y$');		ax.set_zlabel(r'$z$');
	ax.set_box_aspect(aspect = (2.5, 1, 1));

	def animate(j, ax, X_wake):
		if j == 0: ax.clear();
		for i in range(np.shape(X_wake)[-1]):
			for nb in range(np.shape(X_wake)[0]):
				ax.plot([X_wake[nb, j, 0, i], X_wake[nb, j + 1, 0, i]], \
						[X_wake[nb, j, 1, i], X_wake[nb, j + 1, 1, i]], \
						[X_wake[nb, j, 2, i], X_wake[nb, j + 1, 2, i]]);
		return 0;

	anim = ani.FuncAnimation(fig, animate, np.shape(X_wake)[1] - 1, fargs = (ax, X_wake), interval = 50);
	if save_gif:
		anim.save('./wake.gif');
	plt.show();