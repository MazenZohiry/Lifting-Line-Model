import numpy as np
import matplotlib.pyplot as plt
from Generic_Functions import Cosine_Sampler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
# import numba.experimental as nub_exp
# import numba as nub
# spec = [
# 	('N', nub.int32),
# 	('R', nub.float32),
# 	('Nb', nub.int32),
# 	('twist', nub.float32[:]),
# 	('chord', nub.float32[:]),
# 	('dist', nub.char),
# 	('r_R', nub.float32[:]),
# 	('r', nub.float32[:]),
# 	('control_p', nub.float32[:, :, :]),
# 	('bound_vort_p', nub.float32[:, :, :]),
# 	('r_start', nub.float32),
# 	('r_end', nub.float32),
# 	('L', nub.float32),
# 	('phi', nub.float32),
# 	('rot_angle', nub.float32),
# 	('airfoil', nub.char),
# 	('polar_alpha', nub.float32[:]),
# 	('polar_cl', nub.float32[:]),
# 	('polar_cd', nub.float32[:]),
# 	('data1', nub.float32[:, :])
# ];
# @nub_exp.jitclass(spec)
class LL_sys():
	def __init__(self, N, Nb, airfoil, L, phi, R, r_start, r_end, distribution):
		self.N = N;												# Number of horseshoe rings, (segments along the span in each blade)
		self.R = R;												# Radius of blade [m]
		self.Nb = Nb;											# Number of bladess
		self.twist = None;										# Twist distribution
		self.chord = None;										# Chord distribution
		self.dist = distribution;								# Distribution type ['uniform', 'cosine']
		self.r_R = None;										# Spanwise locations along the blade normalised by R [-]
		self.r = None;											# Spanwise locations along the blade [m]
		self.control_p = None;									# Matrix of control points coordinates, shape = (Nb, 3, N)
		self.bound_vort_p = None;								# Matrix of bound vortex points, shape = (Nb, 3, N + 1)
		self.r_start = r_start;									# r/R start location of the blade
		self.r_end = r_end;										# r/R end location of the blade
		self.L = L;												# Spacing between turbines [m]
		self.phi = phi;											# Phase offset angle
		self.rot_angle = 2*np.pi/self.Nb;						# Angle of rotation to the next blade [rad]
		self.airfoil = airfoil;									# Airfoil name
		self.polar_alpha = None;								# Alpha polars
		self.polar_cl = None;									# Cl polars
		self.polar_cd = None;									# Cd polars
		self.Aifoil_data();

	# @property
	def Aifoil_data(self):
		data1 = np.genfromtxt(self.airfoil, skip_header = 2);
		self.polar_alpha = data1[:, 0];
		self.polar_cl = data1[:, 1];
		self.polar_cd = data1[:, 2];
		return 0;

	def Transformation_Mat(self, rot_angle = None):
		""" This function returns a linear transformation (rotations)
			 matrix to transform (rotate) coordinates on a blade. Th-
			 is uses the unit-axis transformation in the x-direction
		"""
		if rot_angle == None:
			rot_angle = self.rot_angle;

		rot_mat = np.array([[1, 		0,					0], \
							[0,  np.cos(rot_angle), np.sin(rot_angle)], \
							[0, -np.sin(rot_angle), np.cos(rot_angle)]]);		# Rotation matrix (3D)
		return rot_mat;

	def Blade_points(self, p0, p):
		""" Function that takes an initial set of points (p0) along a given (single) blade
			and transform the points to the points on the other blades. The transformation
			simply rotates the initial points (p0) using self.Transformation_Mat() to com-
			pute the other points and puts them in the matrix p. p.shape = (Nb, p0.shape)
		"""
		p[0, :, :] = p0;										# Add p0 to the main matrix
		rot_mat = self.Transformation_Mat();					# Create rotation matrix
		for nb in range(1, self.Nb):							# Loop over each blade
			p[nb, :, :] = rot_mat.dot(p[nb - 1, :, :]);			# Add transformed points to the matrix
		return 0;

	def Blade_Geometry(self, twist_f, chord_f):
		""" This function defines the blade geometry for the Lifting line model.
			It creates the (x, y, z) points for the bound vortices at 0.25c, as
			well as the control point at 0.75c.
		"""
		# Set spacing
		if self.dist == 'uniform':
			self.r_R = np.linspace(self.r_start, self.r_end, self.N + 1);
		elif self.dist == 'cosine':
			self.r_R = Cosine_Sampler(self.r_start, self.r_end, self.N + 1);
		self.r = self.r_R*self.R;
		self.twist = twist_f(self.r_R);								# Compute twist distribution
		self.chord = chord_f(self.r_R);								# Compute chord distribution
		self.bound_vort_p = np.zeros((self.Nb, 3, len(self.r_R)));	# Initialise bound vortex point array
		B1 = self.Transformation_Mat(rot_angle = self.phi).dot(np.vstack([np.zeros_like(self.r_R), \
													  0.25*self.chord, 		 \
													  self.r]));	# (x, y, z) Bound vortex points on the first blade (chosen to be the vertical blade)
		self.Blade_points(B1, self.bound_vort_p);					# Compute Bound vortex points on all the blades
		self.control_p = np.zeros((self.Nb, 3, self.N));			# Initialise control points array
		CP1 = self.Transformation_Mat(rot_angle = self.phi).dot(np.vstack([np.zeros(self.N), 		\
						 0.25*(self.chord[1:] + self.chord[:-1])/2, \
						 self.R*(self.r_R[1:] + self.r_R[:-1])/2]));	# (x, y, z) control points on the first blade (chosen to be the vertical blade)
		self.Blade_points(CP1, self.control_p);						# Compute control points on all the blades
		self.control_p += np.array([[0.0], [self.L], [0]]);
		self.bound_vort_p += np.array([[0.0], [self.L], [0]]);
		return 0;
	
	def Plot_Turbine(self, fig = None, ax = None):
		""" Function to plot the turbine bound points and control points """
		if fig == None and ax == None:
			fig = plt.figure();
			ax = fig.add_subplot(111, projection = '3d');
			ax.set_xlabel(r'$x$');		ax.set_ylabel(r'$y$');		ax.set_zlabel(r'$z$');
			ax.view_init(elev = 20, azim = -140);
			ax.set_box_aspect(aspect = (2, 1, 1));
		# for i in range(np.shape(self.bound_vort_p)[0]):
		# 	ax.scatter(self.bound_vort_p[i, 0, :], self.bound_vort_p[i, 1, :], self.bound_vort_p[i, 2, :], s = 1);
		# 	ax.scatter(self.control_p[i, 0, :], self.control_p[i, 1, :], self.control_p[i, 2, :], marker = 'x', s = 1);
		blade0 = self.Transformation_Mat(rot_angle = self.phi).dot(np.vstack((np.hstack([np.zeros_like(self.r_R), np.zeros_like(self.r_R)]), np.hstack([np.zeros_like(self.r_R), self.chord[::-1]]), np.hstack([self.r, self.r[::-1]]))));
		blade = np.zeros((self.Nb, blade0.shape[0], blade0.shape[1]));
		self.Blade_points(blade0, blade);
		blade = np.hstack(blade) + np.array([[0.0], [self.L], [0]]);
		ax.plot3D(blade[0, :], blade[1, :], blade[2, :], '-k');
		p = Circle((self.L, 0), self.r_start*self.R, color = 'k');
		ax.add_patch(p);
		art3d.pathpatch_2d_to_3d(p, z = 0, zdir = "x");
		return 0;
		
