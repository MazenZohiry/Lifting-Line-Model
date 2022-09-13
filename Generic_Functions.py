import numpy as np
from matplotlib import pyplot as plt
import nums_from_string as numst
# import scipy.interpolate as spint
def Cosine_Sampler(a, b, n):
	""" Sample with a cosine distribution
		between [a, b] with n points
	"""
	ksi = np.zeros((1, n));
	for i in range(1, n + 1):
		ksi[0][i - 1] = np.cos((2*i - 1)*np.pi/(2*n));
	xi = (a + b)/2 + (b - a)/2*ksi;
	return xi[0][::-1];

def Plot_results_1T(results, Turbines, plot_variables, var_label, save_plots):
	N = Turbines[0].N;
	r_R = results[plot_variables[0]];
	for idx in range(1, len(plot_variables)):
		fig = plt.figure();
		Generic_single_1D_plot(r_R, results[plot_variables[idx]][0:N], var_label[0], var_label[idx], False, None, fig = fig, line_label = r'$Lifting \: Line$');
		BEM_res = np.load('./BEM_results/{}8.npz'.format(plot_variables[idx]));
		Generic_single_1D_plot(BEM_res['arr_0'], BEM_res['arr_1'], var_label[0], var_label[idx], save_plots, 'One_Turbine/%s'%plot_variables[idx], fig = fig, line_label = r'$BEM$', marker = '-o');
	return 0;

def Plot_results_NT(Results_L_phi, names_L, names_phi, plot_variables, var_label, save_plots):
	Turbine = Results_L_phi['Turbines'][names_L[0]];
	N = Turbine[0].N;
	r_R = Results_L_phi['results'][names_L[0]][plot_variables[0]];
	markers = ['-x', '-o', '-^', '-s', '-d', '-*'];
	for turbine in range(len(Turbine)):
		for idx in range(1, len(plot_variables)):
			for nb in range(Turbine[0].Nb):
				fig_L = plt.figure('L Turbine {} Blade {}'.format(turbine, nb + 1));
				for name_idx in range(len(names_L)):
					Generic_single_1D_plot(r_R, Results_L_phi['results'][names_L[name_idx]][plot_variables[idx]][(nb*N + Turbine[0].Nb*N*turbine):((nb + 1)*N + Turbine[0].Nb*N*turbine)], \
											var_label[0], var_label[idx], save_plots, 'Two_Turbine_2/%s_Turbine%i_blade%i_L'%(plot_variables[idx], turbine + 1, nb + 1), r'$L \: = \: %s$'%(names_L[name_idx]), fig = fig_L, marker = markers[name_idx]);
				fig_phi = plt.figure('phi Turbine {} Blade {}'.format(turbine, nb + 1));
				for name_idx in range(len(names_phi)):
					Generic_single_1D_plot(r_R, Results_L_phi['results'][names_phi[name_idx]][plot_variables[idx]][(nb*N + Turbine[0].Nb*N*turbine):((nb + 1)*N + Turbine[0].Nb*N*turbine)], \
										var_label[0], var_label[idx], save_plots, 'Two_Turbine_2/%s_Turbine%i_blade%i_phi'%(plot_variables[idx], turbine + 1, nb + 1), r'$\varphi \: = \: %0.1f %s$'%(float(names_phi[name_idx]) + 90.0, chr(176)), fig = fig_phi, marker = markers[name_idx]);
			if (save_plots == False):
				plt.show();
			plt.close('all');
	return 0;

def Plot_sensitivity_results(Results_sensitivity, names_sensitivity, plot_variables, var_label, save_plots):
	parameters = names_sensitivity.keys();
	markers = ['-x', '-o', '-^', '-s', '-d', '-*', '-h', '-+', '-p'];
	for idx in range(1, len(plot_variables)):
		for parameter in parameters:
			fig = plt.figure();
			names = names_sensitivity[parameter];
			mk_idx = 0;
			for name in names:
				Turbine = Results_sensitivity['Turbines'][name][0];
				results = Results_sensitivity['results'][name];
				r_R = results[plot_variables[0]];
				try:
					if parameter == 'f_w':
						label = r'$%s \: = \: %0.1f$'%(parameter, numst.get_nums(name)[0]);
					else:
						label = r'$%s \: = \: %i$'%(parameter, numst.get_nums(name)[0]);
				except:
					label = r'$%s \: %s$'%(name, parameter);
				Generic_single_1D_plot(r_R, results[plot_variables[idx]][0:Turbine.N], var_label[0], var_label[idx], save_plots, \
											'sensitivity/%s_sensitivity_%s'%(plot_variables[idx], parameter), line_label = label, fig = fig, marker = markers[mk_idx]);
				if mk_idx < len(markers):
					mk_idx += 1;
		if (save_plots == False):
			plt.show();
		plt.close('all');
	return 0;

def Generic_single_1D_plot(x, y, xlabel, ylabel, save_plots, name, line_label = None, fig = None, marker = '-x', xscale = 'linear', yscale = 'linear'):
	if fig == None:	plt.figure();
	else: plt.figure(fig.number);
	plt.plot(x, y, marker, label = line_label, mfc = 'none', linewidth = 0.75);
	plt.xscale(xscale);
	plt.yscale(yscale);
	plt.grid(True);
	plt.minorticks_on();
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	plt.xlabel('{}'.format(xlabel), fontsize = 13);
	plt.ylabel('{}'.format(ylabel), fontsize = 13);
	if line_label != None:
		plt.legend(loc = 'best');
	if (save_plots):
		plt.savefig('./Figures/{}.png'.format(name), dpi = 300, bbox_inches = "tight");
	return 0;

def Plot_Turbine_Wake_system(Turbines, wakes, truncate, save_plot):
	N_turbine = len(Turbines);
	fig = plt.figure();	#figsize = (10, 8)
	ax = fig.add_subplot(111, projection = '3d');
	ax.set_xlabel(r'$x$');		ax.set_ylabel(r'$y$');		ax.set_zlabel(r'$z$');
	ax.view_init(elev = 20, azim = -150);
	ax.set_box_aspect(aspect = (2, N_turbine, 1));
	for i in range(N_turbine):
		wakes[N_turbine - 1 - i].plot_wake(fig, ax, truncate = truncate);
	plt.tight_layout();
	if (save_plot):
		plt.savefig('./Figures/Turbine_L_%i_phi_%i.png'%(Turbines[-1].L, np.degrees(Turbines[-1].phi)), dpi = 300, bbox_inches = "tight");
	return 0;
	
def Plot_N_Turbine_Wake_system(Results_N_Turbine, names, truncate, save_plot):
	Turbine_lst = Results_N_Turbine['Turbines'];
	wake_lst = Results_N_Turbine['wakes'];
	for name in names:
		Plot_Turbine_Wake_system(Turbine_lst[name], wake_lst[name], truncate, save_plot);
	return 0;