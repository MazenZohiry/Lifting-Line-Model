import numpy as np
from matplotlib import pyplot as plt
from LL_system import LL_sys
from Wake import Gen_Wake
from SolveLL_imp import SolveLL, Induced_velocity_matrix
from Generic_Functions import* # Plot_results_1T, Plot_results_NT, Plot_Turbine_Wake_system, Plot_sensitivity_results, Plot_N_Turbine_Wake_system
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import sys
import multiprocessing as mp
from functools import partial
import time
#%%%% Blade Geometry %%%%%%%%%
airfoil = 'polarDU95W180.txt';										# Airfoil name
N = 30;																# Number of elements per blade
distribution = 'uniform';											# Spacing of the elements ('uniform' or 'cosine')
Nb = 3;																# Number of blades
N_turbine = 2;														# Number of turbines
R = 50;																# Radius of turbines
L = np.array([[0, 2*R], \
			  [0, 4*R], \
			  [0, 10*R], \
			  [0, 1000*R]]);										# Span-wise (y) distance between the trubines, must have a shape of (X, N_turbine)
phi = np.array([[np.pi/2, -np.pi/2 + np.pi/6], \
				[np.pi/2, -np.pi/2], \
				[np.pi/2, -np.pi/2 - np.pi/6]]);					# Phase difference between turbines, must have a shape of (X, N_turbine)
# pitch = 2;
# chord_distribution = lambda r_R: 3*(1 - r_R) + 1;
# twist_distribution = lambda r_R: (-14*(1 - r_R) + pitch);
chord_distribution = lambda r_R: -27.32*(r_R)**5 + 98.97*(r_R)**4 - \
								144.5*(r_R)**3 + 108.9*(r_R)**2 - \
								45.06*(r_R) + 9.941;				# Chord distribution function
twist_distribution = lambda r_R: 191.4*(r_R)**5 - 686*(r_R)**4 + \
								 986.7*(r_R)**3 - 728.2*(r_R)**2 + \
								 291.9*(r_R) - 52.56;       		# Twist angle distribution function
r_start = 0.2;														# r/R location of the hub
r_end = 1.0;														# r/R location of the tip
#%%%% Flow Conditions %%%%%%%%%
Uinf = 10;                                      					# Unperturbed wind speed in m/s
wind = np.array([[Uinf], [0], [0]]);								# Velocity vector
TSR = 8;                                        					# Tip speed ratio
Omega = Uinf*TSR/R;													# Rotational frequency
L_w = 5*2*R;															# Considered length of the wake
N_w = 40;															# Number of points in the wake per rotation
aw = 0.25;															# Initial guess of the wake induction factor
wake_offset = 0.25;													# Chordwise offset location (x/c) for the start of the wake
f_w = 1.0;															# Wake convection speed factor
#%%%% Run settings %%%%%%%%
Niterations = 200;													# Number of iterations
errorlimit = 1e-6;													# Convergence criteria
save_plots = False;													# Boolien to save plots
des_variables = ['radial_station', 'a', \
				 'aline', 'fnorm', 'ftan', \
				 'alpha', 'phi', 'Gamma', \
				 'CT', 'CQ', 'CP'];									# List of desired output variables from SolveLL function
plot_variables = des_variables[:-3];
var_label = [r"$\frac{r}{R}$ [-]", r"$a$ [-]", \
			 r"$a'$ [-]", r"$f_{norm}$ [-]", r"$f_{tan}$ [-]", \
			 r"$\alpha$ [{}]".format(chr(176)), r"$\phi$ [{}]".format(chr(176)), \
			 r"$\Gamma$ [-]"];										# Label of each variable
run_settings = (Niterations, errorlimit, des_variables);
#%%%%%% Main Function %%%%%%%%%
def main(N, distribution, L_w, N_w, N_turbine, f_w = f_w, Omega = Omega, run_settings = run_settings, Uinf = Uinf, wind = wind, Nb = Nb, airfoil = airfoil, R = R, r_start = r_start, r_end = r_end, twist_distribution = twist_distribution, chord_distribution = chord_distribution, L = [0], phi = [0], disp_iter = False):
	Turbines = np.zeros(N_turbine, dtype = object);					# Array to store turbine objects
	wakes = np.zeros(N_turbine, dtype = object);					# Array to store wake objects

	for i in range(N_turbine):
		Turbines[i] = LL_sys(N, Nb, airfoil, L[i], phi[i], R, r_start, r_end, distribution);
		Turbines[i].Blade_Geometry(twist_distribution, chord_distribution);
		wakes[i] = Gen_Wake(L_w, N_w, aw, f_w, Uinf, wake_offset, Omega, Turbines[i]);
		wakes[i].compute_wake();

	results = SolveLL(*run_settings, Turbines, wakes, Omega, wind, disp_iter = disp_iter);
	return Turbines, wakes, results;

def Append_Results(data, name, Results):
	Results['Turbines'][name], Results['wakes'][name], Results['results'][name] = list(data);
	return 0;

def Parallel_run_wait(processes):
	running = True;
	bool_arr = np.zeros(processes.shape);
	while running:
		for process_idx in range(len(processes)):
			bool_arr[process_idx] = processes[process_idx].ready();
		if bool_arr.all() == 1:
			running = False;
	return 0;

def Run_L_phi_iters_main(N, distribution, L_w, N_w, N_turbine, L_lst, L0_idx, phi_lst, phi0_idx):
	st = time.time();
	Results_L_phi = dict();
	Results_L_phi['Turbines'] = {};
	Results_L_phi['wakes'] = {};
	Results_L_phi['results'] = {};
	names_L = np.zeros(L_lst.shape[0], dtype = object);
	names_phi = np.zeros(phi_lst.shape[0], dtype = object);
	processes_L = np.zeros_like(names_L);
	sys.stdout.write('########################   Solving system for different L   #######################\n');
	for L_idx in range(L_lst.shape[0]):
		dist = np.linalg.norm(L_lst[L_idx])/(2*R*(N_turbine - 1));
		if dist < 10.0: name = str(dist) + 'D';
		else: name = '\infty';
		names_L[L_idx] = name;
		pool = mp.Pool(mp.cpu_count());
		append_results = partial(Append_Results, name = name, Results = Results_L_phi);
		# sys.stdout.write('########################   Solving system for L = {}   #######################\n'.format(name));
		processes_L[L_idx] = pool.apply_async(main, args = (N, distribution, L_w, N_w, N_turbine), kwds = {'L': L_lst[L_idx], 'phi': phi_lst[phi0_idx, :]}, callback = append_results);
		# Results_L_phi['Turbines'][name], Results_L_phi['wakes'][name], Results_L_phi['results'][name] = main(N, distribution, Lw, Nw, Omega, L = L_lst[L_idx], phi = phi_lst[phi0_idx, :]);
	pool.close();
	Parallel_run_wait(processes_L);
	pool.join();
	sys.stdout.write('================================================================================\n\n');
	sys.stdout.write('########################   Solving system for different phi   #######################\n');
	processes_phi = np.zeros_like(names_phi);
	for phi_idx in range(phi_lst.shape[0]):
		name = str(np.degrees(phi_lst[phi_idx, -1]));
		names_phi[phi_idx] = name;
		pool = mp.Pool(mp.cpu_count());
		append_results = partial(Append_Results, name = name, Results = Results_L_phi);
		processes_phi[phi_idx] = pool.apply_async(main, args = (N, distribution, L_w, N_w, N_turbine), kwds = {'L': L_lst[L0_idx, :], 'phi': phi_lst[phi_idx, :]}, callback = append_results);
		# sys.stdout.write('########################   Solving system for phi = {}   #######################\n'.format(name));
		# Results_L_phi['Turbines'][name], Results_L_phi['wakes'][name], Results_L_phi['results'][name] = main(N, distribution, Lw, Nw, Omega, L = L_lst[L0_idx, :], phi = phi_lst[phi_idx, :]);
		# sys.stdout.write('================================================================================\n\n');
	pool.close();
	Parallel_run_wait(processes_phi);
	pool.join();
	sys.stdout.write('================================================================================\n\n');
	sys.stdout.write('Parallel run complete, time taken = %0.4fs\n'%(time.time() - st));
	return Results_L_phi, names_L, names_phi;

def sensitivity_study(iter_var, var_arr, kwds_dict, Results_sensitivity):
	sys.stdout.write('########################   Running sensitivity for %s   #######################\n'%iter_var);
	names = np.zeros(var_arr.shape[0], dtype = object);
	processes = np.zeros_like(names);
	for var_idx in range(len(var_arr)):
		temp_kwds_dict = kwds_dict.copy();
		temp_kwds_dict[iter_var] = var_arr[var_idx];
		try:
			name = '%s_%0.1f'%(iter_var, var_arr[var_idx]);
		except:
			name = '%s'%(var_arr[var_idx]);
		names[var_idx] = name;
		pool = mp.Pool(mp.cpu_count());
		append_results = partial(Append_Results, name = name, Results = Results_sensitivity);
		processes[var_idx] = pool.apply_async(main, kwds = temp_kwds_dict, callback = append_results);
	pool.close();
	Parallel_run_wait(processes);
	pool.join();
	sys.stdout.write('================================================================================\n\n');
	return names;

def Run_sensitivity_study(iter_var_dict, kwds_dict):
	st = time.time();
	Results_sensitivity = dict();
	names_sensitivity = dict();
	Results_sensitivity['Turbines'] = {};
	Results_sensitivity['wakes'] = {};
	Results_sensitivity['results'] = {};
	for iter_var in iter_var_dict.keys():
		names_sensitivity[iter_var] = sensitivity_study(iter_var, iter_var_dict[iter_var], kwds_dict, Results_sensitivity);
	sys.stdout.write('Parallel run complete, time taken = %0.4f\n'%(time.time() - st));
	return Results_sensitivity, names_sensitivity;
#%%%%%% Run main and plot results %%%%%%%%%%
if __name__ == '__main__':
	if N_turbine == 1:
		# Single Turbine case
		Turbines, wakes, results = main(N, distribution, L_w, N_w, N_turbine, disp_iter = True);
		Plot_Turbine_Wake_system(Turbines, wakes, 10, save_plots);
		Induced_velocity_matrix(Turbines, wakes, plot = True, save_plot = save_plots);
		Plot_results_1T(results, Turbines, plot_variables, var_label, save_plots);
		plt.show();
		plt.close('all');
		# Sensitivity of parameters
		kwds_dict = {'N': N, 'distribution': distribution, 'L_w': L_w, 'N_w': N_w, 'N_turbine': N_turbine, 'f_w': f_w};
		N_arr = np.arange(5, 45, 10, dtype = int);
		L_w_arr = np.linspace(2*R, 5*2*R, 5, dtype = float);
		N_w_arr = np.arange(5, 55, 10, dtype = int);
		f_w_arr = np.linspace(0.7, 1.1, 5);
		distribution_arr = np.array(['uniform', 'cosine'], dtype = object);
		iter_var_dict = {'N': N_arr, 'distribution': distribution_arr, 'L_w': L_w_arr, 'N_w': N_w_arr, 'f_w': f_w_arr};
		Results_sensitivity, names_sensitivity = Run_sensitivity_study(iter_var_dict, kwds_dict);
		Plot_sensitivity_results(Results_sensitivity, names_sensitivity, plot_variables, var_label, save_plots);
		plt.show();
	else:
		Results_L_phi, names_L, names_phi = Run_L_phi_iters_main(N, distribution, L_w, N_w, N_turbine, L, 0, phi, 1);
		Plot_results_NT(Results_L_phi, names_L, names_phi, plot_variables, var_label, save_plots);
		Induced_velocity_matrix(Results_L_phi['Turbines'][list(Results_L_phi['Turbines'].keys())[0]], Results_L_phi['wakes'][list(Results_L_phi['Turbines'].keys())[0]], plot = True, save_plot = save_plots);
		Plot_N_Turbine_Wake_system(Results_L_phi, names_phi, 10, save_plots);
		plt.show();