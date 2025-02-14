import torch
from run import RunSimulation
from constants import DICT_Y0, DICT_CASE, PARAM_ADIM, M

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le code se lance sur {device}")


folder_result_name = "17_huge_inertia"  # name of the result folder

# On utilise hyper_param_init uniquement si c'est un nouveau modèle

hyper_param_init = {
    "num": [2, 3, 5, 6, 8, 9, 11, 12, 14, 2, 3, 5, 6, 8, 9, 11, 12, 14],
    "case": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,],
    "nb_epoch": 1000,
    "save_rate": 5,
    "dynamic_weights": False,
    "lr_weights": 0.1,
    "weight_data": 0.45,
    "weight_pde": 0.1,
    "weight_border": 0.45,
    "batch_size": 10000,
    "nb_points_pde": 1000000,
    "Re": 100,
    "lr_init": 3e-4,
    "gamma_scheduler": 0.999,
    "nb_layers": 15,
    "nb_neurons": 64,
    "n_pde_test": 2500,
    "n_data_test": 5000,
    "nb_points": 144,
    "x_min": -0.06,
    "x_max": 0.06,
    "y_min": -0.06,
    "y_max": 0.06,
    "t_min": 6.5,
    "nb_period": 20,
    "nb_period_plot": 2,
    "force_inertie_bool": True,
    "nb_period": 20,
    "u_border": True,
    "v_border": False,
    "p_border": True,
    "r_min": 0.026/2,
    'theta_border_min': 0.1
}

hyper_param_init['H'] = [DICT_CASE[str(k)] for k in hyper_param_init['case']]
hyper_param_init['ya0'] = [DICT_Y0[str(k)] for k in hyper_param_init['num']]
hyper_param_init['file'] = [
    f"model_{num_}_case_{case_}.csv" for num_, case_ in zip(hyper_param_init['num'], hyper_param_init['case'])
    ]
hyper_param_init['m'] = M

simu = RunSimulation(hyper_param_init, folder_result_name, PARAM_ADIM)

simu.run()
