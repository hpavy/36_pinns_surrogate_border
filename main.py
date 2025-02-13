import torch
from run import RunSimulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le code se lance sur {device}")


folder_result_name = "13_piche"  # name of the result folder

# On utilise hyper_param_init uniquement si c'est un nouveau modèle

hyper_param_init = {
    "H": [
        261.39,
    ],
    "ya0": [
        0.00125,
    ],
    "m": 1.57,
    "file": [
        "model_1_case_1.csv",
    ],
    "nb_epoch": 1000,
    "save_rate": 20,
    "dynamic_weights": False,
    "lr_weights": 0.1,
    "weight_data": 0.33,
    "weight_pde": 0.33,
    "weight_border": 0.33,
    "batch_size": 10000,
    "nb_points_pde": 1000000,
    "Re": 100,
    "lr_init": 3e-4,
    "gamma_scheduler": 0.999,
    "nb_layers": 10,
    "nb_neurons": 64,
    "n_pde_test": 5000,
    "n_data_test": 5000,
    "nb_points": 144,
    "x_min": -0.06,
    "x_max": 0.06,
    "y_min": -0.06,
    "y_max": 0.06,
    "t_min": 6.5,
    "nb_period": 20,
    "nb_period_plot": 2,
    "force_inertie_bool": False,
    "nb_period": 20,
    "u_border": True,
    "v_border": False,
    "p_border": True,
    "r_min": 0.026/2,
    'theta_border_min': 0.1
}

param_adim = {"V": 1.0, "L": 0.025, "rho": 1.2}

simu = RunSimulation(hyper_param_init, folder_result_name, param_adim)

simu.run()
