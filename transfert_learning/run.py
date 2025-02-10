from geometry import RectangleWithoutCylinder
import torch
from transfert_learning.utils import charge_data, init_model
from transfert_learning.train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np
import json


class RunSimulation:
    def __init__(self, hyper_param_transfert, folder_result_name, param_adim, mean_std, model):
        self.hyper_param = hyper_param_transfert
        self.time_start = time.time()
        self.folder_result_name = folder_result_name
        self.folder_result = "results/" + folder_result_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_adim = param_adim
        self.nb_simu = len(self.hyper_param["file"])
        self.mean_std = mean_std
        self.model = model

    def run(self):
        # Charging the model
        # Creation du dossier de result
        Path(self.folder_result).mkdir(parents=True, exist_ok=True)
        (
            X_entry,
            U_entry,
            X_border,
            X_border_test,
            X_pde,
            X_test_pde,
            X_test_data,
            U_test_data,
        ) = charge_data(self.hyper_param, self.param_adim, self.mean_std)

        # Initialiser le modèle

        # On plot les print dans un fichier texte
        with open(self.folder_result + "/print.txt", "a") as f:
            model, optimizer, loss, train_loss, test_loss = (
                init_model(self.hyper_param, self.folder_result, self.model)
            )
            train(
                nb_epoch=1000,
                train_loss=train_loss,
                test_loss=test_loss,
                model=model,
                loss=loss,
                optimizer=optimizer,
                X_pde=X_pde,
                X_test_pde=X_test_pde,
                X_test_data=X_test_data,
                U_test_data=U_test_data,
                Re=self.hyper_param["Re"],
                time_start=self.time_start,
                f=f,
                folder_result=self.folder_result,
                save_rate=self.hyper_param["save_rate"],
                batch_size=self.hyper_param["batch_size"],
                X_border=X_border,
                X_border_test=X_border_test,
                param_adim=self.param_adim,
                force_inertie_bool=self.hyper_param["force_inertie_bool"],
                mean_std=self.mean_std,
                nb_simu=1,
                X_entry=X_entry,
                U_entry=U_entry
            )
