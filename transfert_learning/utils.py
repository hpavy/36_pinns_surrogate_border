import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from geometry import RectangleWithoutCylinder


def init_model(hyper_param, folder_result, model):
    optimizer = optim.Adam(model.final_layer.parameters(), lr=hyper_param["lr"])
    loss = nn.MSELoss()
    # On regarde si notre modèle n'existe pas déjà
    train_loss = {"total": [], "pde": [], "border": []}
    test_loss = {"total": [], "data": [], "pde": [], "border": []}
    return model, optimizer, loss, train_loss, test_loss


def charge_data(hyper_param, param_adim, mean_std):
    """
    Charge the data of X_full, U_full with every points
    And X_train, U_train with less points
    """
    # La data
    # On adimensionne la data

    nb_simu = len(hyper_param["file"])
    x_full, y_full, t_full, ya0_full, w0_full = [], [], [], [], []
    u_full, v_full, p_full = [], [], []
    x_norm_full, y_norm_full, t_norm_full, ya0_norm_full, w0_norm_full = (
        [],
        [],
        [],
        [],
        [],
    )
    u_norm_full, v_norm_full, p_norm_full = [], [], []
    H_numpy = np.array(hyper_param["H"])
    f_numpy = 0.5 * (H_numpy / hyper_param["m"]) ** 0.5
    f = np.min(f_numpy)
    time_tot = hyper_param["nb_period_plot"] / f  # la fréquence de l'écoulement
    t_max = hyper_param["t_min"] + hyper_param["nb_period"] * time_tot
    t_max = hyper_param["t_min"] + hyper_param["nb_period"] / f
    for k in range(nb_simu):
        df = pd.read_csv("31_pinns_surrogate_final/data/" + hyper_param["file"][k])
        df_modified = df.loc[
            (df["Points:0"] >= hyper_param["x_min"])
            & (df["Points:0"] <= hyper_param["x_max"])
            & (df["Points:1"] >= hyper_param["y_min"])
            & (df["Points:1"] <= hyper_param["y_max"])
            & (df["Time"] > hyper_param["t_min"])
            & (df["Time"] < t_max)
            & (df["Points:2"] == 0.0)
            & (df["Points:0"] ** 2 + df["Points:1"] ** 2 > (0.025 / 2) ** 2),
            :,
        ].copy()
        df_modified.loc[:, "ya0"] = hyper_param["ya0"][k]
        df_modified.loc[:, "w0"] = (
            torch.pi * (hyper_param["H"][k] / hyper_param["m"]) ** 0.5
        )

        # Adimensionnement
        x_full.append(
            torch.tensor(df_modified["Points:0"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        y_full.append(
            torch.tensor(df_modified["Points:1"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        f_flow = f_numpy[k]
        time_without_modulo = df_modified["Time"].to_numpy() - hyper_param['t_min']
        time_with_modulo = hyper_param['t_min'] + time_without_modulo % (1/f_flow)
        t_full.append(
            torch.tensor(time_with_modulo, dtype=torch.float32)
            / (param_adim["L"] / param_adim["V"])
        )
        ya0_full.append(
            torch.tensor(df_modified["ya0"].to_numpy(), dtype=torch.float32)
            / param_adim["L"]
        )
        w0_full.append(
            torch.tensor(df_modified["w0"].to_numpy(), dtype=torch.float32)
            / (param_adim["V"] / param_adim["L"])
        )
        u_full.append(
            torch.tensor(df_modified["Velocity:0"].to_numpy(), dtype=torch.float32)
            / param_adim["V"]
        )
        v_full.append(
            torch.tensor(df_modified["Velocity:1"].to_numpy(), dtype=torch.float32)
            / param_adim["V"]
        )
        p_full.append(
            torch.tensor(df_modified["Pressure"].to_numpy(), dtype=torch.float32)
            / ((param_adim["V"] ** 2) * param_adim["rho"])
        )
        print(f"fichier n°{k} chargé")

    X_full = torch.zeros((0, 5))
    U_full = torch.zeros((0, 3))
    for k in range(nb_simu):
        # Normalisation Z
        x_norm_full.append((x_full[k] - mean_std["x_mean"]) / mean_std["x_std"])
        y_norm_full.append((y_full[k] - mean_std["y_mean"]) / mean_std["y_std"])
        t_norm_full.append((t_full[k] - mean_std["t_mean"]) / mean_std["t_std"])
        ya0_norm_full.append((ya0_full[k] - mean_std["ya0_mean"]) / mean_std["ya0_std"])
        w0_norm_full.append((w0_full[k] - mean_std["w0_mean"]) / mean_std["w0_std"])
        p_norm_full.append((p_full[k] - mean_std["p_mean"]) / mean_std["p_std"])
        u_norm_full.append((u_full[k] - mean_std["u_mean"]) / mean_std["u_std"])
        v_norm_full.append((v_full[k] - mean_std["v_mean"]) / mean_std["v_std"])
        X_full = torch.cat(
            (
                X_full,
                torch.stack(
                    (
                        x_norm_full[-1],
                        y_norm_full[-1],
                        t_norm_full[-1],
                        ya0_norm_full[-1],
                        w0_norm_full[-1],
                    ),
                    dim=1,
                ),
            )
        )
        U_full = torch.cat(
            (
                U_full,
                torch.stack((u_norm_full[-1], v_norm_full[-1], p_norm_full[-1]), dim=1),
            )
        )

    # les points du bord
    teta_int = torch.linspace(0, 2 * torch.pi, hyper_param["nb_points_border"])
    X_border = torch.empty((0, 5))
    x_ = (
        (((0.025 / 2) * torch.cos(teta_int)) / param_adim["L"]) - mean_std["x_mean"]
    ) / mean_std["x_std"]
    y_ = (
        (((0.025 / 2) * torch.sin(teta_int)) / param_adim["L"]) - mean_std["y_mean"]
    ) / mean_std["y_std"]

    for nb in range(len(ya0_norm_full)):
        for time_ in torch.unique(t_norm_full[nb]):
            new_x = torch.stack(
                (
                    x_,
                    y_,
                    torch.ones_like(x_) * time_,
                    torch.ones_like(x_) * ya0_norm_full[nb][0],
                    torch.ones_like(x_) * w0_norm_full[nb][0],
                ),
                dim=1,
            )
            X_border = torch.cat((X_border, new_x))
        indices = torch.randperm(X_border.size(0))
        X_border = X_border[indices]
    print("X_border OK")

    teta_int_test = torch.linspace(0, 2 * torch.pi, 15)
    X_border_test = torch.zeros((0, 5))
    x_ = (
        (((0.025 / 2) * torch.cos(teta_int_test)) / param_adim["L"])
        - mean_std["x_mean"]
    ) / mean_std["x_std"]
    y_ = (
        (((0.025 / 2) * torch.sin(teta_int_test)) / param_adim["L"])
        - mean_std["y_mean"]
    ) / mean_std["y_std"]

    for nb in range(len(ya0_norm_full)):
        for time_ in torch.unique(t_norm_full[nb]):
            new_x = torch.stack(
                (
                    x_,
                    y_,
                    torch.ones_like(x_) * time_,
                    torch.ones_like(x_) * ya0_norm_full[nb][0],
                    torch.ones_like(x_) * w0_norm_full[nb][0],
                ),
                dim=1,
            )
            X_border_test = torch.cat((X_border_test, new_x))

    # On charge le pde
    # le domaine de résolution
    rectangle = RectangleWithoutCylinder(
        x_max=X_full[:, 0].max(),
        y_max=X_full[:, 1].max(),
        t_min=X_full[:, 2].min(),
        t_max=X_full[:, 2].max(),
        x_min=X_full[:, 0].min(),
        y_min=X_full[:, 1].min(),
        x_cyl=0,
        y_cyl=0,
        r_cyl=0.025 / 2,
        mean_std=mean_std,
        param_adim=param_adim,
    )

    X_pde = torch.empty((hyper_param["nb_points_pde"] * nb_simu, 5))
    for nb in range(len(ya0_norm_full)):
        X_pde_without_param = torch.concat(
            (
                rectangle.generate_lhs(hyper_param["nb_points_pde"]),
                ya0_norm_full[nb][0] * torch.ones(hyper_param["nb_points_pde"]).reshape(-1, 1),
                torch.ones(hyper_param["nb_points_pde"]).reshape(-1, 1)
                * w0_norm_full[nb][0],
            ),
            dim=1,
        )
        X_pde[
            nb * hyper_param["nb_points_pde"]: (nb + 1) * hyper_param["nb_points_pde"]
        ] = X_pde_without_param
    indices = torch.randperm(X_pde.size(0))
    X_pde = X_pde[indices, :].detach()
    print("X_pde OK")

    # Data test loading
    X_test_pde = torch.empty((hyper_param["n_pde_test"] * nb_simu, 5))
    for nb in range(len(ya0_norm_full)):
        X_test_pde_without_param = torch.concat(
            (
                rectangle.generate_lhs(hyper_param["n_pde_test"]),
                ya0_norm_full[nb][0] * torch.ones(hyper_param["n_pde_test"]).reshape(-1, 1),
                torch.ones(hyper_param["n_pde_test"]).reshape(-1, 1)
                * w0_norm_full[nb][0],
            ),
            dim=1,
        )

        X_test_pde[
            nb * hyper_param["n_pde_test"]: (nb + 1) * hyper_param["n_pde_test"]
        ] = X_test_pde_without_param

    indices = torch.randperm(X_test_pde.size(0))
    X_test_pde = X_test_pde[indices, :].requires_grad_(True)

    points_coloc_test = np.random.choice(
        len(X_full), hyper_param["n_data_test"], replace=False
    )
    X_test_data = X_full[points_coloc_test]
    U_test_data = U_full[points_coloc_test]

    ###### On charge la data d'entrée en vitesse   
    min_max = torch.tensor([0., X_full[:, 1].max()-X_full[:, 1].min(), X_full[:, 2].max()-X_full[:, 2].min(), 0., 0.])
    min_ = torch.tensor([X_full[:, 0].min(), X_full[:, 1].min(), X_full[:, 2].min(), X_full[:, 3][0], X_full[:, 4][0]])
    X_entry = torch.rand((hyper_param['nb_points_pde'], 5)) * min_max + min_
    U_entry = torch.ones((hyper_param['nb_points_pde'], 2))
    U_entry[:, 0] = (0.9526 - mean_std["u_mean"])/mean_std['u_std']
    U_entry[:, 1] = (0. - mean_std['v_mean'])/mean_std['v_std']
    return (
        X_entry,
        U_entry,
        X_border,
        X_border_test,
        X_pde,
        X_test_pde,
        X_test_data,
        U_test_data,
    )