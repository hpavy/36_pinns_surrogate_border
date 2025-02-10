from model import pde
import torch
import time
from utils import write_csv
from pathlib import Path
from torch.utils.data import DataLoader
from utils import CustomDataset


def train(
    nb_epoch,
    train_loss,
    test_loss,
    model,
    loss,
    optimizer,
    X_test_pde,
    X_test_data,
    U_test_data,
    X_pde,
    Re,
    time_start,
    f,
    folder_result,
    save_rate,
    batch_size,
    X_border,
    X_border_test,
    mean_std,
    param_adim,
    nb_simu,
    force_inertie_bool,
    X_entry,
    U_entry
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_it_tot = nb_epoch + len(train_loss["total"])
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}"
        + "\n--------------------------"
    )
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}\n------------"
        + "--------------",
        file=f,
    )

    if device == torch.device("cuda"):
        stream_pde = torch.cuda.Stream()
        stream_border = torch.cuda.Stream()

    nb_batches = len(X_pde) // batch_size
    # batch_size = torch.tensor(batch_size, device=device, dtype=torch.int64)

    Re = torch.tensor(Re, dtype=torch.float32, device=device)
    ya0_mean = torch.tensor(mean_std["ya0_mean"]).clone().to(device)
    ya0_std = torch.tensor(mean_std["ya0_std"]).clone().to(device)
    w0_mean = torch.tensor(mean_std["w0_mean"]).clone().to(device)
    w0_std = torch.tensor(mean_std["w0_std"]).clone().to(device)
    x_std = torch.tensor(mean_std["x_std"]).clone().to(device)
    y_std = torch.tensor(mean_std["y_std"]).clone().to(device)
    u_mean = torch.tensor(mean_std["u_mean"]).clone().to(device)
    v_mean = torch.tensor(mean_std["v_mean"]).clone().to(device)
    p_std = torch.tensor(mean_std["p_std"]).clone().to(device)
    t_std = torch.tensor(mean_std["t_std"]).clone().to(device)
    t_mean = torch.tensor(mean_std["t_mean"]).clone().to(device)
    u_std = torch.tensor(mean_std["u_std"]).clone().to(device)
    v_std = torch.tensor(mean_std["v_std"]).clone().to(device)
    L_adim = torch.tensor(param_adim["L"], device=device, dtype=torch.float32)
    V_adim = torch.tensor(param_adim["V"], device=device, dtype=torch.float32)
    X_border = X_border.to(device)
    X_border_test = X_border_test.to(device).detach()
    X_entry = X_entry.to(device)
    U_entry = U_entry.to(device)

    ########
    X_pde = X_pde.to(device)
    ########
    # dataset = CustomDataset(X_pde)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    ###############
    X_test_pde = X_test_pde.to(device)
    X_test_data = X_test_data.to(device).detach()
    U_test_data = U_test_data.to(device).detach()
    nb_simu = torch.tensor(nb_simu, device=device, dtype=torch.int64)
    model = model.to(device)

    for epoch in range(len(train_loss["total"]), nb_it_tot):
        time_start_batch = time.time()
        total_batch = torch.tensor([0.0], device=device)
        entry_batch = torch.tensor([0.0], device=device)
        pde_batch = torch.tensor([0.0], device=device)
        border_batch = torch.tensor([0.0], device=device)

        # Pour le test :
        model.eval()

        # loss du pde
        test_pde = model(X_test_pde)
        test_pde1, test_pde2, test_pde3 = pde(
            test_pde,
            X_test_pde,
            Re=Re,
            x_std=x_std,
            y_std=y_std,
            u_mean=u_mean,
            v_mean=v_mean,
            p_std=p_std,
            t_std=t_std,
            t_mean=t_mean,
            u_std=u_std,
            v_std=v_std,
            ya0_mean=ya0_mean,
            ya0_std=ya0_std,
            w0_mean=w0_mean,
            w0_std=w0_std,
            L_adim=L_adim,
            V_adim=V_adim,
            force_inertie_bool=force_inertie_bool,
        )
        with torch.no_grad():
            loss_test_pde = (
                torch.mean(test_pde1**2)
                + torch.mean(test_pde2**2)
                + torch.mean(test_pde3**2)
            )
            # loss de la data
            test_data = model(X_test_data)
            loss_test_data = loss(U_test_data, test_data)  # (MSE)

            # loss des bords
            pred_border_test = model(X_border_test)
            goal_border_test = torch.tensor(
                [
                    -mean_std["u_mean"] / mean_std["u_std"],
                    -mean_std["v_mean"] / mean_std["v_std"],
                ],
                dtype=torch.float32,
                device=device,
            ).expand(pred_border_test.shape[0], 2)
            loss_test_border = loss(pred_border_test[:, :2], goal_border_test)  # (MSE)

            # loss totale
            loss_test = (
                0.5 * loss_test_pde
                + 0.5 * loss_test_border
            )
            test_loss["total"].append(loss_test.item())
            test_loss["data"].append(loss_test_data.item())
            test_loss["pde"].append(loss_test_pde.item())
            test_loss["border"].append(loss_test_border.item())

        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}"
        )
        print(
            f"Test  : loss: {test_loss['total'][-1]:.3e}, data: {test_loss['data'][-1]:.3e}, pde: {test_loss['pde'][-1]:.3e}, border: {test_loss['border'][-1]:.3e}",
            file=f,
        )

        model.train()  # on dit qu'on va entrainer (on a le dropout)
        for nb_batch in range(nb_batches):
            with torch.cuda.stream(stream_pde):
                X_pde_batch = (
                    X_pde[nb_batch * batch_size: (nb_batch + 1) * batch_size, :]
                    .clone()
                    .requires_grad_(True)
                ).to(device)
                pred_pde = model(X_pde_batch)
                pred_pde1, pred_pde2, pred_pde3 = pde(
                    pred_pde,
                    X_pde_batch,
                    Re=Re,
                    x_std=x_std,
                    y_std=y_std,
                    u_mean=u_mean,
                    v_mean=v_mean,
                    p_std=p_std,
                    t_std=t_std,
                    t_mean=t_mean,
                    u_std=u_std,
                    v_std=v_std,
                    ya0_mean=ya0_mean,
                    ya0_std=ya0_std,
                    w0_mean=w0_mean,
                    w0_std=w0_std,
                    L_adim=L_adim,
                    V_adim=V_adim,
                    force_inertie_bool=force_inertie_bool,
                )
                loss_pde = (
                    torch.mean(pred_pde1**2)
                    + torch.mean(pred_pde2**2)
                    + torch.mean(pred_pde3**2)
                )

            with torch.cuda.stream(stream_border):
                # loss du border
                pred_border = model(X_border)
                goal_border = torch.tensor(
                    [
                        -mean_std["u_mean"] / mean_std["u_std"],
                        -mean_std["v_mean"] / mean_std["v_std"],
                    ],
                    dtype=torch.float32,
                    device=device,
                ).expand(pred_border.shape[0], 2)
                loss_border_cylinder = loss(pred_border[:, :2], goal_border)  # (MSE)
                X_entry_batch = (
                    X_entry[nb_batch * batch_size: (nb_batch + 1) * batch_size, :]
                    .clone()
                    .requires_grad_(True)
                ).to(device)
                pred_entry = model(X_entry_batch)
                U_entry_batch = (
                    U_entry[nb_batch * batch_size: (nb_batch + 1) * batch_size, :]
                    .clone()
                    .requires_grad_(True)
                ).to(device)
                loss_entry = torch.mean((U_entry_batch[:, 0]-pred_entry[:, 0])**2)
            torch.cuda.synchronize()

            loss_totale = (
                  0.9 * loss_pde
                + 0.05 * loss_border_cylinder
                + 0.05 * loss_entry
            )

            # Backpropagation
            loss_totale.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                total_batch += loss_totale.item()
                pde_batch += loss_pde.item()
                border_batch += loss_border_cylinder.item()
                entry_batch += loss_entry.item()
      
        # Weights
        with torch.no_grad():
            total_batch /= nb_batch
            pde_batch /= nb_batch
            border_batch /= nb_batch
            entry_batch /= nb_batch
            train_loss["total"].append(total_batch.item())
            train_loss["pde"].append(pde_batch.item())
            train_loss["border"].append(border_batch.item())

        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
        print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, entry: {entry_batch.item():.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}"
        )
        print(
            f"Train : loss: {train_loss['total'][-1]:.3e}, entry: {entry_batch.item():.3e}, pde: {train_loss['pde'][-1]:.3e}, border: {train_loss['border'][-1]:.3e}",
            file=f,
        )

        print(f"time: {time.time()-time_start:.0f}s")
        print(f"time: {time.time()-time_start:.0f}s", file=f)

        print(f"time_epoch: {time.time()-time_start_batch:.0f}s")
        print(f"time: {time.time()-time_start_batch:.0f}s", file=f)

        if (epoch + 1) % save_rate == 0:
            with torch.no_grad():
                dossier_midle = Path(
                    folder_result + f"/epoch{len(train_loss['total'])}"
                )
                dossier_midle.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    folder_result
                    + f"/epoch{len(train_loss['total'])}"
                    + "/model_weights.pth",
                )

                write_csv(
                    train_loss,
                    folder_result + f"/epoch{len(train_loss['total'])}",
                    file_name="/train_loss.csv",
                )
                write_csv(
                    test_loss,
                    folder_result + f"/epoch{len(train_loss['total'])}",
                    file_name="/test_loss.csv",
                )
