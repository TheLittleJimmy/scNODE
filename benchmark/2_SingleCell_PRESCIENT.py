'''
Description:
    Run PRESCIENT on single-cell data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from datetime import datetime
import torch
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../baseline/")
sys.path.append("../baseline/prescient_model/")
sys.path.append("../baseline/prescient_model/prescient")
from plotting.visualization import plotPredAllTime, plotPredTestTime
from plotting.PlottingUtils import umapWithPCA
from plotting.Compare_SingleCell_Predictions import globalEvaluation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedPRESCIENTPars, getTimePointIndex
from baseline.prescient_model.process_data import main as prepare_data
from baseline.prescient_model.running import prescientTrain, prescientSimulate

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

def main(split_type):
    # ======================================================
    
    # Load data
    print("=" * 70)
    # Specify the dataset: zebrafish, drosophila, wot
    # Representing ZB, DR, SC, repectively
    data_name= "zebrafish" # zebrafish, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    # Specify the type of prediction tasks: three_interpolation, two_forecasting, three_forecasting, remove_recovery
    # The tasks feasible for each dataset:
    #   zebrafish (ZB): three_interpolation, two_forecasting, remove_recovery
    #   drosophila (DR): three_interpolation, three_forecasting, remove_recovery
    #   wot (SC): three_interpolation, three_forecasting, remove_recovery
    # They denote easy, medium, and hard tasks respectively.
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps, unique_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    data = ann_data.X
    all_tps = unique_tps  # Use actual time points from data

    processed_data = ann_data[
        [t for t in range(ann_data.shape[0]) if ann_data.obs["tp"].values[t] in train_tps]
    ]
    # Convert actual tp values to 0-based consecutive for PRESCIENT compatibility
    unique_train_tps = sorted(list(set(processed_data.obs["tp"].values)))
    tp_to_index = {tp: i for i, tp in enumerate(unique_train_tps)}
    cell_tps = np.array([tp_to_index[tp] for tp in processed_data.obs["tp"].values])
    cell_types = np.repeat("NAN", processed_data.shape[0])
    genes = processed_data.var.index.values

    # Use tuned hyperparameters
    k_dim, layers, sd, tau, clip = tunedPRESCIENTPars(data_name, split_type)

    # PCA and construct data dict (PRESCIENT expects 0-based consecutive indices)
    data_dict, scaler, pca, um = prepare_data(
        processed_data.X, cell_tps, cell_types, genes,
        num_pcs=k_dim, num_neighbors_umap=10
    )
    # Convert train_tps to PRESCIENT indices
    train_tps_prescient = [tp_to_index[t] for t in train_tps if t in tp_to_index]
    max_prescient_tp = max(train_tps_prescient) if train_tps_prescient else 0
    data_dict["y"] = list(range(max_prescient_tp + 1))
    data_dict["x"] = [data_dict["x"][train_tps_prescient.index(t)] if t in train_tps_prescient else [] for t in data_dict["y"]]
    data_dict["xp"] = [data_dict["xp"][train_tps_prescient.index(t)] if t in train_tps_prescient else [] for t in data_dict["y"]]
    data_dict["w"] = [data_dict["w"][train_tps_prescient.index(t)] if t in train_tps_prescient else [] for t in data_dict["y"]]

    # ======================================================
    # Model training
    train_epochs = 2000
    train_lr = 1e-3
    final_model, best_state_dict, config, loss_list = prescientTrain(
        data_dict, data_name=data_name, out_dir="", train_t=train_tps_prescient[1:], timestamp=timestamp,
        k_dim=k_dim, layers=layers, train_epochs=train_epochs, train_lr=train_lr,
        train_sd=sd, train_tau=tau, train_clip=clip
    )

    # Simulation
    n_sim_cells = 2000
    sim_data = prescientSimulate(
        data_dict,
        data_name=data_name,
        best_model_state=best_state_dict,
        num_cells=n_sim_cells,
        num_steps= (len(unique_tps) - 1)*10, # (#tps - 1) * 10, as dt=0.1
        config=config
    )
    sim_tp_latent = [sim_data[int(t * 10)] for t in range(len(unique_tps))]  # dt=0.1 in all cases
    sim_tp_recon = [scaler.inverse_transform(pca.inverse_transform(each)) for each in sim_tp_latent]

    # ======================================================
    # Visualization - 2D UMAP embeddings
    traj_data = [ann_data.X[ann_data.obs["tp"].values == t] for t in unique_tps]  # Use actual tp values
    all_recon_obs = sim_tp_recon
    print("Compare true and reconstructed data...")
    true_data = traj_data
    # Use actual time point labels for visualization
    true_cell_tps = np.concatenate([np.repeat(unique_tps[t], each.shape[0]) for t, each in enumerate(true_data)])
    pred_cell_tps = np.concatenate([np.repeat(unique_tps[t], all_recon_obs[t].shape[0]) for t in range(len(all_recon_obs))])
    reorder_pred_data = all_recon_obs

    true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
    pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))

    save_dir = f"/project/Stat/s1155202253/myproject/babydev/benchmark_zebrafish_results/{split_type}/PRESCIENT"
    os.makedirs(save_dir, exist_ok=True)
    plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps,
                    save_path=f"{save_dir}/{data_name}-{split_type}-PRESCIENT-UMAP-all-time.png")
    plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps,
                     save_path=f"{save_dir}/{data_name}-{split_type}-PRESCIENT-UMAP-test-time.png")

    # Compute evaluation metrics
    print("Compute metrics...")
    test_tps_list = [int(t) for t in test_tps]
    for t in test_tps_list:
        print("-" * 70)
        print("t = {}".format(t))  # t is actual time point value
        # -----
        # Convert time point to array index
        try:
            array_idx = getTimePointIndex(t, unique_tps)
            pred_global_metric = globalEvaluation(traj_data[array_idx], reorder_pred_data[array_idx])
            print(pred_global_metric)
        except ValueError as e:
            print(f"Skipping time point {t}: {e}")

    # ======================================================
    # Save results
    # Note: save_dir already updated above with proper task/method structure
    res_filename = "{}/{}-{}-PRESCIENT-res.npy".format(save_dir, data_name, split_type)
    print("Saving to {}".format(res_filename))
    # Save all cell categories
    traj_data = [ann_data.X[ann_data.obs["tp"].values == t] for t in unique_tps]  # Use actual tp values
    train_data = [traj_data[getTimePointIndex(t, unique_tps)] for t in train_tps]  # Convert to array index
    test_data = [traj_data[getTimePointIndex(t, unique_tps)] for t in test_tps]    # Convert to array index
    res_dict = {
        "true": traj_data,  # all time points cells
        "train": train_data,  # training cells
        "test": test_data,    # testing cells
        "pred": sim_tp_recon,  # predicted cells
        "latent_seq": sim_tp_latent,
        "tps": {"all": unique_tps, "train": train_tps, "test": test_tps}
    }
    res_dict["true_pca"] = [pca.transform(scaler.transform(each)) for each in res_dict["true"]]
    np.save(res_filename, res_dict, allow_pickle=True)

    # save model and config
    model_dir = "{}/{}-{}-PRESCIENT-state_dict.pt".format(save_dir, data_name, split_type)
    config_dir = "{}/{}-{}-PRESCIENT-config.pt".format(save_dir, data_name, split_type)
    torch.save(best_state_dict['model_state_dict'], model_dir)
    torch.save(config.__dict__, config_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark PRESCIENT on zebrafish dataset')
    parser.add_argument('--task', type=str, required=True,
                        choices=['three_interpolation', 'three_forecasting', 'two_forecasting', 'remove_recovery'],
                        help='Task type for benchmarking')
    
    args = parser.parse_args()
    main(args.task)
