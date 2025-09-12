'''
Description:
    Run MIOFlow on single-cell data.
'''
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', message='findfont: Generic family.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure matplotlib fonts
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
})

import numpy as np
import pandas as pd
import scanpy
import torch
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../baseline/MIOFlow_model_revised")

from baseline.MIOFlow_model_revised.running import trainModel, makeSimulation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedMIOFlowPars, getTimePointIndex
from plotting.visualization import plotPredAllTime, plotPredTestTime, computeDrift, plotStream, plotStreamByCellType
from plotting.PlottingUtils import umapWithPCA, computeLatentEmbedding
from sklearn.decomposition import PCA
from optim.evaluation import globalEvaluation

def main(split_type):
    # ======================================================
    
    print("=" * 70)
    # Specify the dataset: zebrafish, drosophila, wot
    # Representing ZB, DR, SC, repectively
    data_name= "zebrafish"
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
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("Train tps={}".format(train_tps))
    print("Test tps={}".format(test_tps))

    data = ann_data.X
    true_data = [data[np.where(cell_tps == t)[0], :] for t in unique_tps]  # (# tps, # cells, # genes)
    true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
    # Convert time points to array indices
    train_data = [true_data[getTimePointIndex(t, unique_tps)] for t in train_tps]
    test_data = [true_data[getTimePointIndex(t, unique_tps)] for t in test_tps]
    print("All shape: ", [each.shape[0] for each in true_data])
    print("Train shape: ", [each.shape[0] for each in train_data])
    print("Test shape: ", [each.shape[0] for each in test_data])
    # Convert to principal components w/ training timepoints
    print("PCA...")
    n_pcs = 50
    pca_model = PCA(n_components=n_pcs, svd_solver="arpack")
    data_pca = pca_model.fit_transform(np.concatenate(train_data, axis=0))
    # Convert train_tps to 0-based consecutive indices for MIOFlow compatibility
    train_indices = [getTimePointIndex(t, unique_tps) for t in train_tps]
    train_tps_0based = list(range(len(train_indices)))
    time_df = pd.DataFrame(
        data=np.concatenate([np.repeat(train_tps_0based[t], x.shape[0]) for t, x in enumerate(train_data)])[:, np.newaxis],
        columns=["samples"]
    )
    data_df = pd.DataFrame(data=data_pca)
    train_df = pd.concat([time_df, data_df], axis=1).reset_index(drop=True)
    n_genes = 50
    # ======================================================
    # Model training
    print("=" * 70)
    print("Model Training...")
    gae_embedded_dim, encoder_layers, layers, lambda_density = tunedMIOFlowPars(data_name, split_type)
    # Convert test_tps to indices relative to unique_tps
    test_indices = [getTimePointIndex(t, unique_tps) for t in test_tps]
    test_tps_0based = test_indices
    model, gae_losses, local_losses, batch_losses, globe_losses, opts = trainModel(
        train_df, train_tps_0based, test_tps_0based, n_genes=n_genes, n_epochs_emb=100, samples_size_emb=(50,),
        gae_embedded_dim=gae_embedded_dim, encoder_layers=encoder_layers,
        layers=layers, lambda_density=lambda_density,
        batch_size=100, n_local_epochs=40, n_global_epochs=40, n_post_local_epochs=0
    )
    # Model prediction
    print("=" * 70)
    print("Model Predicting...")
    # MIOFlow expects 0-based consecutive indices for simulation
    tps_0based = list(range(len(unique_tps)))
    generated = makeSimulation(train_df, model, tps_0based, opts, n_sim_cells=2000, n_trajectories=100, n_bins=100)
    forward_recon_traj = [pca_model.inverse_transform(generated[i, :, :]) for i in range(generated.shape[0])]
    # ======================================================
    # Visualization - 2D UMAP embeddings
    print("Compare true and reconstructed data...")
    # Use actual time point labels for visualization
    true_cell_tps = np.concatenate([np.repeat(unique_tps[t], each.shape[0]) for t, each in enumerate(true_data)])
    pred_cell_tps = np.concatenate(
        [np.repeat(unique_tps[t], forward_recon_traj[t].shape[0]) for t in range(len(forward_recon_traj))]
    )
    reorder_pred_data = forward_recon_traj
    true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
    pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
    save_dir = f"/project/Stat/s1155202253/myproject/babydev/benchmark_zebrafish_results/{split_type}/MIOFlow"
    os.makedirs(save_dir, exist_ok=True)
    plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps,
                    save_path=f"{save_dir}/{data_name}-{split_type}-MIOFlow-UMAP-all-time.png")
    plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps,
                     save_path=f"{save_dir}/{data_name}-{split_type}-MIOFlow-UMAP-test-time.png")

    # Compute metric for testing time points
    print("Compute metrics...")
    test_tps_list = [int(t) for t in test_tps]
    for t in test_tps_list:
        print("-" * 70)
        print("t = {}".format(t))  # t is actual time point value
        # -----
        # Convert time point to array index
        try:
            array_idx = getTimePointIndex(t, unique_tps)
            pred_global_metric = globalEvaluation(true_data[array_idx], reorder_pred_data[array_idx])
            print(pred_global_metric)
        except ValueError as e:
            print(f"Skipping time point {t}: {e}")


    # ======================================================
    # Save results
    # Note: save_dir already updated above with proper task/method structure
    res_filename = "{}/{}-{}-MIOFlow-res.npy".format(save_dir, data_name, split_type)
    state_filename = "{}/{}-{}-MIOFlow-state_dict.pt".format(save_dir, data_name, split_type)
    print("Saving to {}".format(res_filename))
    # Save all cell categories
    # Use actual time points for saving
    tps_actual = unique_tps
    res_dict = {
        "true": true_data,  # all time points cells
        "train": train_data,  # training cells
        "test": test_data,    # testing cells
        "pred": forward_recon_traj,  # predicted cells
        "tps": {"all": tps_actual, "train": train_tps, "test": test_tps},
        "gae_losses": gae_losses,
        "local_losses": local_losses,
        "batch_losses": batch_losses,
        "globe_losses": globe_losses,
    }
    np.save(res_filename, res_dict, allow_pickle=True)
    torch.save(model.state_dict(), state_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark MIOFlow on zebrafish dataset')
    parser.add_argument('--task', type=str, required=True,
                        choices=['three_interpolation', 'three_forecasting', 'two_forecasting', 'remove_recovery'],
                        help='Task type for benchmarking')
    
    args = parser.parse_args()
    main(args.task)
