#!/usr/bin/env python3

import argparse
from benchmarker.core import results_loader
from benchmarker.core import plotter
from pathlib import Path
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from benchmarker.core import results_loader, instance
from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import json

def setup_style(fontsize=15,scale=1.0,grid=True):
    fig_width_pt = 246.0  # width in pt (e.g., for single-column in a paper)
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    eps_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "figure.figsize": fig_size,
        "axes.titlesize": fontsize,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": 'black',
    }
    mpl.rcParams.update(eps_with_latex)

def plot_tts(systems: List[int] = [1,2,5,6,7], file_limit: int = 20, num_reps=0):
    """Plot time-to-solution comparisons for multiple systems."""

    ta = 200
    setup_style(fontsize=15, scale=1.2, grid=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10.colors  # Use tab10 colormap for better consistency
    
    ALL_dfs = []
    loader = results_loader.ResultsLoader(Path("/Users/kuba/Code/Work/dynamics-benchmark/benchmarker/data/results/hessian"))

    for _, system in tqdm(enumerate(systems), total=len(systems)):
        velox_tts = loader.get_velox_tts(system)
        dwave_14_tts = loader.get_dwave_tts(system, topology='1.6', file_limit=file_limit, ta=200,num_reps=num_reps)
        dwave_64_tts = loader.get_dwave_tts(system, topology='6.4', file_limit=file_limit, ta=200,num_reps=num_reps)
        # neal_tts = loader.get_dwave_tts(system, topology='neal', file_limit=file_limit, num_reps=num_reps)
        neal_tts = loader.get_velox_tts(system, solver_name='neal')
        sa_gpu_tts = loader.get_velox_tts(system, solver_name='SA_GPU')
        ALL_dfs.append(velox_tts)
        ALL_dfs.append(dwave_14_tts)
        ALL_dfs.append(dwave_64_tts)
        ALL_dfs.append(neal_tts)
        ALL_dfs.append(sa_gpu_tts)

    sources = ['VELOX', '1.6', '6.4','NEAL', 'SA_GPU']
    linestyles = ['-', '-', '-','-','-']
    all_system_dfs = pd.concat(ALL_dfs, axis=0)
    native_system_df = all_system_dfs[all_system_dfs.system.isin(systems)]

    for i, source in enumerate(sources):
        native_system_df_filtered = native_system_df[native_system_df.source == source]
        
        # Group by num_var and calculate mean and std for tts99
        grouped = native_system_df_filtered.groupby('num_var')['tts99'].agg(['mean', 'std', 'count']).reset_index()
        # Filter out NaN/infinite values
        mask = np.isfinite(grouped['mean'])
        grouped_clean = grouped[mask]
        
        num_var_clean = np.array(grouped_clean['num_var'])
        tts99_mean = np.array(grouped_clean['mean'])
        tts99_std = np.array(grouped_clean['std'])
        
        # Handle cases where std is NaN (only one data point)
        tts99_std = np.where(np.isnan(tts99_std), 0, tts99_std)
        
        # Plot points
        ax.plot(
            num_var_clean,
            tts99_mean,
            marker=['o', 's', '^','x','D'][i],
            linestyle='',
            color=colors[i],
            label=['VeloxQ', 'Adv2 1.6 (forward)', 'Adv 6.4 (forward)','SA (CPU)', 'SA (GPU)'][i],
            markersize=8,
            alpha=0.8,
        )
        
        # Fit exponential trend to averaged data
        if source != 'VELOX' and source != '1.6':
          if len(num_var_clean) > 1:
            log_tts99_mean = np.log(tts99_mean)
            slope, intercept, r_value, p_value, std_err = linregress(num_var_clean, log_tts99_mean)
            
            r_TTS99 = slope
            D = np.exp(intercept)
            
            # Generate smooth curve for fit

            num_var_fit = np.linspace(num_var_clean.min(), num_var_clean.max(), 100)
            TTS99_fit = D * np.exp(r_TTS99 * num_var_fit)
           
            ax.plot(
               num_var_fit, TTS99_fit, 
               linestyle=linestyles[i], 
               color=colors[i], 
               alpha=0.7
           )
           
           # Add equation annotation
            mid_x = 1.1 * (num_var_clean.min() + num_var_clean.max()) / 2
            mid_y = D * np.exp(r_TTS99 * mid_x)
           
            # Format equation text
            # eq_text = rf"$\propto e^{{\frac{{N}}{{{1/r_TTS99:.1f}}}}}$" if abs(r_TTS99) > 0.001 else rf"$\propto \mathrm{{const}}$"
            eq_text = rf"$\propto \exp(N/{1/r_TTS99:.2f})$" if abs(r_TTS99) > 0.001 else rf"$\propto \mathrm{{const}}$" 
            ax.annotate(
                eq_text,
                xy=(mid_x*[1,1,1.5,1.5,1.5][i], mid_y * [1.3, 0.8, 0.1,18.0, 30][i]),
                fontsize=14,
                color=colors[i],
                rotation=[0, 28,63,41,35][i],
                ha='center',
                va='bottom'
            )
        else:
          if len(num_var_clean) > 1:
            idx_small = num_var_clean < 120
            idx_large = num_var_clean >= 120
            log_tts99_mean_small = np.log(tts99_mean[idx_small])
            log_tts99_mean_large = np.log(tts99_mean[idx_large])
            num_var_clean_small = num_var_clean[idx_small]
            num_var_clean_large = num_var_clean[idx_large]
            slope_small, intercept_small, r_value_small, p_value_small, std_err_small = linregress(num_var_clean_small, log_tts99_mean_small)
            slope_large, intercept_large, r_value_large, p_value_large, std_err_large = linregress(num_var_clean_large, log_tts99_mean_large)

            
            r_TTS99_small = slope_small
            r_TTS99_large = slope_large
            D_small = np.exp(intercept_small)
            D_large = np.exp(intercept_large)
            
            # Generate smooth curve for fit

            num_var_fit_small = np.linspace(num_var_clean.min(), num_var_clean[:-3].max(), 100)
            num_var_fit_large = np.linspace(num_var_clean.min(), num_var_clean.max(), 100)
            TTS99_fit_small = D_small * np.exp(r_TTS99_small * num_var_fit_small)
            TTS99_fit_large = D_large * np.exp(r_TTS99_large * num_var_fit_large)
           
            ax.plot(
               num_var_fit_small, TTS99_fit_small,
               linestyle='--', 
               color=colors[i], 
               alpha=0.7
           )
            ax.plot(
               num_var_fit_large, TTS99_fit_large,
               linestyle=linestyles[i], 
               color=colors[i], 
               alpha=0.7
           )

            eq_text_small = rf"$\propto \exp(N/{1/r_TTS99_small:.2f})$" if abs(r_TTS99_small) > 0.001 else rf"$\propto \mathrm{{const}}$" 
            eq_text_large = rf"$\propto \exp(N/{1/r_TTS99_large:.2f})$" if abs(r_TTS99_large) > 0.001 else rf"$\propto \mathrm{{const}}$"

  
            pos1 = (0.17, 0.01) if source == 'VELOX' else (0.17, 0.35)
            rot1 = 41 if source == 'VELOX' else 46
            ax.annotate(
                eq_text_small,
                xy=pos1,
                rotation=rot1,
                fontsize=12,
                color=colors[i],
                ha='center',
                va='bottom',
                xycoords='axes fraction'
            )
            pos2 = (0.6, 0.37) if source == 'VELOX' else (0.56, 0.8)
            rot2 = 15 if source == 'VELOX' else 30
            ax.annotate(
                eq_text_large,
                xy=pos2,
                rotation=rot2,
                fontsize=14,
                color=colors[i],
                ha='center',
                va='bottom',
                xycoords='axes fraction'
            )
           

    # Load and plot cyclic data separately (already processed TTS99 vs N)
    with open('/Users/kuba/Code/Work/dynamics-benchmark/benchmarker/data/cyclic/dynamics_tts_1.10_no_ancilla.json', 'r') as f:
        cyclic_16_data = json.load(f)
    with open('/Users/kuba/Code/Work/dynamics-benchmark/benchmarker/data/cyclic/dynamics_tts_6.4_no_ancilla.json', 'r') as f:
        cyclic_64_data = json.load(f)
    
    # Extract x (num_var) and y (tts99) from cyclic data
    cyclic_16_x = np.array([point['x'] for point in cyclic_16_data['data']])
    cyclic_16_y = np.array([point['y'] for point in cyclic_16_data['data']])
    cyclic_64_x = np.array([point['x'] for point in cyclic_64_data['data']])
    cyclic_64_y = np.array([point['y'] for point in cyclic_64_data['data']])
    
    # Plot cyclic data using same colors as forward data (indices 1 and 2)
    ax.plot(
        cyclic_16_x,
        cyclic_16_y,
        marker='s',
        linestyle='',
        color=colors[1],  # Same color as 1.6 forward
        label='Adv2 1.10 (cyclic)',
        markersize=8,
        alpha=0.8,
        fillstyle='none'  # Hollow markers to distinguish from forward
    )
    
    ax.plot(
        cyclic_64_x,
        cyclic_64_y,
        marker='^',
        linestyle='',
        color='seagreen',
        label='Adv 6.4 (cyclic)',
        markersize=8,
        alpha=0.8,
        fillstyle='none'  # Hollow markers to distinguish from forward
    )
    
    # Add exponential fits for cyclic data
    if len(cyclic_64_x) > 1:
        log_cyclic_64_y = np.log(cyclic_64_y)
        slope_64, intercept_64, r_value_64, p_value_64, std_err_64 = linregress(cyclic_64_x, log_cyclic_64_y)
        
        r_TTS99_64 = slope_64
        D_64 = np.exp(intercept_64)
        
        # Generate smooth curve for fit
        num_var_fit_64 = np.linspace(cyclic_64_x.min(), cyclic_64_x.max(), 100)
        TTS99_fit_64 = D_64 * np.exp(r_TTS99_64 * num_var_fit_64)
        
        ax.plot(
            num_var_fit_64, TTS99_fit_64,
            linestyle='-',
            color='seagreen',
            alpha=0.7
        )
        
        # Add equation annotation
        mid_x_64 = 1.1 * (cyclic_64_x.min() + cyclic_64_x.max()) / 2
        mid_y_64 = D_64 * np.exp(r_TTS99_64 * mid_x_64)
        
        eq_text_64 = rf"$\propto \exp(N/{1/r_TTS99_64:.2f})$" if abs(r_TTS99_64) > 0.001 else rf"$\propto \mathrm{{const}}$"
        ax.annotate(
            eq_text_64,
            xy=(mid_x_64 * 0.9, mid_y_64 * 0.3),
            fontsize=12,
            color='seagreen',
            rotation=60,
            ha='center',
            va='bottom'
        )

    # Place legend inside the plot
    ax.legend(
        loc='lower right',
        ncol=2,
        fontsize=10,
        framealpha=1.0,
        handlelength=1,
        columnspacing=0.7,
        fancybox=True,
        shadow=True
    )
    
    ax.set_xlabel(r"$N$")
    # ax.set_ylabel(r"$\mathrm{TTS}_{\rm 99}$ [ms]")
    ax.set_ylabel(r"$\mathrm{T}_{S}$ [ms]")
    ax.set_yscale("log")
    #ax.set_ylim(1e1, 1e4)
    ax.grid(True)
    
    plt.tight_layout()  # No need for extra space since legend is inside
    systems_str = ''.join(map(str, systems))
    plt.savefig(f"tta_overview_{systems_str}.pdf", bbox_inches="tight")


# systems = [1,2,3,4,5,6,7,8]
# for system in systems:
  # pltr = plotter.BenchmarkPlotter(output_dir=Path("plots"))
  # pltr.plot_tts(file_limit=20,num_reps=1000, systems=[system])

plot_tts(systems=[1,2,5,6,7], file_limit=20, num_reps=1000)

