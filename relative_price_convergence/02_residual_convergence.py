"""
Residual of the SOL~BTC regression as a convergence signal (Oct 2025).

Regresses SOL 5-minute % change on BTC's (SOL = -0.0003 + 1.277 BTC,
R^2 ~ 0.5 on the study data), synchronises the two tables to a continuous
5-minute grid, and regresses the best move towards the fitted relation over
the next 15 / 288 candles on the current residual.

Original file: COR_002.PY
"""

import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from scipy.stats import gaussian_kde
from scipy.stats import linregress

DB_FILE = "data/SQL001.db"
TABLE1_NAME = "Solana3"
TABLE2_NAME = 'BTC'
TABLE3_NAME = 'ETH'

def process_dfs(table_name, db_file = DB_FILE):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_returns(df, price_col='close', new_col='returns'):
    df = df.copy()
    df[new_col] = (df[price_col].pct_change())*100
    df.dropna(subset=[new_col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def synchronise_tables(df1, df2, df3=None):
    """
    Synchronizes 2 or 3 DataFrames by keeping only the rows
    with common 'date' values across all provided DataFrames.
    
    Returns the trimmed DataFrames in the same order.
    """
    # Ensure 'date' columns are datetime
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    if df3 is not None:
        df3['date'] = pd.to_datetime(df3['date'])
    
    # Find common timestamps
    common_dates = pd.Index(df1['date']).intersection(df2['date'])
    if df3 is not None:
        common_dates = common_dates.intersection(df3['date'])
    
    # Trim DataFrames
    df1_trimmed = df1[df1['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    df2_trimmed = df2[df2['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    
    if df3 is not None:
        df3_trimmed = df3[df3['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
        return df1_trimmed, df2_trimmed, df3_trimmed
    
    return df1_trimmed, df2_trimmed

def synchronise_continuous(df1, df2, df3=None, freq='5min'):
    """
    Synchronizes 2 or 3 DataFrames by matching dates and ensuring perfect continuity.
    
    Parameters:
        df1, df2, df3: pandas DataFrames with 'date' column
        freq: frequency string (default '5min')
    
    Returns:
        Trimmed, continuous DataFrames in the same order
    """
    # Ensure datetime
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    if df3 is not None:
        df3['date'] = pd.to_datetime(df3['date'])
    
    # Step 1: synchronize on common dates
    common_dates = pd.Index(df1['date']).intersection(df2['date'])
    if df3 is not None:
        common_dates = common_dates.intersection(df3['date'])
    
    df1_sync = df1[df1['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    df2_sync = df2[df2['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    if df3 is not None:
        df3_sync = df3[df3['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    
    # Step 2: ensure continuous 5-min intervals
    # Find max continuous range
    if df3 is not None:
        min_date = max(df1_sync['date'].min(), df2_sync['date'].min(), df3_sync['date'].min())
        max_date = min(df1_sync['date'].max(), df2_sync['date'].max(), df3_sync['date'].max())
    else:
        min_date = max(df1_sync['date'].min(), df2_sync['date'].min())
        max_date = min(df1_sync['date'].max(), df2_sync['date'].max())
    
    # Create ideal date range
    ideal_range = pd.date_range(start=min_date, end=max_date, freq=freq)
    
    # Keep only rows in ideal_range
    df1_cont = df1_sync[df1_sync['date'].isin(ideal_range)].sort_values('date').reset_index(drop=True)
    df2_cont = df2_sync[df2_sync['date'].isin(ideal_range)].sort_values('date').reset_index(drop=True)
    if df3 is not None:
        df3_cont = df3_sync[df3_sync['date'].isin(ideal_range)].sort_values('date').reset_index(drop=True)
        return df1_cont, df2_cont, df3_cont
    
    return df1_cont, df2_cont

def aggregate_candles(df, jump=3):
    df = df.copy().reset_index(drop=True)

    # Trim the DataFrame to make it divisible by jump
    n_trim = len(df) % jump
    if n_trim != 0:
        df = df[:-n_trim]

    # Create group indices
    groups = df.index // jump

    # Aggregate
    agg_df = df.groupby(groups).agg({
        'date': 'first',         # date of first row
        'open': 'first',         # open of first row
        'high': 'max',           # highest high
        'low': 'min',            # lowest low
        'close': 'last'          # close of last row
    }).reset_index(drop=True)

    return agg_df

def plot_regression(df1, df2, label1='Asset 1', label2='Asset 2', dTx = 'close'):
    """
    Plots a scatter plot of df1['close'] vs df2['close'], 
    fits a linear regression, and returns regression metrics.
    """
    x = df1[dTx]
    y = df2[dTx]
    
    # Run linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Predicted line
    y_pred = intercept + slope * x

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.02, label='Data points', color='blue')
    plt.plot(x, y_pred, color='red', linewidth=2, label=f'Regression line (R²={r_value**2:.3f})')
    plt.title(f"Linear Regression: {label2} vs {label1}")
    plt.xlabel(f"{label1} Close Price")
    plt.ylabel(f"{label2} Close Price")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print and return regression metrics
    metrics = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    print("\nRegression Results:")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.5f}")
    
    return metrics

def plot_regressions(df1, df2, label1='Asset 1', label2='Asset 2'):
    """
    Plots scatter, linear, and quadratic regression between close prices of two dataframes.
    Returns regression metrics for both models.
    """
    x = df1['close'].values
    y = df2['close'].values

    # ---- Linear Regression ----
    lin_slope, lin_intercept, lin_r, lin_p, lin_stderr = linregress(x, y)
    y_lin_pred = lin_intercept + lin_slope * x

    # ---- Quadratic Regression ----
    quad_coeffs = np.polyfit(x, y, 2)  # [a, b, c] for ax² + bx + c
    y_quad_pred = np.polyval(quad_coeffs, x)

    # Compute R² for quadratic model
    ss_res_quad = np.sum((y - y_quad_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot

    # ---- Plot ----
    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, alpha=0.6, label='Data points')
    
    # Sort for smooth curve plotting
    sort_idx = np.argsort(x)
    plt.plot(x[sort_idx], y_lin_pred[sort_idx], color='red', linewidth=2,
             label=f'Linear fit (R²={lin_r**2:.3f})')
    plt.plot(x[sort_idx], y_quad_pred[sort_idx], color='green', linewidth=2,
             label=f'Quadratic fit (R²={r2_quad:.3f})')
    
    plt.title(f"{label2} vs {label1} Close Prices")
    plt.xlabel(f"{label1} Close Price")
    plt.ylabel(f"{label2} Close Price")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ---- Metrics ----
    linear_metrics = {
        'model': 'linear',
        'slope': lin_slope,
        'intercept': lin_intercept,
        'r_value': lin_r,
        'r_squared': lin_r ** 2,
        'p_value': lin_p,
        'std_err': lin_stderr
    }

    quadratic_metrics = {
        'model': 'quadratic',
        'a': quad_coeffs[0],
        'b': quad_coeffs[1],
        'c': quad_coeffs[2],
        'r_squared': r2_quad
    }

    print("\n📈 Regression Metrics")
    print("-" * 35)
    print("Linear Regression:")
    for k, v in linear_metrics.items():
        if k != 'model':
            print(f"  {k:>10}: {v:.6f}")
    print("\nQuadratic Regression:")
    for k, v in quadratic_metrics.items():
        if k != 'model':
            print(f"  {k:>10}: {v:.6f}")

    return {'linear': linear_metrics, 'quadratic': quadratic_metrics}

def plot_regression3d(df1, df2, df3=None, label1='Asset 1', label2='Asset 2', label3='Asset 3'):
    """
    Plots regression between close prices:
    - If df3 is None: 2D scatter + linear regression.
    - If df3 is provided: 3D scatter + regression plane (multiple linear regression).
    Returns regression metrics.
    """
    
    # -------------------------------
    # 2D CASE — SAME AS YOUR CURRENT FUNCTION
    # -------------------------------
    if df3 is None:
        x = df1['close']
        y = df2['close']
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = intercept + slope * x

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.005, label='Data points', color='blue')
        plt.plot(x, y_pred, color='red', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')
        plt.title(f"Linear Regression: {label2} vs {label1}")
        plt.xlabel(f"{label1} Close Price")
        plt.ylabel(f"{label2} Close Price")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        metrics = {
            'type': 'linear_2D',
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err
        }

        print("\nRegression Results (2D):")
        for k, v in metrics.items():
            if k != 'type':
                print(f"{k:>10}: {v:.5f}")

        return metrics
    
    # -------------------------------
    # 3D CASE — MULTIPLE LINEAR REGRESSION
    # -------------------------------
    else:
        x = df1['close'].values
        y = df2['close'].values
        z = df3['close'].values

        # Build design matrix for multiple regression
        X = np.column_stack((x, y, np.ones_like(x)))  # [x, y, 1]
        coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)  # Solve least squares

        a, b, c = coeffs  # z = a*x + b*y + c
        z_pred = a * x + b * y + c

        # Compute R²
        ss_res = np.sum((z - z_pred) ** 2)
        ss_tot = np.sum((z - np.mean(z)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # 3D plot
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, alpha=0.3, color='blue', label='Data points')

        # Create grid for regression plane
        x_surf, y_surf = np.meshgrid(
            np.linspace(min(x), max(x), 30),
            np.linspace(min(y), max(y), 30)
        )
        z_surf = a * x_surf + b * y_surf + c
        ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Regression plane')

        ax.set_title(f"3D Linear Regression: {label3} vs {label1}, {label2}")
        ax.set_xlabel(f"{label1} Close Price")
        ax.set_ylabel(f"{label2} Close Price")
        ax.set_zlabel(f"{label3} Close Price")
        plt.tight_layout()
        plt.show()

        metrics = {
            'type': 'linear_3D',
            'coeff_a': a,
            'coeff_b': b,
            'intercept': c,
            'r_squared': r_squared
        }

        print("\nRegression Results (3D):")
        for k, v in metrics.items():
            if k != 'type':
                print(f"{k:>10}: {v:.5f}")

        return metrics

def time_series_plot2d(df1, df2, label1="asset1", label2="asset2", move_window=5, jumps=1, plot=False, dt_x=False):
    dtxx = []
    plt_indy = []
    #df1x, df2x = df1.copy(), df2.copy()
    dtx = 0

    for i in range(-move_window*jumps, move_window*jumps+1, jumps):
        df1x, df2x = df1.copy(), df2.copy()
        plt_indy.append(i+move_window)
        if i < 0:
            df1x = df1x[:i]
            df2x = df2x[-i:]
            #dtx = gather_and_plot(df1x, df2x, label1=label1, label2=label2, plot=plot)
            plot_density(df1x, df2x, label1=label1, label2=label2)
            if dt_x:
                dtxx.append([f"{i}", dtx])
            #print(i, len(df1x), len(df2x))
        elif i > 0:
            df1x = df1x[i:]
            df2x = df2x[:-i]
            #dtx = gather_and_plot(df1x, df2x, label1=label1, label2=label2, plot=plot)
            plot_density(df1x, df2x, label1=label1, label2=label2)
            if dt_x:
                dtxx.append([f"{i}", dtx])
            #print(i, len(df1x), len(df2x))
        else:
            df1x, df2x = df1x, df2x
            #dtx = gather_and_plot(df1x, df2x, label1=label1, label2=label2, plot=plot)
            plot_density(df1x, df2x, label1=label1, label2=label2)
            if dt_x:
                dtxx.append([f"{i}", dtx])
            #print(i, len(df1x), len(df2x))

    if dt_x:  
        sl, inter, rval, residual_values, pval,  sterr = [], [], [], [], [], [] 
        for i in dtxx:
            sl.append(i[1]['slope'])
            inter.append(i[1]['intercept'])
            rval.append(i[1]['r_value'])
            residual_values.append(i[1]['r_squared'])
            pval.append(i[1]['p_value'])
            sterr.append(i[1]['std_err'])

        plt.plot(plt_indy, sl, label='slope')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('slope')
        plt.show()
        #plt.plot(plt_indy, inter, label='intercept')
        plt.plot(plt_indy, rval, label='r_value')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('r_value')
        plt.show()
        plt.plot(plt_indy, residual_values, label='r_squared')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('r_squared')
        plt.show()
        plt.plot(plt_indy, pval, label='p_value')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('p_value')
        plt.show()
        plt.plot(plt_indy, sterr, label='std_err')
        plt.ylabel('std_err')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.show()

    return dtxx
# modify to make it one sided option
def time_series2_plot2d(df1, df2, label1="asset1", label2="asset2", move_window=5, jumps=1, plot=False, dt_x=False, dTx='close'):
    if jumps > 1:
        df1 = aggregate_candles(df1, jump=jumps)
        df2 = aggregate_candles(df2, jump=jumps)

    if dTx == 'returns':
        df1 = add_returns(df1)
        df2 = add_returns(df2)

    dtxx = []
    plt_indy = []
    #df1x, df2x = df1.copy(), df2.copy()
    dtx = 0

    for i in range(-move_window, move_window+1):
        df1x, df2x = df1.copy(), df2.copy()
        plt_indy.append(i+move_window)
        if i < 0:
            df1x = df1x[:i]
            df2x = df2x[-i:]
            dtx = gather_and_plot(df1x, df2x, label1=label1, label2=label2, plot=False, dTx=dTx)
            if plot:
                plot_density(df1x, df2x, label1=label1, label2=label2, dTx=dTx)
            if dt_x:
                dtxx.append([f"{i}", dtx])
            #print(i, len(df1x), len(df2x))
        elif i > 0:
            df1x = df1x[i:]
            df2x = df2x[:-i]
            dtx = gather_and_plot(df1x, df2x, label1=label1, label2=label2, plot=False, dTx=dTx)
            if plot:
                plot_density(df1x, df2x, label1=label1, label2=label2, dTx=dTx)
            if dt_x:
                dtxx.append([f"{i}", dtx])
            #print(i, len(df1x), len(df2x))
        else:
            df1x, df2x = df1x, df2x
            dtx = gather_and_plot(df1x, df2x, label1=label1, label2=label2, plot=False, dTx=dTx)
            if plot:
                plot_density(df1x, df2x, label1=label1, label2=label2, dTx=dTx)
            if dt_x:
                dtxx.append([f"{i}", dtx])
            #print(i, len(df1x), len(df2x))

    if dt_x:  
        sl, inter, rval, residual_values, pval,  sterr = [], [], [], [], [], [] 
        for i in dtxx:
            sl.append(i[1]['slope'])
            inter.append(i[1]['intercept'])
            rval.append(i[1]['r_value'])
            residual_values.append(i[1]['r_squared'])
            pval.append(i[1]['p_value'])
            sterr.append(i[1]['std_err'])

        plt.plot(plt_indy, sl, label='slope')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('slope')
        plt.show()
        #plt.plot(plt_indy, inter, label='intercept')
        plt.plot(plt_indy, rval, label='r_value')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('r_value')
        plt.show()
        plt.plot(plt_indy, residual_values, label='r_squared')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('r_squared')
        plt.show()
        plt.plot(plt_indy, pval, label='p_value')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.ylabel('p_value')
        plt.show()
        plt.plot(plt_indy, sterr, label='std_err')
        plt.ylabel('std_err')
        #plt.xticks(np.array(plt_indy), [i[0] for i in dtxx])
        plt.show()

    return dtxx

def gather_and_plot(df1, df2, label1='Asset 1', label2='Asset 2', plot = False, dTx = 'close'):
    x = df1[dTx]
    y = df2[dTx]
    
    # Run linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Predicted line
    y_pred = intercept + slope * x

    # Plot
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.05, label='Data points', color='blue')
        plt.plot(x, y_pred, color='red', linewidth=2, label=f'Regression line (R²={r_value**2:.6f})')
        plt.title(f"Linear Regression: {label2} vs {label1}")
        plt.xlabel(f"{label1} Close Price")
        plt.ylabel(f"{label2} Close Price")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # Print and return regression metrics
    metrics = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    #print("\nRegression Results:")
    #for k, v in metrics.items():
    #    print(f"{k:>10}: {v:.5f}")
    
    return metrics

def plot_density(df1, df2, df3=None, label1='Asset 1', label2='Asset 2', label3='Asset 3', sample_size=5000, dTx = 'close'):
    """
    Plots a density-colored scatter plot of close prices.
    - If df3 is None: 2D scatter (df1 vs df2)
    - If df3 is provided: 3D scatter (df1 vs df2 vs df3)
    Colors represent point density using the 'plasma' colormap.
    Randomly samples points for manageability.
    """

    # Extract close values
    x = df1[dTx].values
    y = df2[dTx].values
    n_points = len(x)
    #print(f"Total data points before sampling: {n_points}")

    # Random sampling
    if n_points > sample_size:
        idx = np.random.choice(n_points, sample_size, replace=False)
        x, y = x[idx], y[idx]
        if df3 is not None:
            z = df3[dTx].values[idx]
    else:
        if df3 is not None:
            z = df3[dTx].values

    # ---------------------------------------
    # 2D CASE
    # ---------------------------------------
    if df3 is None:
        # Compute density
        xy = np.vstack([x, y])
        z_density = gaussian_kde(xy)(xy)

        # Sort by density for better color layering
        idx = z_density.argsort()
        x, y, z_density = x[idx], y[idx], z_density[idx]

        # Plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=z_density, cmap='plasma', s=8, alpha=0.7)
        plt.colorbar(scatter, label='Density')
        plt.title(f"Density Scatter: {label2} vs {label1}")
        plt.xlabel(f"{label1} Close Price")
        plt.ylabel(f"{label2} Close Price")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------
    # 3D CASE
    # ---------------------------------------
    else:
        # Compute density in 3D space
        xyz = np.vstack([x, y, z])
        density = gaussian_kde(xyz)(xyz)

        # Sort by density for better layering
        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]

        # 3D scatter
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=density, cmap='plasma', s=6, alpha=0.8)
        fig.colorbar(scatter, ax=ax, label='Density')
        ax.set_title(f"3D Density Scatter: {label3} vs {label1}, {label2}")
        ax.set_xlabel(f"{label1} Close Price")
        ax.set_ylabel(f"{label2} Close Price")
        ax.set_zlabel(f"{label3} Close Price")
        plt.tight_layout()
        plt.show()

def multivariate_regression(df_target, df1, df2, label_target='Target', label1='Predictor1', label2='Predictor2', returns=True):

    # Work on copies to avoid modifying original data
    df_target, df1, df2 = df_target.copy(), df1.copy(), df2.copy()

    if returns:
        # Compute returns (do not dropna here)
        df_target['returns'] = df_target['close'].pct_change()
        df1['returns'] = df1['close'].pct_change()
        df2['returns'] = df2['close'].pct_change()

        # Merge all on date
        merged = (
            df_target[['date', 'returns']]
            .merge(df1[['date', 'returns']], on='date', suffixes=('_target', f'_{label1}'))
            .merge(df2[['date', 'returns']], on='date')
            .rename(columns={'returns': f'returns_{label2}'})
        )
        value_cols = ['returns_target', f'returns_{label1}', f'returns_{label2}']

    else:
        merged = (
            df_target[['date', 'close']]
            .merge(df1[['date', 'close']], on='date', suffixes=('_target', f'_{label1}'))
            .merge(df2[['date', 'close']], on='date')
            .rename(columns={'close': f'close_{label2}'})
        )
        value_cols = ['close_target', f'close_{label1}', f'close_{label2}']

    # Drop rows with any NaN or inf values
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Define dependent (y) and independent (X) variables
    y = merged[value_cols[0]]
    X = merged[value_cols[1:]]
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()

    # Extract key metrics
    results = {
        'R_squared': model.rsquared,
        'Adj_R_squared': model.rsquared_adj,
        'F_statistic': model.fvalue,
        'F_pvalue': model.f_pvalue,
        'Coefficients': model.params.to_dict(),
        'P_values': model.pvalues.to_dict(),
        'Standard_errors': model.bse.to_dict(),
        't_values': model.tvalues.to_dict(),
        'AIC': model.aic,
        'BIC': model.bic,
        'N_observations': int(model.nobs)
    }

    # Display summary
    print("\n📊 MULTIVARIATE REGRESSION RESULTS")
    print(model.summary())

    return results, model

dfsol = process_dfs(TABLE1_NAME)
dfbtc = process_dfs(TABLE2_NAME)
dfeth = process_dfs(TABLE3_NAME)


dfsol, dfbtc, dfeth = synchronise_continuous(dfsol, dfbtc, dfeth, freq='5min')

dfsolc = add_returns(dfsol)
dfbtcc = add_returns(dfbtc)
dfethc = add_returns(dfeth)
dfsol['ratio'] = np.log2(dfbtcc['close']) - np.log2(dfsolc['close'])
plt.plot(dfsol['ratio'], color='cornflowerblue')
plt.show()

#plot_density(dfbtcc, dfsolc, label1='Solana', label2='Bitcoin', sample_size=10000, dTx='returns')
#plot_regression(dfbtcc, dfsolc, label1='Solana', label2='Bitcoin', dTx = 'returns')


dfsol['ln_close'] = np.log2(dfsol['close'])
dfbtc['ln_close'] = np.log2(dfbtc['close'])
dfeth['ln_close'] = np.log2(dfeth['close'])
#plot_regression(dfsol, dfbtc, label1='Solana', label2='Bitcoin', dTx = 'ln_close') 


#plot_density(dfsolc, dfbtcc, label1='solana pct change', label2='bitcoin pct change', sample_size=35000, dTx='returns')
#plot_density(dfsol, dfbtc, dfeth, label1='solana', label2='bitcoin', label3='ethereum', sample_size=10000)
#res, mod = multivariate_regression(dfsolc, dfbtcc, dfethc, label_target='Solana', label1='Bitcoin', label2='Ethereum', returns=True)

'''time_series2_plot2d(dfsolc, dfbtcc, label1='Solana', label2='Bitcoin', move_window=50, jumps=12, plot=False, dt_x=True, dTx ='returns')'''

#dfsol, dfbtc, dfeth = synchronise_tables(dfsol, dfbtc, dfeth)
#print(len(dfsol), len(dfbtc), len(dfeth))



#plot_regression(dfsolc, dfbtcc, label1='Solana', label2='Bitcoin', dTx = 'returns')
#time_series2_plot2d(dfsolc, dfbtcc, label1='Solana', label2='Bitcoin', move_window=10, jumps=3, plot=False, dt_x=True)
#plot_regression3d(dfsol, dfbtc, dfeth, label1='Solana', label2='Bitcoin', label3='Ethereum')
#xxs = time_series_plot2d(dfsol, dfbtc, label1='Solana', label2='Bitcoin', move_window=5, jumps=20, plot=True)



# cv tips       


# independant project section 
# put it in skills
# tailoring the cv for each job posting ==> look for the keywords in the job description and make sure they are in your cv
# don't talk about the group also talk about you 
# deffinetely quality over qunatity
# be very sepcifque about the compnay/commercial awareness, understdand the market, the competitors etc
# makes a lot of reearch about the company
# google news 
# network through bath students etc alimni network 
# follow people on linkedin
# maybe reach out through linkedin to people 
