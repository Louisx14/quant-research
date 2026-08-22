import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

DB_FILE = "SQL001.db"
TABLE1_NAME = "Solana3"
TABLE2_NAME = 'BTC'

def create_ready_table(table_name, db_file = DB_FILE):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def synchronise_tables(df1, df2):
    # Ensure 'date' columns are datetime
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    
    # Find common timestamps
    common_dates = pd.Index(df1['date']).intersection(df2['date'])
    
    # Trim both DataFrames to only include those timestamps
    df1_trimmed = df1[df1['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    df2_trimmed = df2[df2['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    
    return df1_trimmed, df2_trimmed

def continuous_synch(df1, df2):
    pass

def plot_regression(df1, df2, label1='Asset 1', label2='Asset 2'):
    """
    Plots a scatter plot of df1['close'] vs df2['close'], 
    fits a linear regression, and returns regression metrics.
    """
    x = df1['close']
    y = df2['close']
    
    # Run linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Predicted line
    y_pred = intercept + slope * x

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.005, label='Data points', color='blue')
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

dfsol = create_ready_table(TABLE1_NAME)
dfbtc = create_ready_table(TABLE2_NAME)

dfsol, dfbtc = synchronise_tables(dfsol, dfbtc)

plot_regressions(dfsol, dfbtc, label1='Solana', label2='Bitcoin')



#df = df.iloc[80000:-4322].reset_index(drop=True)

# test for differemt tieframes 
# test if error scales with price or changes throuhg time 
# try linking to halving cycles
# run mutli parameter regression with ETH price.
# try and incorporate volumes.