import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from scipy.stats import gaussian_kde
from scipy.stats import linregress

DB_FILE = "SQL001.db"
TABLE1_NAME = "Solana3"
TABLE2_NAME = 'BTC'
TABLE3_NAME = 'ETH'

def proccess_dfs(table_name, db_file = DB_FILE):
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

def synchronise_continuous(df1, df2, df3=None, freq='5T'):
    """
    Synchronizes 2 or 3 DataFrames by matching dates and ensuring perfect continuity.
    
    Parameters:
        df1, df2, df3: pandas DataFrames with 'date' column
        freq: frequency string (default '5T' for 5 minutes)
    
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

def spread_from_prediction(df1, df2, dTx='returns'):
    # Ensure synchronization
    df1_sync, df2_sync = synchronise_tables(df1, df2)
    
    X = df2_sync[dTx].values
    Y = df1_sync[dTx].values
  # Adds a constant term to the predictor
    model = sm.OLS(Y, X).fit()
    #results_summary = model.summary()
    #print(results_summary)
    
    Y_pred = model.predict(X)
    
    spread = Y - Y_pred
    
    return pd.Series(spread, index=df1_sync.index), model

def future_movements(df, steps=5, key='returns'):
    mini, maxi = [], []
    for i in range(len(df)-steps):
        if i%1000 == 0:
            print(f"processing row {i}")
        travel = [1]
        for j in range(steps):
            move = travel[j]*(1+df.iloc[i+j+1][key]/100)
            travel.append(move)
        travel.pop(0)
        mini.append(np.min(travel))
        maxi.append(np.max(travel))

    return pd.Series(mini, index=df.index[:len(mini)]), pd.Series(maxi, index=df.index[:len(maxi)])

def plot_converge (df):
    # if residual < 0, Sol has underperformed BTC, so we look at MaxFuture
    # if residual > 0, Sol has overperformed BTC, so we look at Min
    y = np.where(df['residuals'] < 0, df['MaxFuture'], df['MinFuture'])
    plt.scatter(df['residuals'], y * 100, s=3, color='skyblue', alpha=0.5, linewidth=0.001)
    plt.xlabel('Residuals')
    plt.ylabel('Future Movement (%)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axhline(-0.15, color='cornflowerblue', linestyle='--', alpha=0.3)
    plt.axhline(0.15, color='cornflowerblue', linestyle='--', alpha=0.3)
    plt.show()

def plot_converge_with_regression(df):
    # Choose y depending on residual sign
    y = np.where(df['residuals'] < 0, df['MinFuture'], df['MaxFuture'])
    
    # Convert to percentage for plotting
    y_percent = y * 100
    x = df['residuals'].values

    # Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y_percent, s=3, alpha=0.5, color='skyblue', label='Future Movement')

    # Linear regression
    model = LinearRegression()
    x_reshaped = x.reshape(-1, 1)
    model.fit(x_reshaped, y_percent)
    y_pred = model.predict(x_reshaped)

    # Plot regression line
    plt.plot(x, y_pred, color='red', linewidth=2, label='Linear Regression')

    # Labels
    plt.xlabel('Residuals')
    plt.ylabel('Future Movement (%)')
    plt.title('Residuals vs Future Movement with Linear Regression')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Regression metrics
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y_percent, y_pred)
    mse = mean_squared_error(y_percent, y_pred)
    mae = mean_absolute_error(y_percent, y_pred)

    print("Linear Regression Metrics:")
    print(f"  Slope       : {slope:.5f}")
    print(f"  Intercept   : {intercept:.5f}")
    print(f"  R^2 Score   : {r2:.5f}")
    print(f"  MSE         : {mse:.5f}")
    print(f"  MAE         : {mae:.5f}")

def plot_convergex():
    dfsol = proccess_dfs(TABLE1_NAME)
    dfbtc = proccess_dfs(TABLE2_NAME)
    dfsol, dfbtc = synchronise_continuous(dfsol, dfbtc, freq='5T')
    dfsolc = add_returns(dfsol)
    dfbtcc = add_returns(dfbtc)
    X = dfbtcc['returns'].values
    Y = dfsolc['returns'].values
    model = sm.OLS(Y, X).fit()
    Y_pred = model.predict(X)
    spread = Y - Y_pred
    res = pd.Series(spread, index=dfsolc.index)

def plot_converge2(df, df2):
    # Align the two DataFrames by index to make sure lengths match
    df, df2 = df.align(df2, join='inner', axis=0)
    y = np.where(df['residuals'] < 0, df['MaxFuture'], df['MinFuture']) * 100
    colors = df2['corr_sol_returns_btc_returns']
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df['residuals'], y, c=colors, cmap='plasma', s=8, alpha=0.5)
    plt.xlabel('Residuals')
    plt.ylabel('Future Movement (%)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axhline(-0.15, color='cornflowerblue', linestyle='--', alpha=0.3)
    plt.axhline(0.15, color='cornflowerblue', linestyle='--', alpha=0.3)

    cbar = plt.colorbar(sc)
    cbar.set_label('Correlation Score (0–100)')
    plt.title('Residuals vs Future Movement\nColored by Correlation Score')
    plt.show()

def covariance(x, y, length):
    avex, avey = np.mean(x), np.mean(y)
    cov = np.sum((x - avex) * (y - avey)) / len(x) - 1
    return cov

def correlation_coefficient(x, y, length):
    cov = covariance(x, y, length)
    stdx = np.std(x)
    stdy = np.std(y)
    return cov / (stdx * stdy)

def corr_coef_columns(df, col1, col2, length):
    # Compute rolling correlation between two columns
    df['corr_' + col1 + '_' + col2] = (
        df[col1]
        .rolling(window=length)
        .corr(df[col2])
    )
    df = df.dropna()
    return df

def binshit (df):
    values = np.where(df["residuals"] < 0, df["MaxFuture"], df["MinFuture"])
    plt.hist(values, bins=100, edgecolor='black')
    plt.show()


dfsol = proccess_dfs(TABLE1_NAME)
dfbtc = proccess_dfs(TABLE2_NAME)
dfeth = proccess_dfs(TABLE3_NAME)
dfsol, dfbtc, dfeth = synchronise_continuous(dfsol, dfbtc, dfeth, freq='5T')
dfsolc = add_returns(dfsol)
dfbtcc = add_returns(dfbtc)
dfethc = add_returns(dfeth)

dfx = pd.DataFrame()
dfx['sol_returns'] = dfsolc['returns']
dfx['btc_returns'] = dfbtcc['returns']
corr_coef_columns(dfx, 'sol_returns', 'btc_returns', length=50)
dfx['btc_std'] = dfx['btc_returns'].rolling(window=10).std()
#dfx['rank_cols'] = dfx['corr_sol_returns_btc_returns'].rank(pct=True)*100
dfx['rank_cols'] = dfx['btc_std'].rank(pct=True)*100
dfx = dfx.dropna()
#print(dfx)
#print(dfx['corr_sol_btc'].describe())
#print(dfx['rank_cols'].describe())
#print(dfsolc)
#print(dfsolc['returns'].describe())
#print(f"sol max: {dfsolc['returns'].max()}, min: {dfsolc['returns'].min()}")


res, _ = spread_from_prediction(dfsolc, dfbtcc, dTx='returns')
Min, Max = future_movements(dfsolc, steps=288, key='returns')
result = pd.concat([res, Min, Max], axis=1)
result.columns = ['residuals', 'MinFuture', 'MaxFuture']
result['MinFuture'] = result['MinFuture']-1
result['MaxFuture'] = result['MaxFuture']-1
result = result.dropna()
#print(result)
#plot_converge(result)
#plot_converge2(result, dfx)
binshit(result)


# convergence test
'''result['y_future'] = np.where(result['residuals'] < 0, result['MaxFuture'], result['MinFuture'])
X = sm.add_constant(result['residuals'])
y = result['y_future']
model = sm.OLS(y, X).fit()
print(model.summary())
print(model.params)
print(model.bse)
print(X.var())

print(dfsolc['returns'].describe())'''


