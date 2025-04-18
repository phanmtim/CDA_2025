import pandas as pd
import matplotlib.pyplot as plt


def loadOrderbookCsv(filePath: str) -> pd.DataFrame:
    """
    Load per-minute Bitcoin orderbook CSV.

    Expects columns: ['timestamp','bid_orders','ask_orders','best_bid','best_ask','mid_price'].
    Returns:
        DataFrame indexed by timestamp.
    """
    df = pd.read_csv(
        filePath,
        parse_dates=['timestamp'],
        dtype={
            'bid_orders': float,
            'ask_orders': float,
            'best_bid': float,
            'best_ask': float,
            'mid_price': float
        }
    )
    df = df.set_index('timestamp').sort_index()
    return df


def plotMidPrice(df: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot mid price over time.
    """
    plt.figure(figsize=figsize)
    plt.plot(df.index, df['mid_price'], label='Mid Price')
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.title('Bitcoin Mid Price Over Time')
    plt.tight_layout()
    plt.show()


def plotOrderCounts(df: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot bid and ask order counts over time.
    """
    plt.figure(figsize=figsize)
    plt.plot(df.index, df['bid_orders'], label='Bid Orders')
    plt.plot(df.index, df['ask_orders'], label='Ask Orders')
    plt.xlabel('Time')
    plt.ylabel('Number of Orders')
    plt.title('Bid vs Ask Order Counts Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotSpread(df: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot bid-ask spread over time.
    """
    spread = df['best_ask'] - df['best_bid']
    plt.figure(figsize=figsize)
    plt.plot(df.index, spread)
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.title('Bid-Ask Spread Over Time')
    plt.tight_layout()
    plt.show()


def detectFlashCrashesHourly(
    df: pd.DataFrame,
    thresholdPct: float = 0.05,
    windowHours: int = 1
) -> pd.DataFrame:
    """
    Detect flash crashes defined as a drop ≥ thresholdPct of mid_price
    within the last windowHours hours.

    Returns:
        DataFrame of crash events with columns ['startTime','endTime','drop_pct','duration'].
    """
    d = df.copy()
    # rolling max of mid_price over the past windowHours
    d['rolling_max'] = d['mid_price'].rolling(f'{windowHours}H', min_periods=1).max()
    d['drop_pct'] = (d['rolling_max'] - d['mid_price']) / d['rolling_max']
    # flag first crossings
    d['above'] = d['drop_pct'] >= thresholdPct
    d['prev_above'] = d['above'].shift(1, fill_value=False)
    events = d[d['above'] & (~d['prev_above'])]

    records = []
    for endTime, row in events.iterrows():
        window_start = endTime - pd.Timedelta(hours=windowHours)
        window_slice = d.loc[window_start:endTime]
        startTime = window_slice['mid_price'].idxmax()
        duration = endTime - startTime
        records.append({
            'startTime': startTime,
            'endTime': endTime,
            'drop_pct': row['drop_pct'],
            'duration': duration
        })
    return pd.DataFrame(records)


def plotFlashCrashes(
    df: pd.DataFrame,
    crashes: pd.DataFrame,
    figsize: tuple = (12, 6)
):
    """
    Plot mid_price with highlighted crash intervals.
    """
    plt.figure(figsize=figsize)
    plt.plot(df.index, df['mid_price'], label='Mid Price')
    for _, row in crashes.iterrows():
        plt.axvspan(row['startTime'], row['endTime'], alpha=0.3, color='red')
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.title('Flash Crashes in Bitcoin')
    plt.tight_layout()
    plt.show()
