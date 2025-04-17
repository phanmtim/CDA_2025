import pandas as pd
import matplotlib.pyplot as plt


def loadOrderbookData(filePath: str) -> pd.DataFrame:
    """
    Load orderbook CSV file and parse timestamp to datetime.

    Args:
        filePath: Path to the CSV file.

    Returns:
        DataFrame with parsed timestamps.
    """
    df = pd.read_csv(filePath)
    # assume timestamp is in seconds since epoch
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df


def aggregateOrderbook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw orderbook to compute best bid/ask and total sizes per timestamp.

    Args:
        df: Raw orderbook DataFrame with columns ['timestamp','side','price','size',...].

    Returns:
        DataFrame with columns ['timestamp','bestBid','bestAsk','bidSize','askSize'].
    """
    bids = df[df['side'] == 'bid'].groupby('timestamp').agg(
        bestBid=('price', 'max'),
        bidSize=('size', 'sum')
    )
    asks = df[df['side'] == 'ask'].groupby('timestamp').agg(
        bestAsk=('price', 'min'),
        askSize=('size', 'sum')
    )
    book = pd.concat([bids, asks], axis=1).reset_index().sort_values('timestamp')
    return book


def computeMetrics(book: pd.DataFrame) -> pd.DataFrame:
    """
    Given aggregated book DataFrame, compute mid price, spread, and imbalance.

    Args:
        book: DataFrame from aggregateOrderbook.

    Returns:
        Same DataFrame with added ['midPrice','spread','imbalance'].
    """
    book = book.copy()
    book['midPrice'] = (book['bestBid'] + book['bestAsk']) / 2
    book['spread'] = book['bestAsk'] - book['bestBid']
    book['imbalance'] = (
        book['bidSize'] - book['askSize']
    ) / (book['bidSize'] + book['askSize'])
    return book


def plotMidPrice(book: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot mid price over time.
    """
    plt.figure(figsize=figsize)
    plt.plot(book['timestamp'], book['midPrice'])
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.title('Bitcoin Mid Price Over Time')
    plt.tight_layout()
    plt.show()


def plotSpread(book: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot bid-ask spread over time.
    """
    plt.figure(figsize=figsize)
    plt.plot(book['timestamp'], book['spread'])
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.title('Bid-Ask Spread Over Time')
    plt.tight_layout()
    plt.show()


def plotImbalance(book: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot order book imbalance over time.
    """
    plt.figure(figsize=figsize)
    plt.plot(book['timestamp'], book['imbalance'])
    plt.xlabel('Time')
    plt.ylabel('Imbalance')
    plt.title('Order Book Imbalance Over Time')
    plt.tight_layout()
    plt.show()


def detectFlashCrashes(
    book: pd.DataFrame,
    thresholdPct: float = 0.05,
    windowSeconds: int = 5
) -> pd.DataFrame:
    """
    Detect flash crashes defined as drops > thresholdPct within windowSeconds.

    Args:
        book: DataFrame with 'timestamp' index and 'midPrice'.
        thresholdPct: Fractional drop threshold (e.g., 0.05 for 5%).
        windowSeconds: Sliding window in seconds to compare prices.

    Returns:
        DataFrame of crash events with ['startTime','endTime','dropPct'].
    """
    df = book.copy().set_index('timestamp')
    # shift midPrice by windowSeconds ahead
    shifted = df['midPrice'].shift(freq=pd.Timedelta(seconds=windowSeconds))
    df = df.dropna(subset=['midPrice'])
    df['prevPrice'] = shifted
    df = df.dropna(subset=['prevPrice'])
    df['dropPct'] = (df['prevPrice'] - df['midPrice']) / df['prevPrice']
    crashes = df[df['dropPct'] >= thresholdPct].reset_index()
    crashes['startTime'] = crashes['timestamp'] - pd.Timedelta(seconds=windowSeconds)
    crashes = crashes[['startTime', 'timestamp', 'dropPct']].rename(
        columns={'timestamp': 'endTime'}
    )
    return crashes


def plotFlashCrashes(
    book: pd.DataFrame,
    crashes: pd.DataFrame,
    figsize: tuple = (12, 6)
):
    """
    Plot mid price and highlight flash crash intervals.
    """
    plt.figure(figsize=figsize)
    plt.plot(book['timestamp'], book['midPrice'], label='Mid Price')
    for _, row in crashes.iterrows():
        plt.axvspan(row['startTime'], row['endTime'], alpha=0.3, color='red')
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.title('Flash Crashes in Bitcoin')
    plt.tight_layout()
    plt.show()


def analyzeOrderbook(
    filePath: str,
    thresholdPct: float = 0.05,
    windowSeconds: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end analysis: load data, compute metrics, plot visualizations, detect and plot crashes.

    Returns:
        metrics DataFrame and crashes DataFrame.
    """
    raw = loadOrderbookData(filePath)
    book = aggregateOrderbook(raw)
    metrics = computeMetrics(book)
    # Visualizations
    plotMidPrice(metrics)
    plotSpread(metrics)
    plotImbalance(metrics)
    crashes = detectFlashCrashes(metrics, thresholdPct, windowSeconds)
    plotFlashCrashes(metrics, crashes)
    return metrics, crashes
