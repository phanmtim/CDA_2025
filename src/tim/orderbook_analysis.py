import pandas as pd
import matplotlib.pyplot as plt


def loadReconstructedOrderbook(filePath: str) -> pd.DataFrame:
    """
    Load a per-second reconstructed orderbook CSV (reconstructed_orderbook_small).

    Expects columns: ['timestamp','bid_orders','ask_orders','best_bid','best_ask','mid_price'].
    Sets 'timestamp' as a DatetimeIndex.

    Args:
        filePath: Path to the reconstructed CSV file.

    Returns:
        DataFrame indexed by timestamp.
    """
    df = pd.read_csv(
        filePath,
        parse_dates=['timestamp'],
        dtype={
            'bid_orders': int,
            'ask_orders': int,
            'best_bid': float,
            'best_ask': float,
            'mid_price': float
        }
    )
    df = df.set_index('timestamp').sort_index()
    return df


def plotMidPrice(book: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot mid price over time from reconstructed orderbook.
    """
    plt.figure(figsize=figsize)
    plt.plot(book.index, book['mid_price'], label='Mid Price')
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.title('Bitcoin Mid Price Over Time')
    plt.tight_layout()
    plt.show()


def plotOrderCounts(book: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot bid and ask order counts over time.
    """
    plt.figure(figsize=figsize)
    plt.plot(book.index, book['bid_orders'], label='Bid Orders')
    plt.plot(book.index, book['ask_orders'], label='Ask Orders')
    plt.xlabel('Time')
    plt.ylabel('Number of Orders')
    plt.title('Bid vs Ask Order Counts Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotSpread(book: pd.DataFrame, figsize: tuple = (12, 6)):
    """
    Plot bid-ask spread over time using best_bid and best_ask.
    """
    spread = book['best_ask'] - book['best_bid']
    plt.figure(figsize=figsize)
    plt.plot(book.index, spread)
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.title('Bid-Ask Spread Over Time')
    plt.tight_layout()
    plt.show()


def detectFlashCrashes(
    book: pd.DataFrame,
    thresholdRangePct: float = 0.05
) -> pd.DataFrame:
    """
    Detect crashes defined as drops >= thresholdRangePct * (max(mid_price) - min(mid_price))
    between consecutive seconds.

    Args:
        book: DataFrame from loadReconstructedOrderbook, indexed by timestamp.
        thresholdRangePct: Fraction of full mid_price range to flag as crash.

    Returns:
        DataFrame of crash events with ['startTime','endTime','drop_amount'].
    """
    df = book.copy()
    # compute previous-second mid_price
    df['prev_mid'] = df['mid_price'].shift(1)
    df = df.dropna(subset=['prev_mid'])
    # absolute threshold based on full range
    full_range = df['mid_price'].max() - df['mid_price'].min()
    threshold = thresholdRangePct * full_range
    # compute drop amount
    df['drop_amount'] = df['prev_mid'] - df['mid_price']
    # select events
    crashes = df[df['drop_amount'] >= threshold].copy()
    # assign start and end times
    crashes['startTime'] = crashes.index.to_series().shift(1)
    crashes['endTime'] = crashes.index
    # format result
    result = crashes.reset_index()[['startTime', 'endTime', 'drop_amount']]
    return result


def detectFlashCrashesRolling(
    book: pd.DataFrame,
    thresholdRangePct: float = 0.05,
    windowSeconds: int = 5
) -> pd.DataFrame:
    """
    Detect crashes defined as drops >= thresholdRangePct * (max(mid_price) - min(mid_price))
    over any windowSeconds.

    Args:
        book: DataFrame from loadReconstructedOrderbook, indexed by timestamp.
        thresholdRangePct: Fraction of full mid_price range to flag as crash.
        windowSeconds: Number of seconds for rolling window.

    Returns:
        DataFrame of crash events with ['startTime','endTime','drop_amount'].
    """
    # compute absolute threshold
    full_range = book['mid_price'].max() - book['mid_price'].min()
    threshold = thresholdRangePct * full_range

    df = book.copy()
    # rolling_max of the last windowSeconds
    df['rolling_max'] = (
        df['mid_price']
        .rolling(window=windowSeconds, min_periods=1)
        .max()
    )
    df['drop_amount'] = df['rolling_max'] - df['mid_price']
    crashes = df[df['drop_amount'] >= threshold].copy()

    # label start/end times
    crashes['endTime'] = crashes.index
    crashes['startTime'] = crashes.index - pd.Timedelta(seconds=windowSeconds)
    return crashes.reset_index()[['startTime', 'endTime', 'drop_amount']]


def plotFlashCrashes(
    book: pd.DataFrame,
    crashes: pd.DataFrame,
    figsize: tuple = (12, 6)
):
    """
    Plot mid price with highlighted crash intervals.
    """
    plt.figure(figsize=figsize)
    plt.plot(book.index, book['mid_price'], label='Mid Price')
    for _, row in crashes.iterrows():
        plt.axvspan(row['startTime'], row['endTime'], alpha=0.3, color='red')
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.title('Defined Flash Crashes in Bitcoin')
    plt.tight_layout()
    plt.show()
