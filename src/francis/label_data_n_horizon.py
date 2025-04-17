import pandas as pd


def prepare_training_data_fullbook_future_label(
    orderbook_csv, mid_price_csv, threshold=0.0002, horizon=10
):
    """
    建立保留完整 bids/asks 的 training data 並用 future max/min 方式貼標

    Parameters:
    - orderbook_csv: 重建後的 orderbook CSV 檔案（含所有 bids/asks）
    - mid_price_csv: mid price series 檔案
    - threshold: 幅度門檻（預設 0.0002 即 0.02%）
    - horizon: 向前觀察的步數（預設 10 步）

    Returns:
    - final_df: 包含 timestamp, bids, asks, label 的 DataFrame
    """

    # Step 1: 載入資料
    ob_df = pd.read_csv(orderbook_csv)
    mp_df = pd.read_csv(mid_price_csv)

    # Step 2: 建立 future-N-step 標籤
    mp_df = mp_df.copy().reset_index(drop=True)
    mid_prices = mp_df["mid_price"].values
    labels = []

    for i in range(len(mp_df)):
        future_slice = mid_prices[i + 1 : i + 1 + horizon]

        if len(future_slice) == 0:
            labels.append(None)
            continue

        max_future = max(future_slice)
        min_future = min(future_slice)
        current = mid_prices[i]

        if (max_future - current) / current > threshold:
            labels.append(1)
        elif (min_future - current) / current < -threshold:
            labels.append(-1)
        else:
            labels.append(0)

    mp_df["label"] = labels
    mp_labeled = mp_df.dropna().reset_index(drop=True)

    # Step 3: 整理 orderbook 成 bids / asks list
    features = []
    for ts, group in ob_df.groupby("timestamp"):
        bids = group[group["side"] == "bid"].sort_values("price", ascending=False)
        asks = group[group["side"] == "ask"].sort_values("price", ascending=True)

        bid_list = bids[["price", "size"]].values.tolist()
        ask_list = asks[["price", "size"]].values.tolist()

        features.append(
            {
                "timestamp": ts,
                "bids": bid_list,
                "asks": ask_list,
            }
        )

    feature_df = pd.DataFrame(features)

    # Step 4: merge features + labels
    final_df = pd.merge(
        feature_df, mp_labeled[["timestamp", "label"]], on="timestamp", how="inner"
    )
    return final_df
