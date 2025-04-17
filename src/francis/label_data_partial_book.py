import pandas as pd


def prepare_training_data_partial_book(
    orderbook_csv, mid_price_csv, threshold=0.0002, top_n=5
):

    # 載入 orderbook 和 mid price
    ob_df = pd.read_csv(orderbook_csv)
    mp_df = pd.read_csv(mid_price_csv)

    # 計算 mid price label
    mp_labeled = mp_df.copy()
    mp_labeled["mid_next"] = mp_labeled["mid_price"].shift(-1)
    mp_labeled["ret"] = (mp_labeled["mid_next"] - mp_labeled["mid_price"]) / mp_labeled[
        "mid_price"
    ]

    def classify_direction(r):
        if r > threshold:
            return +1
        elif r < -threshold:
            return -1
        else:
            return 0

    mp_labeled["label"] = mp_labeled["ret"].apply(classify_direction)
    mp_labeled = mp_labeled.dropna().reset_index(drop=True)

    # 對每個 timestamp 做特徵擷取
    features = []

    for ts, group in ob_df.groupby("timestamp"):
        group = group.sort_values(
            "price", ascending=False if "bid" in group["side"].values else True
        )
        bids = (
            group[group["side"] == "bid"]
            .sort_values("price", ascending=False)
            .head(top_n)
        )
        asks = (
            group[group["side"] == "ask"]
            .sort_values("price", ascending=True)
            .head(top_n)
        )

        if len(bids) < top_n or len(asks) < top_n:
            continue

        bid_prices = bids["price"].tolist()
        bid_sizes = bids["size"].tolist()
        ask_prices = asks["price"].tolist()
        ask_sizes = asks["size"].tolist()

        spread = ask_prices[0] - bid_prices[0]
        bid_vol = sum(bid_sizes)
        ask_vol = sum(ask_sizes)
        imbalance = bid_vol / (bid_vol + ask_vol + 1e-6)

        row = {
            "timestamp": ts,
            "spread": spread,
            "imbalance": imbalance,
        }

        for i in range(top_n):
            row[f"bid_{i+1}"] = bid_prices[i]
            row[f"bid_{i+1}_size"] = bid_sizes[i]
            row[f"ask_{i+1}"] = ask_prices[i]
            row[f"ask_{i+1}_size"] = ask_sizes[i]

        features.append(row)

    feature_df = pd.DataFrame(features)

    # merge 特徵 + label
    final_df = pd.merge(
        feature_df, mp_labeled[["timestamp", "label"]], on="timestamp", how="inner"
    )
    return final_df
