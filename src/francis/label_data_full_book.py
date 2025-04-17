import pandas as pd


def prepare_training_data_full_book(orderbook_csv, mid_price_csv, threshold=0.0002):
    ob_df = pd.read_csv(orderbook_csv)
    mp_df = pd.read_csv(mid_price_csv)

    # Step 1: label generation
    mp_df["mid_next"] = mp_df["mid_price"].shift(-1)
    mp_df["ret"] = (mp_df["mid_next"] - mp_df["mid_price"]) / mp_df["mid_price"]

    def classify_direction(r):
        if r > threshold:
            return +1
        elif r < -threshold:
            return -1
        else:
            return 0

    mp_df["label"] = mp_df["ret"].apply(classify_direction)
    mp_labeled = mp_df.dropna().reset_index(drop=True)

    # Step 2: collect all bids/asks for each timestamp (keep the original price levels)
    features = []

    for ts, group in ob_df.groupby("timestamp"):
        bids = group[group["side"] == "bid"].sort_values("price", ascending=False)
        asks = group[group["side"] == "ask"].sort_values("price", ascending=True)

        bid_list = bids[["price", "size"]].values.tolist()
        ask_list = asks[["price", "size"]].values.tolist()

        features.append({"timestamp": ts, "bids": bid_list, "asks": ask_list})

    feature_df = pd.DataFrame(features)

    # Step 3: merge with label
    final_df = pd.merge(
        feature_df, mp_labeled[["timestamp", "label"]], on="timestamp", how="inner"
    )

    return final_df
