# Re-import necessary libraries and reprocess the CSV after the session reset
import pandas as pd
import json

# Reload the uploaded CSV file
file_path = "data/orderbook_data.csv"
df = pd.read_csv(file_path, header=None)

# Assign column names
df.columns = ["timestamp", "side", "price", "size", "number_of_orders"]
df = df[df["side"].isin(["bid", "ask"])]
df["timestamp"] = df["timestamp"].astype(float)
df["price"] = df["price"].astype(float)
df["size"] = df["size"].astype(float)

# Sort by timestamp
df = df.sort_values(by="timestamp")


def reconstruct_orderbook_fast(
    df,
    top_n=50,
    snap_interval=1.0,
    json_path="data/orderbook_snapshots.json",
    csv_path="data/orderbook_snapshots.csv",
):
    current_bids = {}  # price: size
    current_asks = {}
    orderbook_history = {}
    csv_rows = []

    prev_snap_ts = None

    for row in df.itertuples(index=False):
        ts = row.timestamp
        side = row.side
        price = float(row.price)
        size = float(row.size)
        num_orders = int(row.number_of_orders)

        # 更新書本
        book = current_bids if side == "bid" else current_asks
        if size == 0 and num_orders == 0:
            book.pop(price, None)
        else:
            book[price] = size

        # 如果滿足時間條件，就 snapshot 一次
        if prev_snap_ts is None or ts - prev_snap_ts >= snap_interval:
            bids_sorted = sorted(current_bids.items(), reverse=True)[:top_n]
            asks_sorted = sorted(current_asks.items())[:top_n]

            # 存到 orderbook_history（for JSON）
            snapshot = {
                "bids": dict(bids_sorted),
                "asks": dict(asks_sorted),
            }
            orderbook_history[ts] = snapshot

            # 存到 CSV rows
            for price, size in bids_sorted:
                csv_rows.append([ts, "bid", price, size])
            for price, size in asks_sorted:
                csv_rows.append([ts, "ask", price, size])

            prev_snap_ts = ts

    # save the JSON file
    with open(json_path, "w") as jf:
        json.dump(orderbook_history, jf)

    # save the CSV file
    df_out = pd.DataFrame(csv_rows, columns=["timestamp", "side", "price", "size"])
    df_out.to_csv(csv_path, index=False)

    return orderbook_history, df_out


df_small = df.head(100000)
sample_orderbook, sample_df = reconstruct_orderbook_fast(df_small)

# Save the result to JSON
sample_json_path = "data/reconstructed_orderbook_small.json"
with open(sample_json_path, "w") as f:
    json.dump(sample_orderbook, f)

sample_df.to_csv("data/reconstructed_orderbook_small.csv", index=False)
