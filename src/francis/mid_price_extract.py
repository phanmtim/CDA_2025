import pandas as pd


def extract_mid_prices(csv_path):
    df = pd.read_csv(csv_path)

    # group by timestamp
    grouped = df.groupby("timestamp")

    mid_prices = []

    for ts, group in grouped:
        best_bid = group[group["side"] == "bid"]["price"].max()
        best_ask = group[group["side"] == "ask"]["price"].min()

        if pd.notna(best_bid) and pd.notna(best_ask):
            mid_price = (best_bid + best_ask) / 2
            mid_prices.append(
                {
                    "timestamp": ts,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price": mid_price,
                }
            )

    return pd.DataFrame(mid_prices)


# usage
mid_df = extract_mid_prices("data/reconstructed_orderbook_small.csv")
mid_df.to_csv("data/mid_price_small.csv", index=False)

# plot the chart
import matplotlib.pyplot as plt

# set the chart size
plt.figure(figsize=(12, 6))

# plot the mid_price chart
plt.plot(mid_df["timestamp"], mid_df["mid_price"], label="Mid Price")

# add the title and axis labels
plt.title("Mid Price Series")
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")

# show the chart
plt.legend()
plt.show()
