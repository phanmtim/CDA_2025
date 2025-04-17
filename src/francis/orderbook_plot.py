import pandas as pd
import matplotlib.pyplot as plt

# read original data
df = pd.read_csv("data/orderbook_data.csv")


def plot_orderbook_over_time(df):

    # calculate mid price
    def extract_mid_prices(df):
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
                        "mid_price": mid_price,
                    }
                )

        return pd.DataFrame(mid_prices)

    mid_df = extract_mid_prices(df)

    # merge mid price back to original data, for plotting
    df = df.merge(mid_df, on="timestamp", how="inner")

    # plot
    plt.figure(figsize=(14, 7))

    # bid points
    bid_df = df[df["side"] == "bid"]
    plt.scatter(
        bid_df["timestamp"],
        bid_df["price"],
        s=bid_df["size"] * 2,
        color="blue",
        alpha=0.4,
        label="Bid Orders",
    )

    # ask points
    ask_df = df[df["side"] == "ask"]
    plt.scatter(
        ask_df["timestamp"],
        ask_df["price"],
        s=ask_df["size"] * 2,
        color="red",
        alpha=0.4,
        label="Ask Orders",
    )

    # Mid price line
    plt.plot(
        mid_df["timestamp"],
        mid_df["mid_price"],
        color="green",
        linestyle="--",
        label="Mid Price",
    )

    # title and legend
    plt.title("Orderbook Price Levels Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
