import pandas as pd
import matplotlib.pyplot as plt

# 讀取原始資料
df = pd.read_csv("data/reconstructed_orderbook_small.csv")


def plot_orderbook_over_time(df):

    # 先計算 mid price
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

    # 合併 mid price 回到原始資料，方便畫圖
    df = df.merge(mid_df, on="timestamp", how="inner")

    # 畫圖
    plt.figure(figsize=(14, 7))

    # 買單點點
    bid_df = df[df["side"] == "bid"]
    plt.scatter(
        bid_df["timestamp"],
        bid_df["price"],
        s=bid_df["size"] * 2,
        color="blue",
        alpha=0.4,
        label="Bid Orders",
    )

    # 賣單點點
    ask_df = df[df["side"] == "ask"]
    plt.scatter(
        ask_df["timestamp"],
        ask_df["price"],
        s=ask_df["size"] * 2,
        color="red",
        alpha=0.4,
        label="Ask Orders",
    )

    # Mid price 線
    plt.plot(
        mid_df["timestamp"],
        mid_df["mid_price"],
        color="green",
        linestyle="--",
        label="Mid Price",
    )

    # 標題與圖例
    plt.title("Orderbook Price Levels Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
