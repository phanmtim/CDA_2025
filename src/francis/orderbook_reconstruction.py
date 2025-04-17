# Re-import necessary libraries and reprocess the CSV after the session reset
import pandas as pd


def reconstruct_orderbook_every_tick_with_orders(df: pd.DataFrame, top_n: int = 50):
    current_bids = {}  # price: (size, number_of_orders)
    current_asks = {}
    orderbook_history = {}
    csv_rows = []

    grouped = df.groupby("timestamp")

    for ts, group in grouped:
        for row in group.itertuples(index=False):
            price = float(row.price)
            size = float(row.size)
            side = row.side
            num_orders = int(row.number_of_orders)

            book = current_bids if side == "bid" else current_asks

            if size == 0 and num_orders == 0:
                book.pop(price, None)
            else:
                book[price] = (size, num_orders)

        # sort + take top_n
        bids_sorted = sorted(current_bids.items(), reverse=True)[:top_n]
        asks_sorted = sorted(current_asks.items())[:top_n]

        # format to dict
        snapshot = {
            "bids": {
                price: {"size": s, "number_of_orders": n}
                for price, (s, n) in bids_sorted
            },
            "asks": {
                price: {"size": s, "number_of_orders": n}
                for price, (s, n) in asks_sorted
            },
        }

        orderbook_history[ts] = snapshot

        # add to csv rows
        for price, data in bids_sorted:
            csv_rows.append([ts, "bid", price, data[0], data[1]])
        for price, data in asks_sorted:
            csv_rows.append([ts, "ask", price, data[0], data[1]])

    return orderbook_history, csv_rows
