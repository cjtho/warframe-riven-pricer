from typing import Dict, Any

import pandas as pd
import tqdm

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.misc.bias_adjustor import adjust_values
from warframe_marketplace_predictor.shtuff.storage_handling import read_json


def create_df() -> None:
    """Creates a dataframe from raw marketplace data and saves it as a CSV."""
    df_rows = []

    # Load raw marketplace data
    marketplace_data = read_json(raw_marketplace_data_file_path)

    # Convert marketplace data into a pandas dataframe of auctions
    pbar = tqdm.tqdm(marketplace_data, desc="Auctions Processed", unit="auction", total=len(marketplace_data))
    for auction in pbar:
        df_row = dict()

        # This is the thing at the end of the warframe marketplace url
        # e.g. https://warframe.market/auction/66dec1f6c405f62fa6c92529
        # 66dec1f6c405f62fa6c92529 is auction["id"]
        df_row["id"] = auction["id"]
        df_row["created"] = auction["created"]

        item = auction["item"]
        df_row["weapon_url_name"] = item["weapon_url_name"]
        pbar.set_postfix(weapon=item["weapon_url_name"])  # slows it down lol ... but it pretty
        df_row["polarity"] = item["polarity"]
        df_row["mod_rank"] = item["mod_rank"]
        df_row["re_rolls"] = item["re_rolls"]
        df_row["master_level"] = item["mastery_level"]

        # Get riven attribute names and values
        attributes = item["attributes"]
        attribute_names = {"positive1": None, "positive2": None, "positive3": None, "negative": None}
        attribute_values = {"positive1_value": None, "positive2_value": None, "positive3_value": None,
                            "negative_value": None}
        i = 1
        for attribute in attributes:
            if attribute["positive"]:
                attribute_names[f"positive{i}"] = attribute["url_name"]
                attribute_values[f"positive{i}_value"] = attribute["value"]
                i += 1
            else:
                attribute_names["negative"] = attribute["url_name"]
                attribute_values["negative_value"] = attribute["value"]
        df_row.update(attribute_names)
        df_row.update(attribute_values)

        # Get prices associated with riven
        df_row["is_direct_sell"] = auction["is_direct_sell"]
        df_row["starting_price"] = auction["starting_price"]
        df_row["buyout_price"] = auction["buyout_price"]
        df_rows.append(df_row)

    # Save dataframe to CSV
    df = pd.DataFrame(df_rows)
    df.to_csv(marketplace_dataframe_file_path, index=False)
    print("Marketplace Dataframe created. You are ready to train your model.")


def handle_prices() -> None:
    """
    This function consolidates the starting and buyout prices into a single price. It filters out users who set
    unrealistic pricing, such as infinite maximum bids or excessively large spreads between starting and buyout
    prices. The goal is to return a dataset of more reasonable and focused price estimates.
    """
    data = pd.read_csv(marketplace_dataframe_file_path)
    original_size = data.shape[0]
    valid_data = data.dropna(subset=["buyout_price"])  # Drop rows with None buyout_price
    valid_data = valid_data[valid_data["buyout_price"] <= 10_000]  # Keep rows with buyout_price <= 10,000
    valid_data = valid_data[valid_data["buyout_price"] <= 5 * valid_data["starting_price"]]
    valid_data["listing_price"] = valid_data["buyout_price"]  # shhh
    valid_data.to_csv(marketplace_dataframe_file_path, index=False)
    print(f"Price column created. Dropped {original_size - valid_data.shape[0]} rows.")


def _process_weapon(weapon_name: str, df: pd.DataFrame, developer_summary_statistics: Dict[str, Any]):
    weapon_listings = df[df["weapon_url_name"] == weapon_name]
    listing_prices = weapon_listings["listing_price"]
    traded_summary_statistics = developer_summary_statistics[weapon_name]
    estimated_trade_prices = adjust_values(listing_prices, traded_summary_statistics)
    return weapon_name, estimated_trade_prices


def create_estimated_trade_price() -> None:
    """Attempts to shift the listed price distribution to more accurately reflect the traded price distribution."""

    # Read the data files
    df = pd.read_csv(marketplace_dataframe_file_path)
    developer_summary_statistics = read_json(developer_summary_stats_file_path)

    # Get unique weapon names
    weapon_names = df["weapon_url_name"].unique()

    results = []
    pbar = tqdm.tqdm(weapon_names, total=len(weapon_names), desc="Processing weapons")
    for weapon_name in pbar:
        estimated_trade_prices = _process_weapon(weapon_name, df, developer_summary_statistics)
        results.append((weapon_name, estimated_trade_prices))

    # Update the dataframe with the estimated trade prices
    for weapon_name, estimated_trade_prices in results:
        df.loc[df["weapon_url_name"] == weapon_name, "estimated_trade_price"] = estimated_trade_prices

    # Save the updated dataframe
    df.to_csv(marketplace_dataframe_file_path, index=False)
    print("Estimated trade price added via the traded summary statistics.")


def main():
    create_df()
    handle_prices()
    # create_estimated_trade_price()  # TODO: Underdevelopment


if __name__ == "__main__":
    main()
