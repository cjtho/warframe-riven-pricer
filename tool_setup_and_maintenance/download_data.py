import time
from typing import Dict

import requests
import tqdm

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.storage_handling import save_json, read_json


# If anything breaks, surely it was a cosmic bit flip.


def fetch_data(url: str, delay: float = 0.1) -> Dict:
    """
    Fetches data from a given URL, with retries in case of rate limiting or failure.

    Args:
        url (str): The API endpoint to fetch data from.
        delay (float): The delay in seconds before retrying on rate limits. Defaults to 0.1.

    Returns:
        Dict: JSON data fetched from the API or an empty dictionary in case of an error.
    """
    try:
        response = requests.get(url, headers={"accept": "application/json"})
        # Handle rate-limiting (status code 429)
        if response.status_code == 429:  # Too Many Requests
            print("Rate limited. Retrying...")
            time.sleep(delay)
            return fetch_data(url, min(60.0, delay * 2))

        # Raise an exception for other HTTP errors
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")

    return dict()  # Fallback in case of any error


def download_items_data(the_url: str = "https://api.warframe.market/v1/riven/items") -> None:
    """
    Downloads item data from the API and saves mappings between item names and their URL representations.

    Args:
        the_url (str): The API endpoint to fetch item data from. Defaults to Warframe Riven items API.
    """
    items_data = fetch_data(the_url)["payload"]["items"]
    item_name_items_data = {x["item_name"]: x for x in items_data}
    save_json(items_data_file_path, item_name_items_data)
    print("Downloaded and saved items data.")


def download_attributes_data(the_url: str = "https://api.warframe.market/v1/riven/attributes") -> None:
    """
    Downloads attribute data from the API and saves it.

    Args:
        the_url (str): The API endpoint to fetch attribute data from. Defaults to Warframe Riven attributes API.
    """
    attributes_data = fetch_data(the_url)["payload"]["attributes"]
    attributes_data_mapped = {x["url_name"]: x for x in attributes_data if x["url_name"] not in ["has", "none"]}
    save_json(attributes_data_file_path, attributes_data_mapped)
    print("Downloaded and saved attributes data.")


def download_marketplace_database(overwrite: bool = True) -> None:
    """
    Downloads marketplace data and saves the raw data to a file.

    Args:
        overwrite (bool): If True, it downloads a fresh batch. If False, will update and append to existing data.
    """
    if overwrite:
        auctions_data = dict()
        original_length = 0
    else:
        auctions = read_json(raw_marketplace_data_file_path)
        auctions_data = {auction["id"]: auction for auction in auctions}
        original_length = len(auctions_data)

    weapon_url_names = [v["url_name"] for v in read_json(items_data_file_path).values()]

    price_orderings = ["price_asc", "price_desc"]
    pbar = tqdm.tqdm(weapon_url_names, "Fetching Marketplace Data", unit="weapon")
    for weapon_name in pbar:
        pbar.set_postfix(weapon=weapon_name, added=len(auctions_data) - original_length)
        for price_ordering in price_orderings:
            the_url = f"https://api.warframe.market/v1/auctions/search?type=riven"
            the_url += f"&weapon_url_name={weapon_name}"
            the_url += f"&sort_by={price_ordering}"
            auctions = fetch_data(the_url)["payload"]["auctions"]
            id_auctions = {auction["id"]: auction for auction in auctions}
            auctions_data.update(id_auctions)

    auctions_data = list(auctions_data.values())
    save_json(raw_marketplace_data_file_path, auctions_data)

    print(f"{len(auctions_data)} total entries. Marketplace data saved.")


def download_developer_riven_summary_stats(the_url: str = "https://api.warframestat.us/pc/rivens"):
    """
    Downloads and processes summary statistics for traded Rivens from the provided API, then saves the data.
    The data is organized into a dictionary with weapon names as keys and a dictionary containing rolled, unrolled,
    and combined statistics as values.

    Args:
        the_url (str): The URL of the API endpoint to retrieve Riven statistics. Defaults to the official
                       Warframe Rivens API for the PC platform.
    """
    # Load the mapping of item names to their respective URL-friendly names.
    items_data = read_json(items_data_file_path)

    # Fetch Riven summary statistics data from the API.
    riven_stats_data = fetch_data(the_url)

    # Initialize a dictionary to store reformatted summary statistics.
    reformatted_riven_stats = dict()

    for riven_type in riven_stats_data.values():
        for weapon_name, riven_stats in riven_type.items():
            if weapon_name not in items_data:
                continue

            url_name = items_data[weapon_name]["url_name"]
            stats_entry = {"rolled": None, "unrolled": None, "combined_stats": None}

            # Check and store unrolled statistics if available
            if "unrolled" in riven_stats:
                stats_entry["unrolled"] = riven_stats["unrolled"]

            # Check and store rerolled statistics if available
            if "rerolled" in riven_stats:
                stats_entry["rolled"] = riven_stats["rerolled"]

            # Combine unrolled and rerolled statistics if both are available
            if "unrolled" in riven_stats and "rerolled" in riven_stats:
                unrolled_stats = riven_stats["unrolled"]
                rerolled_stats = riven_stats["rerolled"]

                total_popularity = unrolled_stats["pop"] + rerolled_stats["pop"]
                unrolled_weight = unrolled_stats["pop"] / total_popularity
                rerolled_weight = rerolled_stats["pop"] / total_popularity

                combined_stats = {
                    "avg": (unrolled_weight * unrolled_stats["avg"]
                            + rerolled_weight * rerolled_stats["avg"]),
                    "stddev": (unrolled_weight * unrolled_stats["stddev"]
                               + rerolled_weight * rerolled_stats["stddev"]),
                    "median": (unrolled_weight * unrolled_stats["median"]
                               + rerolled_weight * rerolled_stats["median"]),
                    "min": min(unrolled_stats["min"], rerolled_stats["min"]),
                    "max": max(unrolled_stats["max"], rerolled_stats["max"])
                }
                stats_entry["combined_stats"] = combined_stats

            # Store the statistics entry for the weapon
            reformatted_riven_stats[url_name] = stats_entry

    # Save the reformatted Riven statistics to a JSON file.
    save_json(developer_summary_stats_file_path, reformatted_riven_stats)
    print("Downloaded and saved Riven summary statistics.")


def main(running_all: bool = False, overwrite_marketplace_data: bool = False):
    """
    Downloads all the data you'll need from the interweb.
    """

    if running_all or 0:
        # Very quick < 1 min
        download_items_data()

    if running_all or 0:
        # Very quick < 1 min
        download_attributes_data()

    if running_all or 0:
        # Takes 15 minutes I think.
        download_marketplace_database(overwrite=overwrite_marketplace_data)

    if running_all or 0:
        download_developer_riven_summary_stats()


if __name__ == "__main__":
    main()
