import collections
from typing import List, Dict, Any

import numpy as np
import tqdm

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.data_handler import WeaponsDataHandler
from warframe_marketplace_predictor.shtuff.make_prediction import PricePredictor
from warframe_marketplace_predictor.shtuff.storage_handling import save_json
from warframe_marketplace_predictor.riven_tool.tmp import get_possible_rivens


def calculate_attribute_importance(rivens: List[Dict[str, Any]], prices: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate the direct impact of each trait on the price of Rivens, distinguishing between positive and negative
    traits.

    Parameters:
    - rivens: List of dictionaries, each representing a Riven with keys "positives", "negatives", and "price".
    - prices: numpy array of prices corresponding to each Riven.

    Returns:
    - A dictionary with two keys "positive" and "negative", each containing trait impacts as normalized scores.
    """
    # Initialize dicts to track price sums and trait occurrences
    positive_trait_prices = collections.defaultdict(list)
    negative_trait_prices = collections.defaultdict(list)

    # Collect prices for each trait based on whether it's positive or negative
    for riven, price in zip(rivens, prices):
        for trait in riven.get("positives", []):
            if trait:  # Avoid empty strings
                positive_trait_prices[trait].append(price)
        for trait in riven.get("negatives", []):
            if trait:  # Avoid empty strings
                negative_trait_prices[trait].append(price)

        # Calculate global average price for normalization
    global_avg_price = np.mean(prices)

    positive_trait_impacts = {trait: float(np.mean(local_prices) / global_avg_price)
                              for trait, local_prices in positive_trait_prices.items()}
    negative_trait_impacts = {trait: float(np.mean(local_prices) / global_avg_price)
                              for trait, local_prices in negative_trait_prices.items()}

    return {
        "positive": positive_trait_impacts,
        "negative": negative_trait_impacts
    }


def calculate_expected_value(prices: np.ndarray, possible_re_rolled_rivens: List[Dict[str, Any]]) -> float:
    # Get current trait combination counts
    riven_attribute_combo_types = [f"p{len(x['positives'])}n{len(x['negatives'])}"
                                   for x in possible_re_rolled_rivens]
    attribute_combo_frequencies = collections.Counter(riven_attribute_combo_types)

    # Determine target count for each trait (since you want equal distribution)
    target_count = max(attribute_combo_frequencies.values())

    # Calculate the price distribution using the weighted data
    prices_pdf = np.array([target_count / attribute_combo_frequencies[trait_combo_type]
                           for trait_combo_type in riven_attribute_combo_types])
    prices_pdf /= np.sum(prices_pdf)
    sorted_prices_indices = np.argsort(prices)
    prices, prices_pdf = prices[sorted_prices_indices], prices_pdf[sorted_prices_indices]

    # Calculate the expected value (EV)
    expected_price = np.dot(prices, prices_pdf)
    return expected_price


def create_weapon_information():
    price_predictor = PricePredictor(model_predict_batch_size=4096, load_weapon_expected_values=False)
    weapon_data_handler = WeaponsDataHandler(load_weapon_expected_values=False)
    weapon_names = weapon_data_handler.get_url_names()

    rankings = []
    pbar = tqdm.tqdm(weapon_names, desc="Determining Characteristics")
    for weapon_name in pbar:
        pbar.set_postfix(weapon=weapon_name)

        weapon_attributes = weapon_data_handler.get_weapon_specific_attributes(weapon_name)
        possible_re_rolled_rivens = list(get_possible_rivens(weapon_name, 0, weapon_attributes))
        prices = price_predictor.predict(possible_re_rolled_rivens, verbose=False)

        attribute_importance = calculate_attribute_importance(possible_re_rolled_rivens, prices)

        expected_price = calculate_expected_value(prices, possible_re_rolled_rivens)
        rankings.append((weapon_name, expected_price, attribute_importance))

    # Sort the rankings by expected value
    rankings.sort(key=lambda x: x[1], reverse=True)
    rankings = {weapon_name: {"rank": i, "expected_value": expected_value, "attribute_importance": attribute_importance}
                for i, (weapon_name, expected_value, attribute_importance) in enumerate(rankings, start=1)}

    # Save the results to a JSON file
    save_json(weapon_information_file_path, rankings)
    print("Finished evaluating weapons.")


def main():
    create_weapon_information()  # ~20 min


if __name__ == "__main__":
    main()
