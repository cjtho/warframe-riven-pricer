import collections
from typing import List, Dict, Any

import numpy as np
import tqdm

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.data_handler import WeaponsDataHandler
from warframe_marketplace_predictor.shtuff.make_prediction import PricePredictor
from warframe_marketplace_predictor.shtuff.storage_handling import save_json
from warframe_marketplace_predictor.riven_tool.tmp import get_possible_rivens


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

        # Get current trait combination counts
        riven_attribute_combo_types = [f"p{len(x['positives'])}n{len(x['negatives'])}"
                                       for x in possible_re_rolled_rivens]
        attribute_combo_frequencies = collections.Counter(riven_attribute_combo_types)

        # Determine target count for each trait (since you want equal distribution)
        target_count = max(attribute_combo_frequencies.values())

        # Calculate the price distribution using the weighted data
        prices_pdf = np.array([target_count / attribute_combo_frequencies[trait_combo_type]
                               for trait_combo_type in riven_attribute_combo_types])
        total_riven_amount = np.sum(prices_pdf)
        prices_pdf_norm = prices_pdf / total_riven_amount

        # Calculate the expected value (EV)
        expected_price = np.dot(prices_pdf_norm, prices)

        positive_trait_prices = collections.defaultdict(lambda: {"price": 0, "freq": 0})
        negative_trait_prices = collections.defaultdict(lambda: {"price": 0, "freq": 0})

        # Collect prices for each trait based on whether it's positive or negative
        for riven, freq, price in zip(possible_re_rolled_rivens, prices_pdf, prices):
            for trait in riven["positives"]:
                positive_trait_prices[trait]["price"] += freq * price
                positive_trait_prices[trait]["freq"] += freq
            for trait in riven["negatives"]:
                negative_trait_prices[trait]["price"] += freq * price
                negative_trait_prices[trait]["freq"] += freq

        positive_trait_impacts = {trait: float((local_prices["price"] / local_prices["freq"]) / expected_price)
                                  for trait, local_prices in positive_trait_prices.items()}
        negative_trait_impacts = {trait: float((local_prices["price"] / local_prices["freq"]) / expected_price)
                                  for trait, local_prices in negative_trait_prices.items()}

        attribute_importance = {
            "positive": positive_trait_impacts,
            "negative": negative_trait_impacts
        }

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
