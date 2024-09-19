import collections
import itertools
from typing import List, Dict, Any, Generator

import numpy as np
import tqdm
from prettytable import PrettyTable

from warframe_marketplace_predictor.shtuff.data_handler import WeaponsDataHandler
from warframe_marketplace_predictor.shtuff.lru_cache import LRUCache
from warframe_marketplace_predictor.shtuff.make_prediction import PricePredictor


def calculate_kuva_cost(re_rolls: int) -> int:
    return {
        0: 900,
        1: 1000,
        2: 1200,
        3: 1400,
        4: 1700,
        5: 2000,
        6: 2350,
        7: 2750,
        8: 3150,
        9: 3500
    }.get(re_rolls, 3500)


def get_possible_rivens(item_name: str, re_rolls: int, attributes: List[str]) -> Generator[Dict[str, Any], None, None]:
    elementals = ["cold_damage", "electric_damage", "heat_damage", "toxin_damage"]  # can't be negative
    for positive_count, negative_count in [(2, 0), (2, 1), (3, 0), (3, 1)]:
        for attribute_group in itertools.permutations(attributes, r=(positive_count + negative_count)):
            positives = attribute_group[:positive_count]
            negatives = attribute_group[positive_count:] if negative_count > 0 else tuple()
            if all(n not in elementals for n in negatives):
                yield {
                    "name": item_name,
                    "positives": positives,
                    "negatives": negatives,
                    "re_rolls": re_rolls
                }


def generate_table(rivens: List[Dict[str, Any]]) -> None:
    rivens.sort(key=lambda x: x["expected_profit_per_kuva"], reverse=True)
    kuva_scale = 1000

    key = {
        "Name": "Weapon Name",
        "Pos": "Positives",
        "Neg": "Negatives",
        "Rerolls": "Number of Rerolls",
        "WRank": "Weapon Ranking",
        "WList": "Weapon Expected Listing Price",
        "Prob (%)": "Probability to Roll a Better Riven (%)",
        "LPrice": "Predicted Riven Listing Price",
        "LExpRoll": "Expected Riven Listing Price on Reroll",
        "LProfitRoll": "Expected Riven Listing Profit on Reroll",
        f"LProfit/{kuva_scale}K": f"Expected Riven Listing Profit per {kuva_scale} Kuva",
    }

    table = PrettyTable()
    table.field_names = list(key.keys())
    for riven in rivens:
        table.add_row([
            riven["name"],
            ", ".join(riven["positives"]),
            ", ".join(riven["negatives"]),
            riven["re_rolls"],
            riven["weapon_ranking"],
            f"{riven['weapon_expected_list_price']:.0f}",
            f"{100 * riven['probability_to_roll_better']:.1f}",
            f"{riven['listing_price']:.0f}",
            f"{riven['expected_price_per_reroll']:.0f}",
            f"{riven['expected_profit_per_reroll']:.0f}",
            f"{kuva_scale * riven['expected_profit_per_kuva']:.1f}",
        ])
    print(table)

    key_table = PrettyTable()
    key_table.field_names = ["Key", "Meaning"]
    for ev, full_text in key.items():
        key_table.add_row([ev, full_text])
    print(key_table)


def main(rivens: List[Dict[str, Any]], use_cache: bool = True, flush_cache: bool = False) -> None:
    price_predictor = PricePredictor(model_predict_batch_size=4096)
    weapons_data_handler = WeaponsDataHandler()

    # Load cache
    cache = LRUCache(max_size=1000)
    if use_cache:
        cache.load()
    if flush_cache:
        cache.clear()
        cache.save()

    # Check and validate riven attributes
    if not all(map(price_predictor.is_valid, rivens)):
        return

    pbar = tqdm.tqdm(rivens, desc="Analysing Rivens")
    for riven in pbar:
        pbar.set_postfix(riven=riven["name"])

        price_predictor.map_attribute_shortcuts(riven)

        cache_key = cache.generate_cache_key(riven)
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data:
                riven.update(cached_data)
                continue

        # Predict the prices of all rivens
        weapon_attributes = weapons_data_handler.get_weapon_specific_attributes(riven["name"])
        possible_re_rolled_rivens = list(get_possible_rivens(riven["name"], riven["re_rolls"] + 1, weapon_attributes))
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
        prices_pdf /= np.sum(prices_pdf)  # normalize
        sorted_prices_indices = np.argsort(prices)
        prices, prices_pdf = prices[sorted_prices_indices], prices_pdf[sorted_prices_indices]
        prices_cdf = np.cumsum(prices_pdf)
        prices_cdf /= prices_cdf[-1]

        listing_price = price_predictor.predict(riven, verbose=False)
        price_position = np.searchsorted(prices, listing_price, side="right")

        if price_position >= len(prices) - 1:
            probability_stagnant_roll = 1
            expected_improved_listing_price = listing_price
        else:
            probability_stagnant_roll = prices_cdf[price_position]
            improved_prices = prices[price_position + 1:]
            improved_prices_pdf = prices_pdf[price_position + 1:]
            improved_prices_pdf /= np.sum(improved_prices_pdf)
            expected_improved_listing_price = np.dot(improved_prices, improved_prices_pdf)

        expected_price_per_reroll = ((probability_stagnant_roll * listing_price)
                                     + (1 - probability_stagnant_roll) * expected_improved_listing_price)
        expected_profit_per_reroll = expected_price_per_reroll - listing_price
        kuva_cost = calculate_kuva_cost(riven["re_rolls"])
        expected_profit_per_kuva = expected_profit_per_reroll / kuva_cost

        # Calculate weapon-related info
        rank_data = weapons_data_handler.get_weapon_rank_data(riven["name"])
        rank = rank_data["rank"]
        expected_value = rank_data["expected_value"]
        total_ranks = rank_data["total_ranks"]
        weapon_ranking = f"{rank}/{total_ranks}"

        riven.update({
            "weapon_ranking": weapon_ranking,
            "weapon_expected_list_price": expected_value,
            "listing_price": listing_price,
            "probability_to_roll_better": (1 - probability_stagnant_roll),
            "expected_price_per_reroll": expected_price_per_reroll,
            "expected_profit_per_reroll": expected_profit_per_reroll,
            "expected_profit_per_kuva": expected_profit_per_kuva
        })

        # Cache the calculated values
        if use_cache:
            cache.set(cache_key, {k: float(v) if not isinstance(v, (str, list)) else v for k, v in riven.items()})

    if use_cache:
        cache.save()

    generate_table(rivens)
