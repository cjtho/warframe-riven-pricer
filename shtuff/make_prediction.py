from typing import List, Dict, Any, Union, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.data_handler import WeaponsDataHandler
from warframe_marketplace_predictor.training.model_preprocessor import Preprocessor


class PricePredictor:
    def __init__(self, model_predict_batch_size: int = 256, load_items: bool = True, load_attributes: bool = True,
                 load_attribute_shortcuts: bool = True, load_weapon_expected_values: bool = True):
        """
        Initializes the PricePredictor class, loading the preprocessor, model,
        attribute name shortcuts, and item name to URL mapping only once.
        """
        self.model_predict_batch_size = model_predict_batch_size
        self.weapon_data_handler = WeaponsDataHandler(load_items, load_attributes,
                                                      load_attribute_shortcuts, load_weapon_expected_values)

        # Load the preprocessor and model once
        self.preprocessor = Preprocessor().load()
        self.model = tf.keras.models.load_model(model_model_file_path)

        self._mask_token = "<NONE>"

    def is_valid(self, item: Dict[str, Any]) -> bool:
        """
        Checks if the provided attribute names (positives + negatives) are valid by comparing them against the shortcuts.

        Args:
            item (Dict[str, Any]): A dictionary containing the item's "positives" and "negatives" attributes.

        Returns:
            bool: True if all attribute names are valid, False otherwise.
        """
        if not self.weapon_data_handler.weapon_exists(item["name"]):
            print(f"{item['name']} is not a valid weapon name")
            print("Name suggestions:")
            print([k for k in sorted(self.weapon_data_handler.get_item_names())
                   if k and item["name"] and (k[0]).lower() == (item["name"][0]).lower()])
            return False

        if "re_rolls" not in item or not isinstance(item["re_rolls"], int):
            print("'re_rolls' is missing or incorrectly formatted.")
            return False

        # Combine the positives and negatives from the item to validate
        attribute_names = item["positives"] + item["negatives"]
        for attribute_name in attribute_names:
            if not self.weapon_data_handler.is_valid_attribute_shortcut(attribute_name):
                print(f"{attribute_name} is not a valid attribute.")
                print("Did you mean:")
                print([k for k in sorted(self.weapon_data_handler.get_attribute_shortcuts())
                       if k and attribute_name and k[0] == attribute_name[0]])
                return False

        return True

    def map_attribute_shortcuts(self, item: Dict[str, Any]) -> None:
        """
        Maps the potential attribute shortcuts to their proper names using the attribute_name_shortcuts dictionary.

        Args:
            item (Dict[str, Any]): The item dictionary containing positives and negatives.
        """
        item["positives"] = [self.weapon_data_handler.get_proper_attribute_name(x) for x in item["positives"]]
        item["negatives"] = [self.weapon_data_handler.get_proper_attribute_name(x) for x in item["negatives"]]

    def predict(self, data: Union[Iterable[Dict[str, Any]], Dict[str, Any]], verbose: bool = True) \
            -> Union[np.ndarray, np.float32]:
        """
        Predicts outcomes based on the provided input data using a pre-trained model.

        Args:
            data (Union[List[Dict[str, Any]], Dict[str, Any]]): Either a list of dictionaries where each dictionary contains
                                                                information about an item with attributes such as "positives",
                                                                "negatives", and "name", or a single dictionary of the same structure.

        Returns:
            np.array: A NumPy array containing the predictions from the model.
        """
        # If data is a single dictionary, wrap it in a list for consistent processing
        single_entry_flag = False
        if isinstance(data, dict):
            data = [data]
            single_entry_flag = True

        processed_data = []

        for item in data:
            row = dict()

            # Check for invalid names or shortcuts
            if not self.is_valid(item):
                return np.array([])

            # Map the item name to its corresponding URL name
            row["weapon_url_name"] = self.weapon_data_handler.get_url_name(item["name"])
            row["re_rolls"] = item["re_rolls"]

            # Map attribute shortcuts to proper names
            self.map_attribute_shortcuts(item)

            # Extract positive attributes and handle missing values
            attribute_values = {
                "positive1": item["positives"][0] if len(item["positives"]) >= 1 else self._mask_token,
                "positive2": item["positives"][1] if len(item["positives"]) >= 2 else self._mask_token,
                "positive3": item["positives"][2] if len(item["positives"]) >= 3 else self._mask_token,
                "negative": item["negatives"][0] if len(item["negatives"]) >= 1 else self._mask_token
            }
            row.update(attribute_values)

            processed_data.append(row)

        model_ready_data = self.preprocessor.transform(pd.DataFrame(processed_data))

        predictions = self.model.predict(model_ready_data,
                                         batch_size=self.model_predict_batch_size,
                                         verbose=verbose).reshape(-1)
        return np.expm1(predictions) if not single_entry_flag else np.expm1(predictions)[0]


def main():
    # Examples
    rivens = [
        {
            "name": "Sonicor",
            "positives": ["dmg", "cc", "impact"],
            "negatives": [""],
            "re_rolls": 0
        },
    ]

    predictor = PricePredictor()
    predictions = predictor.predict(rivens)

    for riven, prediction in zip(rivens, predictions):
        print(f"{riven['name']} riven is estimated to be listed at {prediction:.0f} platinum")


if __name__ == "__main__":
    main()
