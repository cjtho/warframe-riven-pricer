import json
from typing import List, Dict, Any

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.storage_handling import read_json


class WeaponsDataHandler:
    def __init__(self, load_items: bool = True, load_attributes: bool = True,
                 load_attribute_shortcuts: bool = True, load_weapon_expected_values: bool = True):
        if load_items:
            self.items_data_mapped_by_item_name = read_json(items_data_file_path)
            self.items_data_mapped_by_url_name = {v["url_name"]: v for v in
                                                  self.items_data_mapped_by_item_name.values()}
            self.name_inverse_mapping = {k: v["url_name"] for k, v in self.items_data_mapped_by_item_name.items()}
            self.name_inverse_mapping.update({v: k for k, v in self.name_inverse_mapping.items()})

        if load_attributes:
            self.attributes_data = read_json(attributes_data_file_path)

        if load_attribute_shortcuts:
            self.attribute_name_shortcuts = read_json(attribute_name_shortcuts_file_path)
            self.attribute_name_shortcuts.update({v: v for v in self.attribute_name_shortcuts.values()})

        if load_weapon_expected_values:
            self.weapon_rankings = read_json(weapon_information_file_path)

    def get_item_names(self) -> List[str]:
        return list(self.items_data_mapped_by_item_name.keys())

    def get_url_names(self) -> List[str]:
        return list(self.items_data_mapped_by_url_name.keys())

    def get_attribute_names(self) -> List[str]:
        return list(self.attributes_data.keys())

    def get_attribute_shortcuts(self) -> List[str]:
        return list(self.attribute_name_shortcuts.keys())

    def get_proper_attribute_name(self, attribute_name: str) -> str:
        return self.attribute_name_shortcuts[attribute_name]

    def get_url_name(self, weapon_name: str) -> str:
        if weapon_name in self.items_data_mapped_by_item_name:
            return self.name_inverse_mapping[weapon_name]
        elif weapon_name in self.items_data_mapped_by_url_name:
            return weapon_name
        else:
            raise ValueError(f"{weapon_name} does not exist.")

    def get_item_name(self, weapon_name: str) -> str:
        if weapon_name in self.items_data_mapped_by_url_name:
            return self.name_inverse_mapping[weapon_name]
        elif weapon_name in self.items_data_mapped_by_item_name:
            return weapon_name
        else:
            raise ValueError(f"{weapon_name} does not exist.")

    def weapon_exists(self, weapon_name: str) -> bool:
        return self.get_url_name(weapon_name) in self.items_data_mapped_by_url_name

    def is_valid_attribute_shortcut(self, attribute_name: str) -> bool:
        return attribute_name in self.attribute_name_shortcuts

    def get_weapon_specific_attributes(self, weapon_name: str) -> List[str]:
        weapon_name = self.get_url_name(weapon_name)
        weapon_type = self.items_data_mapped_by_url_name[weapon_name]["riven_type"]
        specific_attributes = [attribute_name for attribute_name, attribute_data in self.attributes_data.items()
                               if not attribute_data["exclusive_to"] or weapon_type in attribute_data["exclusive_to"]]
        return specific_attributes

    def get_weapon_rank_data(self, weapon_name: str) -> Dict[str, Any]:
        weapon_name = self.get_url_name(weapon_name)
        rank_data = {"total_ranks": len(self.weapon_rankings)}
        rank_data.update(self.weapon_rankings[weapon_name])
        return rank_data
