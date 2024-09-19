import collections
import hashlib
from typing import Dict, Any

from warframe_marketplace_predictor.shtuff.storage_handling import read_json, save_json
from warframe_marketplace_predictor.filepaths import *


class LRUCache:
    def __init__(self, max_size=100):
        self.cache = collections.OrderedDict()
        self.max_size = max_size

    def get(self, key):
        # Return the item if it exists and update its position to the front
        if key in self.cache:
            self.cache.move_to_end(key, last=False)
            return self.cache[key]
        return None

    def set(self, key, value):
        # If key exists, update it and move it to the front
        if key in self.cache:
            self.cache.move_to_end(key, last=False)
        else:
            # Add new item and check cache size
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=True)  # Remove the oldest item

    def clear(self):
        self.cache = collections.OrderedDict()

    def save(self, file_path: str = None):
        file_path = file_path if file_path else rivens_full_distribution_cache_file_path
        save_json(file_path, self.cache)

    def load(self, file_path: str = None):
        file_path = file_path if file_path else rivens_full_distribution_cache_file_path
        self.cache = read_json(file_path) if os.path.getsize(file_path) > 0 else dict()
        self.cache = collections.OrderedDict(self.cache)

    @staticmethod
    def generate_cache_key(riven: Dict[str, Any]) -> str:
        """Generate a unique key based on riven's attributes."""
        key = (
            riven["name"],
            tuple(sorted(riven["positives"])),
            tuple(sorted(riven["negatives"])),
            riven["re_rolls"]
        )
        key_str = ",".join(map(str, key))
        return hashlib.sha256(key_str.encode()).hexdigest()  # There is a simpler approach. I too dumb.
