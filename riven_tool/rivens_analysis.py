from warframe_marketplace_predictor.riven_tool import tmp

if __name__ == "__main__":
    rivens = [
        {
            "name": "Mire",
            "positives": ["cd", "dmg", "additional"],
            "negatives": ["speed"],
            "re_rolls": 9
        },
        {
            "name": "Sibear",
            "positives": ["heavy", "cc", "cold"],
            "negatives": ["ccs"],
            "re_rolls": 9
        },
        {
            "name": "Ankyros",
            "positives": ["cold", "ccs", "cc"],
            "negatives": ["sd"],
            "re_rolls": 9
        },
        {
            "name": "Reaper Prime",
            "positives": ["heat", "corpus", "cc"],
            "negatives": [""],
            "re_rolls": 10
        },
        {
            "name": "Sporothrix",
            "positives": ["ms", "dmg", ""],
            "negatives": ["grineer"],
            "re_rolls": 48
        },
        {
            "name": "Sporothrix",
            "positives": ["speed", "cc", "toxin"],
            "negatives": ["reload"],
            "re_rolls": 48
        },
        {
            "name": "Acceltra",
            "positives": ["speed", "cd", "toxin"],
            "negatives": [""],
            "re_rolls": 9
        },
        {
            "name": "Tonkor",
            "positives": ["ms", "dmg", ""],
            "negatives": ["sd"],
            "re_rolls": 13
        },
        {
            "name": "Synapse",
            "positives": ["toxin", "ms", "cold"],
            "negatives": [""],
            "re_rolls": 11
        },
        {
            "name": "Phage",
            "positives": ["cc", "cd", ""],
            "negatives": ["reload"],
            "re_rolls": 10
        },
        {
            "name": "Praedos",
            "positives": ["heat", "sc", "cd"],
            "negatives": ["chance_combo"],
            "re_rolls": 23
        },
        {
            "name": "Nepheri",
            "positives": ["dmg", "cd", ""],
            "negatives": [""],
            "re_rolls": 9
        },
        {
            "name": "Sibear",
            "positives": ["heavy", "cc", "cold"],
            "negatives": ["ccs"],
            "re_rolls": 9
        },
        {
            "name": "Gorgon",
            "positives": ["cd", "ms", "zoom"],
            "negatives": ["mag"],
            "re_rolls": 20
        },
        {
            "name": "Nukor",
            "positives": ["mag", "ms", "toxin"],
            "negatives": ["infested"],
            "re_rolls": 19
        },
        {
            "name": "Sun & Moon",
            "positives": ["cd", "slash", "range"],
            "negatives": [""],
            "re_rolls": 18
        },
        {
            "name": "Nataruk",
            "positives": ["cc", "dmg", "electricity"],
            "negatives": ["sd"],
            "re_rolls": 30
        },
        {
            "name": "Soma",
            "positives": ["speed", "cc", ""],
            "negatives": [""],
            "re_rolls": 23
        },
        {
            "name": "Dokrahm",
            "positives": ["speed", "dmg", "slash"],
            "negatives": ["finisher"],
            "re_rolls": 12
        },
    ]

    tmp.main(rivens, use_cache=True, flush_cache=False)
