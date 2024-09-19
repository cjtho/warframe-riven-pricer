import os

# Get the absolute path to the directory containing this file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the folder names
data_folder_name = "data_files"
training_folder_name = "training"
model_data_folder_name = "model_data"

# Downloaded data
items_data_file_path = os.path.join(base_dir, data_folder_name, "items_data.json")
attributes_data_file_path = os.path.join(base_dir, data_folder_name, "attributes_data.json")
attribute_name_shortcuts_file_path = os.path.join(base_dir, data_folder_name, "attribute_name_shortcuts.json")
raw_marketplace_data_file_path = os.path.join(base_dir, data_folder_name, "raw_marketplace_data.json")
developer_summary_stats_file_path = os.path.join(base_dir, data_folder_name, "developer_summary_stats.json")

# Generated data
marketplace_dataframe_file_path = os.path.join(base_dir, training_folder_name, model_data_folder_name,
                                               "marketplace_dataframe.csv")
model_model_file_path = os.path.join(base_dir, training_folder_name, model_data_folder_name,
                                     "model.h5")
model_preprocessor_file_path = os.path.join(base_dir, training_folder_name, model_data_folder_name,
                                            "preprocessor.pkl")
rivens_full_distribution_cache_file_path = os.path.join(base_dir, data_folder_name,
                                                        "riven_full_distribution_cache.json")
weapon_information_file_path = os.path.join(base_dir, data_folder_name,
                                            "weapon_information.json")


def create_files_if_not_exist(paths):
    for path in paths:
        # Ensure the directory exists
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")

        # Check if the file already exists
        if not os.path.isfile(path):
            # Create the file if it doesn't exist
            with open(path, "w"):
                pass
            print(f"Created empty file: {path}")
        else:
            print(f"File already exists: {path}")


def main():
    # List of file paths to check and create if they don't exist
    file_paths = [
        items_data_file_path,
        attributes_data_file_path,
        attribute_name_shortcuts_file_path,
        raw_marketplace_data_file_path,
        developer_summary_stats_file_path,
        marketplace_dataframe_file_path,
        model_model_file_path,
        model_preprocessor_file_path,
        rivens_full_distribution_cache_file_path,
        weapon_information_file_path
    ]

    # Call the function to create files
    create_files_if_not_exist(file_paths)


if __name__ == "__main__":
    main()
