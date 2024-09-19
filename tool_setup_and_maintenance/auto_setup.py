from warframe_marketplace_predictor import filepaths
from warframe_marketplace_predictor.tool_setup_and_maintenance import setup_weapon_information, download_data, \
    create_dataframe
from warframe_marketplace_predictor.training import train_model


def main(update=True):
    print("Verifying file paths...")
    filepaths.main()
    print("File paths verified.")

    print("Downloading marketplace data... (This may take approximately 15 minutes)")
    download_data.main(running_all=True, overwrite_marketplace_data=not update)
    print("Marketplace data downloaded successfully.")

    print("Creating training dataframe...")
    create_dataframe.main()
    print("Dataframe created successfully.")

    print("Training the model... (This may take approximately 5 minutes)")
    train_model.main()
    print("Model training completed.")

    print("Setting up weapon ranks... (This may take approximately 20 minutes)")
    setup_weapon_information.main()
    print("Weapon ranks setup complete.")

    print("Setup complete. You may now navigate to 'rivens_analysis',"
          " scroll to the bottom, input your rivens, and run the script.")


if __name__ == "__main__":
    main()
