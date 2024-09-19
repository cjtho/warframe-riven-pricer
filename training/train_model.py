import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.data_handler import WeaponsDataHandler
from warframe_marketplace_predictor.training.model_preprocessor import Preprocessor


def get_model_architecture():
    """
    Basically an embedding for the weapon name and a separate embedding for weapon traits.
    Followed by a simple neural network. Justification in github README.
    """

    # Load vocabularies
    weapon_data_handler = WeaponsDataHandler(load_weapon_expected_values=False)
    weapon_url_names = weapon_data_handler.get_url_names()
    attribute_names = weapon_data_handler.get_attribute_names()

    # Inputs
    weapon_url_name_input = layers.Input(shape=(1,), dtype=tf.string, name="weapon_url_name_input")
    re_rolls_input = layers.Input(shape=(1,), dtype=tf.float32, name="re_rolls_input")
    attribute_names_input = layers.Input(shape=(4,), dtype=tf.string, name="attribute_names_input")
    inputs = [weapon_url_name_input, re_rolls_input, attribute_names_input]

    # Lookups
    weapon_url_name_lookup = layers.StringLookup(vocabulary=weapon_url_names, mask_token="<NONE>",
                                                 name="weapon_url_name_lookup")
    attribute_names_lookup = layers.StringLookup(vocabulary=attribute_names, mask_token="<NONE>",
                                                 name="attribute_names_lookup")
    weapon_url_name_indices = weapon_url_name_lookup(weapon_url_name_input)
    attribute_names_indices = attribute_names_lookup(attribute_names_input)

    # Embeddings
    weapon_url_name_embedding_layer = layers.Embedding(input_dim=len(weapon_url_names) + 1,
                                                       output_dim=32,
                                                       name="weapon_url_name_embedding")
    attribute_names_embedding_layer = layers.Embedding(input_dim=len(attribute_names) + 1,
                                                       output_dim=32,
                                                       name="attribute_names_embedding")
    weapon_url_name_embedding_output = layers.Flatten()(weapon_url_name_embedding_layer(weapon_url_name_indices))
    attribute_names_embedding_output = layers.Flatten()(attribute_names_embedding_layer(attribute_names_indices))

    # Concatenate the embedding outputs
    flattened_everything = layers.Concatenate(name="concatenated_inputs")(  # what a garbage variable name
        [weapon_url_name_embedding_output, re_rolls_input, attribute_names_embedding_output]
    )

    # Cos I like softplus more than relu sue me
    x = layers.Dense(units=256, activation="softplus")(flattened_everything)
    x = layers.Dense(units=128, activation="softplus")(x)
    x = layers.Dense(units=32, activation="softplus")(x)
    output = layers.Dense(units=1, activation="linear")(x)  # regression task (predicting price)

    # Define the model
    model = Model(inputs=inputs, outputs=output, name="weapon_and_attributes_model")
    return model


def plot_performance(y_test, y_test_pred):
    # Calculate performance metrics
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    # Print the metrics
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Scatter plot of Actual vs Predicted values
    plt.figure(figsize=(16, 10))
    plt.scatter(y_test, y_test_pred, alpha=0.5,
                label="Predicted vs Actual")

    # Perfect prediction line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--",
             label="Perfect Prediction")

    # Line of best fit
    slope, intercept = np.polyfit(y_test, y_test_pred, 1)
    line_of_best_fit = slope * np.array(y_test) + intercept
    plt.plot(y_test, line_of_best_fit, color="blue", linestyle="-",
             label="Line of Best Fit")

    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Read in data
    try:
        df = pd.read_csv(marketplace_dataframe_file_path)
    except FileNotFoundError as f:
        print("Original Error:", f)
        print("You need to run 'setup_data' first.")
        exit()

    # Quick data examination
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", None)  # No max width for display
    pd.set_option("display.max_colwidth", None)  # No limit on column width
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.columns)
    print(df.describe())

    # Setup data
    df = df.fillna("<NONE>")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

    # Prepare your data
    features = ["weapon_url_name", "re_rolls",
                "positive1", "positive2", "positive3", "negative"]
    target = "listing_price"
    X = df[features]
    y = df[target]

    # Visual inspection of y's values justify this decision -- Trust me bro
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # Preprocessing data
    preprocessor = Preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    model = get_model_architecture()
    model.compile(optimizer="adam", loss="logcosh")
    model.summary()

    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train_preprocessed,
        y_train_log,
        epochs=100,
        validation_split=0.1,
        batch_size=256,
        callbacks=[early_stopping],
        verbose=1
    )

    y_test_log_pred = model.predict(X_test_preprocessed)
    y_test_pred = np.expm1(y_test_log_pred)
    y_test = np.expm1(y_test_log)

    plot_performance(y_test, y_test_pred)

    # Rebuild and recompile the model
    model = get_model_architecture()
    model.compile(optimizer="adam", loss="logcosh")

    preprocessor = Preprocessor()
    X_preprocessed = preprocessor.fit_transform(X)

    # Retrain the model on the full dataset
    model.fit(
        X_preprocessed,
        y_log,
        epochs=max(1, early_stopping.best_epoch),
        batch_size=256,
        verbose=1
    )

    # Save model.
    model.save(model_model_file_path)
    preprocessor.save(model_preprocessor_file_path)


if __name__ == "__main__":
    main()
