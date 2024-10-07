# Warframe Riven Pricer

Slightly outdated. TODO: Upload the new version.

This project aims to predict the price of Warframe Rivens using a neural network trained on a comprehensive dataset from the Warframe Marketplace. Below is a detailed guide on setting up, training, and using the model.

## Setup

1. **Install Dependencies**
   - Ensure you have the necessary packages installed, primarily TensorFlow and essential data science libraries.

2. **Download and Prepare Data**
   - Run the `auto_setup.py` script to download and prepare Warframe Riven data from the Warframe Marketplace. This dataset includes various attributes of Rivens, such as their names and other relevant details.

3. **Make Predictions**
   - Use the `rivens_analysis` script to predict the price of new Rivens based on their attributes.

## How It Works

The model uses a neural network to estimate the market price of Rivens based on their attributes and reroll count. By analyzing a large dataset of Rivens and their listing prices, the model predicts how much a new Riven might be worth. 

The neural network was extensively tested with various levels of complexity in both its structure and data preparation. It was found that increasing model complexity had minimal impact on performance. Additionally, including detailed attributes such as mastery level, polarity, mod rank, disposition, popularity, and specific Riven values (e.g., crit_chance=97% vs. just crit_chance) did not significantly improve accuracy. 

The model provides an estimate of Riven value based on market trends but does not guarantee the actual selling price. The predicted value reflects how similar Rivens are valued by others and may differ from the final selling price. Marketplace prices may often be higher than actual traded prices, reflecting an upward shift in listing prices.

### Key Points
- **Dataset**: Contains approximately 200K Riven entries from the Warframe Marketplace.
- **Prediction**: The model offers a general idea of a Riven’s market value. It cannot predict the exact selling price but provides an estimate based on market listings.
- **Limitations**: The model may not account for every factor influencing Riven prices, such as specific attributes or nuances that affect actual trade values.

## Insights and Future Plans

1. **Value Prediction (DONE)**: The model could be expanded to predict the value of every possible Riven, offering a probability distribution over potential prices and insights into investment potential.
![](https://i.imgur.com/UJKjYV4.png)

3. **Outlier Detection**: Applying the model to the entire dataset may help identify undervalued or overvalued Rivens, highlighting potential bargains or overpriced items.

4. **Bias Correction**: Future work will focus on addressing model biases. One approach involves utilizing summary statistics of actually traded Rivens provided by the developers. By applying gradient descent (or similar techniques) to our predicted price distribution, we can adjust and shift it towards these true summary statistics. This method maps each value in the original distribution to a new value, forming a distribution that more closely aligns with the actual traded distribution.

Feel free to contribute to the project or reach out if you have any questions or suggestions!
