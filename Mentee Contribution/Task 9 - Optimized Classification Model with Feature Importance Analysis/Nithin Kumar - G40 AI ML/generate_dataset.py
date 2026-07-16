import pandas as pd
import numpy as np

np.random.seed(42)

number_of_customers = 500

data = {
    "Age": np.random.randint(18, 70, number_of_customers),
    "Annual_Income": np.random.randint(20000, 150000, number_of_customers),
    "Purchase_Frequency": np.random.randint(1, 30, number_of_customers),
    "Average_Order_Value": np.random.randint(500, 10000, number_of_customers),
    "Website_Visits": np.random.randint(1, 50, number_of_customers),
    "Discount_Used": np.random.randint(0, 2, number_of_customers),
}

df = pd.DataFrame(data)

df["Purchased"] = (
    (
        df["Annual_Income"] > 60000
    )
    & (
        df["Purchase_Frequency"] > 10
    )
).astype(int)

df.to_csv("data/customer_purchase_data.csv", index=False)

print("Dataset generated successfully.")
print(df.head())