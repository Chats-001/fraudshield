# Dataset

Download `creditcard.csv.zip` from the
[upstream project link](https://clouda-labs-assets.s3-us-west-2.amazonaws.com/fraud-detection/creditcard.csv.zip),
extract it, and place the CSV at `data/raw/creditcard.csv`.

Expected columns: `Time`, `V1`–`V28`, `Amount`, and `Class`. Raw and processed data are ignored by
Git. The public dataset is highly imbalanced; FraudShield validates its schema before training.

