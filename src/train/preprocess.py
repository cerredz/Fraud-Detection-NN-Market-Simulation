import pandas as pd
import numpy as np
import os
import random

def preprocess(n):
    pp_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Preprocessed")
    p_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Processed", "10mil")

    df1 = pd.read_parquet(f"{pp_folder_path}/cc-fraud.parquet").head(n)
    df2 = pd.read_csv(f"{pp_folder_path}/fraud.csv", nrows=n)
    
    df1 = df1.fillna({
        "gender": random.choice(["M", "F"]),
        'amt': df1['amt'].median(),
    })

    print(df1.columns)
    print(df2.columns)

    #  model 1 (personal info)
    full_names = df1["first"] + " " + df1["last"]
    personal_df = df1[['gender', 'dob', 'job', "is_fraud"]].copy()
    personal_df["name"] = full_names
    personal_df.to_csv(os.path.join(p_folder_path, f"personal_{n}.csv"), index=False)
    
    # model 2 (account info)
    df1_cols = df1[["ssn", "cc_num", "acct_num", "is_fraud"]]
    df2_cols = df2[["payment_channel"]]
    min_rows = min(len(df1_cols), len(df2_cols))
    df1_cols = df1_cols.head(min_rows)
    df2_cols = df2_cols.head(min_rows)
    account_df = pd.concat([df1_cols, df2_cols], axis=1)
    account_df.to_csv(os.path.join(p_folder_path, f"account_{n}.csv"), index=False)

    # model 3 (city / location )
    location_df = df1[["ssn", "city", "state", "zip", "is_fraud"]].copy()
    location_df["name"] = full_names
    location_df.to_csv(os.path.join(p_folder_path, f"location_{n}.csv"), index=False)

    # model 4 (time / amount)
    time_df = df1[["trans_time", "trans_date", "amt", "ssn", "is_fraud"]].copy() # ssn is to calc recent transactions / purchase amounts
    time_df.to_csv(os.path.join(p_folder_path, f"time_{n}.csv"), index=False)

    # model 5 (device / ip)
    device_df = df2[["device_used", "new_device_transaction", "device_hash", "ip_address", "is_fraud"]].copy()
    device_df.to_csv(os.path.join(p_folder_path, f"device_{n}.csv"), index=False)

    # model 6 (transaction details)
    trans_details_df = df2[["transaction_type", "merchant_category", "payment_channel", "is_fraud"]].copy()
    trans_details_df.to_csv(os.path.join(p_folder_path, f"trans_details_{n}.csv"), index=False)

    # model 7 (all features (general model))
    all_df1_cols = df1[["ssn", "cc_num", "gender", "city", "state", "zip", "dob", "job", "acct_num", "trans_time", "trans_date", "is_fraud"]].copy()
    all_df1_cols["name"] = full_names

    all_df2_cols = df2[["transaction_type", "merchant_category", "payment_channel", "ip_address", "device_hash", "device_used", "new_device_transaction"]].copy()
    all_min_rows = min(len(all_df1_cols), len(all_df2_cols))
    all_df1_cols = all_df1_cols.head(all_min_rows)
    all_df2_cols = all_df2_cols.head(all_min_rows)
    all_df_cols = pd.concat([all_df1_cols, all_df2_cols], axis=1)
    all_df_cols.to_csv(os.path.join(p_folder_path, f"all_{n}.csv"), index=False)

if __name__ == "__main__":
    preprocess(10000000)




