import pandas as pd
import csv
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil
from tqdm import tqdm
from tabulate import tabulate
from fpdf import FPDF
from tabulate import tabulate
import numpy as np
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split
import ftfy
import math



context_datasets = [
    "MUStARD",
    "SPIRS",
    "SARC"
]

datasets = [
    "MUStARD",
    "Riloff",
    "SemEval2022",
    "iSarcasmEval",
    "SARC",
    "SPIRS",
    "multimodal_sarcasm_detection",
    "Ptacek"
    #"MSTI",

]





def add_column_to_csv_from_other(file_path_to_add, file_path_to_save, column_name_to_add):
    try:
        # Load the CSV files using the fallback methods
        df_add = read_csv_with_fallback(file_path_to_add)
        df_save = read_csv_with_fallback(file_path_to_save)

        # Ensure the column exists in the file to add
        if column_name_to_add not in df_add.columns:
            raise ValueError(f"Column {column_name_to_add} not found in {file_path_to_add}")

        # Ensure the key columns exist in both files (text, dataset, label)
        for col in ['text', 'dataset', 'label']:
            if col not in df_add.columns or col not in df_save.columns:
                raise ValueError(f"Column {col} is missing in one of the files")

        # Make sure the new column exists in df_save; if not, initialize with empty strings
        if column_name_to_add not in df_save.columns:
            df_save[column_name_to_add] = [''] * len(df_save)

        # Loop through each row in df_save and find the corresponding row in df_add
        with tqdm(range(len(df_save)), desc="Updating rows in file") as progress_bar:
            for row in progress_bar:
                text_save = df_save.at[row, 'text']
                dataset_save = df_save.at[row, 'dataset']
                label_save = df_save.at[row, 'label']

                # Find the matching row in df_add based on text, dataset, and label
                matching_row = df_add[(df_add['text'] == text_save) &
                                      (df_add['dataset'] == dataset_save) &
                                      (df_add['label'] == label_save)]

                if not matching_row.empty:
                    # Add the value from column_name_to_add to df_save
                    df_save.at[row, column_name_to_add] = matching_row.iloc[0][column_name_to_add]

        # Save the updated CSV using the fallback method
        save_csv_with_fallback(file_path_to_save, df_save)
        print(f"Updated CSV saved to {file_path_to_save}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def save_random_subset(original_file_path, to_save_path):
    # Load the original file
    df = read_csv_with_fallback(original_file_path)

    # Ensure that the file has the required columns
    if set(['label', 'text', 'dataset']).issubset(df.columns):
        
        # Get all unique datasets
        unique_datasets = df['dataset'].unique()

        # Create an empty list to store filtered data
        filtered_data = []

        # Process each dataset
        for dataset in unique_datasets:
            dataset_df = df[df['dataset'] == dataset]
            if len(dataset_df) > 2500:
                # Select 2500 random rows if the dataset has more than 2500 rows
                dataset_df = dataset_df.sample(n=2500, random_state=42)
            # Append to the filtered data
            filtered_data.append(dataset_df)

        # Concatenate all the filtered dataframes
        final_df = pd.concat(filtered_data)

        # Save the new file to the specified path
        save_csv_with_fallback(to_save_path, final_df)

    else:
        print("The input file does not have the required columns: label, text, and dataset")


def analyze_file(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize an empty DataFrame to store the results with custom titles
    results = pd.DataFrame(columns=[
        'Dataset', 'total_samples', 
        'all right', 'all wrong',
        'both help', 'only rag_in help', 'only rag_all help',        
        'rag_all interrupted', 'rag_in interrupted', 'both interrupted'
        
    ])
    
    # Group by 'dataset' and analyze each subset separately
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        df_subset = df[df['dataset'] == dataset]
        
        # Calculate the total number of samples for the dataset
        total_samples = df_subset.shape[0]
        
        # Calculate values for each combination
        all_right = df_subset.query("zero_shot == label & rag_in == label & rag_all == label").shape[0]
        all_wrong = df_subset.query("zero_shot != label & rag_in != label & rag_all != label").shape[0] 
        both_help = df_subset.query("zero_shot != label & rag_in == label & rag_all == label").shape[0]
        only_rag_in_help = df_subset.query("zero_shot != label & rag_in == label & rag_all != label").shape[0]
        only_rag_all_help = df_subset.query("zero_shot != label & rag_in != label & rag_all == label").shape[0]               
        rag_all_interrupted = df_subset.query("zero_shot == label & rag_in == label & rag_all != label").shape[0]
        rag_in_interrupted = df_subset.query("zero_shot == label & rag_in != label & rag_all == label").shape[0]
        both_interrupted = df_subset.query("zero_shot == label & rag_in != label & rag_all != label").shape[0]        

        
        # Calculate percentages
        all_right_percentage = (all_right / total_samples) * 100 if total_samples > 0 else 0
        all_wrong_percentage = (all_wrong / total_samples) * 100 if total_samples > 0 else 0   
        both_help_percentage = (both_help / total_samples) * 100 if total_samples > 0 else 0
        only_rag_in_help_percentage = (only_rag_in_help / total_samples) * 100 if total_samples > 0 else 0
        only_rag_all_help_percentage = (only_rag_all_help / total_samples) * 100 if total_samples > 0 else 0             
        rag_all_interrupted_percentage = (rag_all_interrupted / total_samples) * 100 if total_samples > 0 else 0
        rag_in_interrupted_percentage = (rag_in_interrupted / total_samples) * 100 if total_samples > 0 else 0
        both_interrupted_percentage = (both_interrupted / total_samples) * 100 if total_samples > 0 else 0
        
        # Create a new DataFrame for the current dataset's results with a single row
        new_row = pd.DataFrame({
            'Dataset': [dataset],
            'total_samples': [total_samples],
            'all right': [f"{all_right} ({all_right_percentage:.2f}%)"],
            'all wrong': [f"{all_wrong} ({all_wrong_percentage:.2f}%)"],            
            'both help': [f"{both_help} ({both_help_percentage:.2f}%)"],
            'only rag_in help': [f"{only_rag_in_help} ({only_rag_in_help_percentage:.2f}%)"],
            'only rag_all help': [f"{only_rag_all_help} ({only_rag_all_help_percentage:.2f}%)"],            
            'rag_all interrupted': [f"{rag_all_interrupted} ({rag_all_interrupted_percentage:.2f}%)"],
            'rag_in interrupted': [f"{rag_in_interrupted} ({rag_in_interrupted_percentage:.2f}%)"],
            'both interrupted': [f"{both_interrupted} ({both_interrupted_percentage:.2f}%)"]
        })
        
        # Concatenate the new row with the results DataFrame
        results = pd.concat([results, new_row], ignore_index=True)
    
    # Display the DataFrame
    display(results)


def split_csv(file_path, column_name, num_splits):
    # Load the CSV file
    df = read_csv_with_fallback(file_path)
    
    # Filter rows with empty values (NaN or empty string) in the specified column
    empty_rows_df = df[df[column_name].isna() | (df[column_name] == "")]
    
    # Calculate the size of each split in terms of empty values
    split_size = math.ceil(len(empty_rows_df) / num_splits)
    
    # Initialize variables
    base_dir = os.path.dirname(file_path)
    file_paths = []
    chunk_dfs = []
    empty_count = 0
    chunk_number = 1
    
    for i, row in df.iterrows():
        chunk_dfs.append(row)
        if pd.isna(row[column_name]) or row[column_name] == "":
            empty_count += 1
        
        # Check if the current chunk has reached the desired number of empty values
        if empty_count >= split_size:
            # Create a DataFrame for the current chunk
            chunk_df = pd.DataFrame(chunk_dfs)
            split_file_path = os.path.join(base_dir, f"part_{chunk_number}.csv")
            save_csv_with_fallback(split_file_path, chunk_df)            
            file_paths.append(split_file_path)
            
            # Reset for the next chunk
            chunk_number += 1
            empty_count = 0
            chunk_dfs = []
    
    # Save any remaining rows as the last chunk
    if chunk_dfs:
        chunk_df = pd.DataFrame(chunk_dfs)
        split_file_path = os.path.join(base_dir, f"part_{chunk_number}.csv")
        save_csv_with_fallback(split_file_path, chunk_df)
        file_paths.append(split_file_path)
    
    return file_paths

def combine_csv(file_paths, output_file_path):
    combined_df = pd.concat([read_csv_with_fallback(fp) for fp in file_paths])
    save_csv_with_fallback(output_file_path, combined_df)
    #print(f"Combined files into {output_file_path}")


def get_f1_scores_with_datasets(column_names, file_path):
    # Load the dataset
    data = pd.read_csv(file_path, encoding='latin1')
    
    for column_name in column_names:
        print(f"F1 Scores for {column_name} prompt:")
        
        # Extract the relevant columns
        results = data[['dataset', 'label', column_name]]

        # Grouping the data by dataset and analyzing the results
        grouped = results.groupby('dataset')

        # Initialize lists to store F1 scores across all datasets
        all_macro_f1_1 = []
        all_weighted_f1_1 = []
        all_f1_class_1 = []
        
        all_macro_f1_0 = []
        all_weighted_f1_0 = []
        all_f1_class_0 = []

        all_custom_f1_pos = []
        all_custom_f1_neg = []

        for dataset_name, group in grouped:
            y_true = group['label']
            y_pred = group[column_name]

            # Calculate true positives, true negatives, false positives, and false negatives
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            # Custom F1 calculations
            f1_pos = 2 * tp / ((2 * tp) + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            f1_neg = 2 * tn / ((2 * tn) + fn + fp) if (2 * tn + fn + fp) > 0 else 0

            # Calculate standard F1 scores using sklearn
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
            f1_class_1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
            f1_class_0 = f1_score(y_true, y_pred, average='binary', pos_label=0)

            # Store the results
            all_macro_f1_1.append(macro_f1)
            all_weighted_f1_1.append(weighted_f1)
            all_f1_class_1.append(f1_class_1)
            all_f1_class_0.append(f1_class_0)

            all_custom_f1_pos.append(f1_pos)
            all_custom_f1_neg.append(f1_neg)

            # Print all F1 scores for the current dataset in one row
            print(f"Dataset: {dataset_name} | F1 Class 0 (f1_neg): {f1_neg:.3f} and {f1_class_0:.2f} | F1 Class 1 (f1_pos): {f1_pos:.3f} and {f1_class_1:.2f} | Macro F1: {macro_f1:.2f} | Weighted F1: {weighted_f1:.3f}")
            print("-----------------------------")
        
        # Calculate the average F1 scores across all datasets
        avg_macro_f1_1 = sum(all_macro_f1_1) / len(all_macro_f1_1)
        avg_weighted_f1_1 = sum(all_weighted_f1_1) / len(all_weighted_f1_1)
        avg_f1_class_1 = sum(all_f1_class_1) / len(all_f1_class_1)
        avg_f1_class_0 = sum(all_f1_class_0) / len(all_f1_class_0)

        avg_custom_f1_pos = sum(all_custom_f1_pos) / len(all_custom_f1_pos)
        avg_custom_f1_neg = sum(all_custom_f1_neg) / len(all_custom_f1_neg)
        
        # Print the average F1 scores
        print("\nAverage F1 Scores across all datasets:")
        print(f"  Average F1 Score for class 1 (f1_pos): {avg_f1_class_1:.2f}")
        print(f"  Average F1 Score for class 0 (f1_neg): {avg_f1_class_0:.2f}")
        print(f"  Average F1 Score (macro): {avg_macro_f1_1:.2f}")
        print(f"  Average F1 Score (weighted): {avg_weighted_f1_1:.2f}")
        print(f"  Custom Average F1 Score (f1_pos): {avg_custom_f1_pos:.2f}")
        print(f"  Custom Average F1 Score (f1_neg): {avg_custom_f1_neg:.2f}")
        print("\n+++++++++++++++++++++\n")



def read_csv_with_fallback(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, encoding='latin1')
        df = df.applymap(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        df = df.applymap(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)        
        return df

def save_csv_with_fallback(csv_file_path, df):
    #df = df.applymap(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
    try:
        return df.to_csv(csv_file_path, encoding='latin1', index=False)
    except UnicodeEncodeError:
        return df.to_csv(csv_file_path, encoding='utf-8', index=False)
        
def get_column_content(csv_file_path, column_name):
    try:
        # Read the CSV file with fallback encoding
        df = read_csv_with_fallback(csv_file_path)

        # Check if the column exists
        if column_name in df.columns:
            # Extract the column content and return it as a list
            return df[column_name].tolist()
        else:
            raise ValueError(f"Column '{column_name}' does not exist in the CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def add_column_to_csv(csv_file_path, new_column_name, list_of_new_data):
    try:
        # Read the CSV file with fallback encoding
        df = read_csv_with_fallback(csv_file_path)

        # Check if the length of new data matches the number of rows in the CSV
        if len(list_of_new_data) != len(df):
            raise ValueError("Length of new data does not match the number of rows in the CSV file.")

        # Add the new column to the DataFrame
        df[new_column_name] = list_of_new_data

        # Attempt to write the updated DataFrame back to the CSV file with UTF-8 encoding, fallback to Latin-1
        save_csv_with_fallback(csv_file_path, df)

        print(f"New column '{new_column_name}' added successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def fix_column_text(file_path, column_name, output_file_path):
    # Step 1: Read the CSV file with latin1 encoding (or you can adjust if you know the encoding)
    df = pd.read_csv(file_path, encoding='latin1')

    # Step 2: Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the file.")
        return

    # Step 3: Apply ftfy.fix_text to the specified column, converting NaN to empty string
    def safe_fix_text(value):
        if pd.isna(value):  # Check if the value is NaN
            return ""  # Convert NaN to empty string
        elif isinstance(value, str):
            return ftfy.fix_text(value)
        else:
            return str(value)  # Convert non-string values to string

    df[column_name] = df[column_name].apply(safe_fix_text)

    # Step 4: Save the DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False, encoding='utf-8')

    print(f"File saved successfully to '{output_file_path}'.")


def compare_files(wanted_file, check_file, column_name):
    # Read the files into DataFrames with latin1 encoding
    wanted_df = pd.read_csv(wanted_file, encoding='latin1')
    check_df = pd.read_csv(check_file, encoding='latin1')

    # Check if the specified column exists in both DataFrames
    if column_name not in wanted_df.columns or column_name not in check_df.columns:
        print(f"Column '{column_name}' not found in one or both files.")
        return

    # Get the minimum number of rows to avoid out-of-bound errors
    min_rows = min(len(wanted_df), len(check_df))

    # Iterate over each row in the wanted file and compare with check file
    for i in range(min_rows):
        wanted_value = wanted_df.at[i, column_name]
        check_value = check_df.at[i, column_name]

        # Print the row number if the values are not the same
        if wanted_value != check_value:
            print(f"Difference found in row {i + 1} of the check file: wanted value = '{wanted_value}', check value = '{check_value}'")

    # If wanted_df has more rows than check_df, note that those rows have no comparison
    if len(wanted_df) > len(check_df):
        print(f"Note: wanted_file has {len(wanted_df) - len(check_df)} more rows than check_file.")

    # If check_df has more rows than wanted_df, note that those rows have no comparison
    if len(check_df) > len(wanted_df):
        print(f"Note: check_file has {len(check_df) - len(wanted_df)} more rows than wanted_file.")


def replace_invalid_labels(file_path, label_name, output_file_path):
    """
    Replace invalid values in the specified column with NaN and save the cleaned DataFrame.

    :param file_path: str, path to the CSV file.
    :param label_name: str, the name of the column to check.
    :param output_file_path: str, path to save the cleaned CSV file.
    """
    try:
        # Attempt to specify dtype for the column to avoid mixed types
        df = pd.read_csv(file_path, encoding='latin1', dtype={label_name: 'object'}, low_memory=False)
    except ValueError as e:
        print(f"Error reading the file with specified dtype: {e}")
        return
    
    # Convert the column to numeric, coercing errors to NaN
    df[label_name] = pd.to_numeric(df[label_name], errors='coerce')
    
    # Identify invalid rows where the value is not 0 or 1
    invalid_rows = df[(df[label_name] != 0) & (df[label_name] != 1)]
    
    if not invalid_rows.empty:
        print(f"Replacing invalid values in column '{label_name}' with NaN:")
        print(invalid_rows.index.tolist())
        
        # Replace invalid values with NaN
        df.loc[invalid_rows.index, label_name] = np.nan
        
        # Save the cleaned DataFrame to a new CSV file
        df.to_csv(output_file_path, index=False, encoding='latin1')
        print(f"Cleaned data saved to {output_file_path}")
    else:
        print("All values in the column are either 0 or 1. No replacements made.")


def balance_dataset(file_path):
    # Read the CSV file with encoding 'latin1'
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Group by the dataset and label columns
    grouped = df.groupby(['dataset', 'label'])

    # Initialize a list to hold balanced dataframes
    balanced_dfs = []

    # Iterate over each group (each unique dataset)
    for dataset_name, group in df.groupby('dataset'):
        # Separate the group into label 0 and label 1
        label_0 = group[group['label'] == 0]
        label_1 = group[group['label'] == 1]

        # Determine the minimum size between the two labels
        min_size = min(len(label_0), len(label_1))

        # Sample the minimum number of rows from both labels to balance the dataset
        label_0_balanced = label_0.sample(n=min_size, random_state=109)
        label_1_balanced = label_1.sample(n=min_size, random_state=109)

        # Concatenate the balanced dataframes for label 0 and label 1
        balanced_df = pd.concat([label_0_balanced, label_1_balanced])

        # Append the balanced dataframe to the list
        balanced_dfs.append(balanced_df)

    # Concatenate all balanced datasets into one dataframe
    balanced_df_final = pd.concat(balanced_dfs)

    # Save the balanced dataframe to a new CSV file
    file_name_without_extension = file_path.replace(".csv", "")

    output_file_path = f"{file_name_without_extension}_equal.csv"
    balanced_df_final.to_csv(output_file_path, index=False, encoding='latin1')

    print(f"Balanced dataset saved to {output_file_path}")


def split_data(file_path):
    
    input_path = f'{file_path}data.csv'

    # Read the input CSV file
    data = pd.read_csv(input_path, encoding='latin1')
    
    # Split the data into 80% training and 20% testing
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save the resulting data into new CSV files
    train_data.to_csv(f'{file_path}train_data.csv', index=False)
    test_data.to_csv(f'{file_path}test_data.csv', index=False)
    
    print("Data successfully split and saved as 'train_data.csv' and 'test_data.csv'.")

# Function to count total samples, label 0, label 1 for each dataset
def count_samples_labels(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    results = []
    grouped_data = data.groupby('dataset')
    
    for dataset, group in grouped_data:
        total_samples = len(group)
        label_0_count = len(group[group['label'] == 0])
        label_1_count = len(group[group['label'] == 1])
        
        results.append({
            'dataset': dataset,
            'total_samples': total_samples,
            'label_0': label_0_count,
            'label_1': label_1_count
        })

    return results


def merge_csv(data_name, input_path):
    all_data = []

    for dataset in datasets:
        file_path = f'{input_path}{dataset}/{data_name}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='latin1')
            all_data.append(df)
        else:
            print(f"File not found: {file_path}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        to_save = f'{input_path}all/{data_name}_all.csv'
        merged_df.to_csv(to_save, index=False)
        print(f"Data saved to {to_save}")
    else:
        print("No data files were found and merged.")


def load_csv_as_dict(file_path, encoding='latin1'):
    with open(file_path, mode='r', encoding=encoding) as file:
        reader = csv.DictReader(file)
        if 'text' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError(f"CSV file at {file_path} must contain 'text' and 'label' columns.")
        return [{'text': row['text'], 'label': row['label'], 'dataset': row['dataset']} for row in reader if row['text'].strip() != '']


def save_dict_as_csv(data, file_path, encoding='latin1'):
    with open(file_path, mode='w', newline='', encoding=encoding) as file:
        fieldnames = ['label', 'text', 'dataset']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def contains_english(text):
    # Remove URLs from the text
    text_without_urls = re.sub(r'http\S+', '', text)
    # Extract words from the cleaned text
    words = re.findall(r'\b\w+\b', text_without_urls)
    # Check if any word contains English letters
    for word in words:
        if re.search('[a-zA-Z]', word):
            return True
    return False


def normalize_text(text):
    # Remove leading and trailing spaces, and normalize internal spaces
    return re.sub(r'\s+', ' ', text).strip()


def filter_duplicates_and_non_english_save_train_data(input_path):
    data_dicts = {}
    test_data_dicts = {}

    # Load data and duplicate files
    for dataset in datasets:
        file_path = f'{input_path}{dataset}/train_data.csv'
        test_path = f'{input_path}{dataset}/test_data.csv'
        to_save = f'{input_path}{dataset}/train_data_filtered.csv'
        
        # Copy the original file to the new file path
        shutil.copy(file_path, to_save)
        
        # Load the CSV data as dictionaries
        data_dicts[dataset] = load_csv_as_dict(to_save)
        test_data_dicts[dataset] = load_csv_as_dict(test_path)
    
    # Track texts to remove and count duplicates with different labels
    texts_to_remove = defaultdict(set)
    train_test_duplicates_count = defaultdict(int)
    inter_dataset_duplicates_count = defaultdict(int)
    train_other_test_duplicates_count = defaultdict(int)
    different_label_count = defaultdict(lambda: defaultdict(int))

    # Check for duplicates between each pair of datasets (train data)
    for i, dataset1 in enumerate(datasets):
        for j, dataset2 in enumerate(datasets):
            if i < j:
                texts1 = {normalize_text(item['text']): item['label'] for item in data_dicts[dataset1]}
                texts2 = {normalize_text(item['text']): item['label'] for item in data_dicts[dataset2]}
                duplicates = set(texts1.keys()) & set(texts2.keys())
                inter_dataset_duplicates_count[dataset1] += len(duplicates)
                inter_dataset_duplicates_count[dataset2] += len(duplicates)
                for text in duplicates:
                    texts_to_remove[dataset1].add(text)
                    texts_to_remove[dataset2].add(text)
                    if texts1[text] != texts2[text]:
                        different_label_count[dataset1]['inter_dataset'] += 1
                        different_label_count[dataset2]['inter_dataset'] += 1
    
    # Check for duplicates between train and test data within each dataset
    for dataset in datasets:
        train_texts = {normalize_text(item['text']): item['label'] for item in data_dicts[dataset]}
        test_texts = {normalize_text(item['text']): item['label'] for item in test_data_dicts[dataset]}
        duplicates = set(train_texts.keys()) & set(test_texts.keys())
        train_test_duplicates_count[dataset] = len(duplicates)
        for text in duplicates:
            texts_to_remove[dataset].add(text)
            if train_texts[text] != test_texts[text]:
                different_label_count[dataset]['train_test'] += 1
    
    # Check for duplicates between each train data and all other test data
    for i, train_dataset in enumerate(datasets):
        train_texts = {normalize_text(item['text']): item['label'] for item in data_dicts[train_dataset]}
        for j, test_dataset in enumerate(datasets):
            if i != j:
                test_texts = {normalize_text(item['text']): item['label'] for item in test_data_dicts[test_dataset]}
                duplicates = set(train_texts.keys()) & set(test_texts.keys())
                train_other_test_duplicates_count[train_dataset] += len(duplicates)
                for text in duplicates:
                    texts_to_remove[train_dataset].add(text)
                    if train_texts[text] != test_texts[text]:
                        different_label_count[train_dataset]['train_other_test'] += 1

    # Filter and save data
    for dataset in datasets:
        filtered_data = []
        no_english_count = 0

        for item in data_dicts[dataset]:
            text = item['text']
            n_text = normalize_text(text)
            if n_text in texts_to_remove[dataset]:
                continue
            if not contains_english(text):
                no_english_count += 1
                continue
            filtered_data.append(item)

        to_save = f'{input_path}{dataset}/train_data_filtered.csv'
        save_dict_as_csv(filtered_data, to_save)

        print(f"\nRemove from {dataset}:")
        print(f"{inter_dataset_duplicates_count[dataset]} duplicates with other datasets' train data ({different_label_count[dataset]['inter_dataset']} with different labels)")
        print(f"{train_test_duplicates_count[dataset]} duplicates with test data ({different_label_count[dataset]['train_test']} with different labels)")
        print(f"{train_other_test_duplicates_count[dataset]} duplicates with other datasets' test data ({different_label_count[dataset]['train_other_test']} with different labels)")
        print(f"{no_english_count} texts with no English at all")
    
        
def filter_non_english_and_save_test_data(input_path):
    for dataset in datasets:
        test_path = f'{input_path}{dataset}/test_data.csv'
        to_save = f'{input_path}{dataset}/test_data_filtered.csv'
        
        # Copy the original file to the new file path
        shutil.copy(test_path, to_save)
        
        # Load the test CSV data as dictionaries
        test_data = load_csv_as_dict(to_save)
        
        filtered_data = []
        non_english_count = 0

        for item in test_data:
            text = item['text']
            if not contains_english(text):
                non_english_count += 1
                continue
            filtered_data.append(item)

        # Save the filtered data
        save_dict_as_csv(filtered_data, to_save)

        print(f"\nRemove from {dataset} test data:")
        print(f"{non_english_count} texts with no English at all")


def check_empty_cells(file_path, label_name):
    df = pd.read_csv(file_path, encoding='latin1')

    # Check if there are any empty cells (NaN or empty strings) in the specified column
    empty_cells = df[label_name].isnull() | df[label_name].eq('')
    
    # Return True if any empty cells are found, otherwise False
    return empty_cells.any()


def get_empty_first_rows(file_path, rows_num, label_name):
    df = pd.read_csv(file_path, encoding='latin1')

    # Find the first 5 row numbers where the 'rag_in_domain' column has empty values
    first_five_empty_rag_in_domain_rows = df[df[label_name].isna()].index.tolist()[:rows_num]

    # Print the row numbers
    return first_five_empty_rag_in_domain_rows

def print_empty_first_rows(file_path, rows_num, label_name):
    df = pd.read_csv(file_path, encoding='latin1')

    # Find the first 5 row numbers where the 'rag_in_domain' column has empty values
    first_five_empty_rag_in_domain_rows = df[df[label_name].isna()].index.tolist()[:rows_num]

    # Print the row numbers
    print(first_five_empty_rag_in_domain_rows)


def show_results_from_data_dict(data):
    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    df.set_index('Dataset').plot(kind='bar', ax=ax)

    # Customization
    ax.set_title('Performance Comparison Across Datasets', fontsize=16)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.legend(title='Model', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()


def get_weighted_f1_scores_with_datasets(column_names, file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, encoding='latin1')

    # Iterate over each column name provided
    for column_name in column_names:
        print(f"Weighted F1 Scores for {column_name} prompt:")
        
        # Initialize variables for weighted averages
        total_samples = 0
        weighted_positive_f1_sum = 0
        weighted_negative_f1_sum = 0
        weighted_macro_f1_sum = 0
        
        grouped_df = df.groupby('dataset')
        
        for dataset_name, group in grouped_df:
            dataset_size = len(group)
            positive_f1 = f1_score(group['label'], group[column_name],average='binary', pos_label=1)
            negative_f1 = f1_score(group['label'], group[column_name],average='binary', pos_label=0)
            macro_f1 = f1_score(group['label'], group[column_name], average='weighted')
            
            # Round the scores to 3 decimal places
            positive_f1 = round(positive_f1, 3)
            negative_f1 = round(negative_f1, 3)
            macro_f1 = round(macro_f1, 3)
            
            # Accumulate weighted sums
            weighted_positive_f1_sum += positive_f1 * dataset_size
            weighted_negative_f1_sum += negative_f1 * dataset_size
            weighted_macro_f1_sum += macro_f1 * dataset_size
            total_samples += dataset_size
            
            # Print the results for this dataset
            print(f'Dataset: {dataset_name}: size {dataset_size}, neg {negative_f1}, pos {positive_f1}, macro {macro_f1}')

        
        # Calculate and print the weighted average F1 scores across all datasets
        avg_weighted_positive_f1 = round(weighted_positive_f1_sum / total_samples, 3)
        avg_weighted_negative_f1 = round(weighted_negative_f1_sum / total_samples, 3)
        avg_weighted_macro_f1 = round(weighted_macro_f1_sum / total_samples, 3)
        
        print("\n=====================\n")        
        print(f"Weighted Average F1 for {column_name}: neg {avg_weighted_negative_f1} pos {avg_weighted_positive_f1} macro {avg_weighted_macro_f1}")



def show_results_from_csv(column_names, file_path):
    tables = []
    for column_name in column_names:
        print(f"Results for {column_name} prompt:")
        analysis_table = create_analysis_table_by_datasets(file_path, column_name)
        tables.append(analysis_table)
        print(tabulate(analysis_table, headers='keys', tablefmt='pretty', showindex=False))
        print("\n+++++++++++++++++++++\n")
    plot_metrics(tables,column_names, "dataset")


def create_analysis_table_by_prompt(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path, encoding='latin1')
    
    def calculate_metrics(df, col_name):
        accuracy = accuracy_score(df['label'], df[col_name])
        precision = precision_score(df['label'], df[col_name])
        recall = recall_score(df['label'], df[col_name])
        f1 = f1_score(df['label'], df[col_name])
        # Round the metrics to 3 decimal places
        accuracy = round(accuracy, 3)
        precision = round(precision, 3)
        recall = round(recall, 3)
        f1 = round(f1, 3)
        return accuracy, precision, recall, f1

    # Define the columns to analyze
    columns_to_analyze = ["after_explain", "basic", "instructions"]

    # Initialize a list to store the results
    results_list = []

    # Calculate metrics for each specified column
    for col in columns_to_analyze:
        accuracy, precision, recall, f1 = calculate_metrics(df, col)
        results_list.append({'prompt': col, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1})

    # Convert the list to a dataframe
    results_table = pd.DataFrame(results_list)
    results_table.set_index('prompt', inplace=True)
    
    # Reset index to include 'column' as a column
    results_table.reset_index(inplace=True)

    return results_table


def create_analysis_table_by_datasets(csv_file, column_name):
    # Load the dataset
    data = pd.read_csv(csv_file, encoding='latin1')
    
    # Extract the relevant columns
    results = data[['dataset', 'label', column_name]]

    # Grouping the data by dataset and analyzing the results
    grouped_results = results.groupby('dataset').agg(
        total_samples=('label', 'count'),
        true_positives=('label', lambda x: ((x == 1) & (results[column_name] == 1)).sum()),
        true_negatives=('label', lambda x: ((x == 0) & (results[column_name] == 0)).sum()),
        false_positives=('label', lambda x: ((x == 0) & (results[column_name] == 1)).sum()),
        false_negatives=('label', lambda x: ((x == 1) & (results[column_name] == 0)).sum())
    )

    # Calculating precision, recall, F1 score, and accuracy
    grouped_results['precision'] = grouped_results['true_positives'] / (grouped_results['true_positives'] + grouped_results['false_positives'])
    grouped_results['recall'] = grouped_results['true_positives'] / (grouped_results['true_positives'] + grouped_results['false_negatives'])
    grouped_results['f1_score'] = 2 * (grouped_results['precision'] * grouped_results['recall']) / (grouped_results['precision'] + grouped_results['recall'])
    grouped_results['accuracy'] = (grouped_results['true_positives'] + grouped_results['true_negatives']) / grouped_results['total_samples']

    # Handling division by zero in precision, recall, and F1 score calculations
    grouped_results['precision'] = grouped_results['precision'].fillna(0)
    grouped_results['recall'] = grouped_results['recall'].fillna(0)
    grouped_results['f1_score'] = grouped_results['f1_score'].fillna(0)
    
    # Round the metrics to 3 decimal places
    grouped_results['precision'] = round(grouped_results['precision'], 3)
    grouped_results['recall'] = round(grouped_results['recall'], 3)
    grouped_results['f1_score'] = round(grouped_results['f1_score'], 3)
    grouped_results['accuracy'] = round(grouped_results['accuracy'], 3)

    # Reset the index for easier display
    grouped_results.reset_index(inplace=True)

    return grouped_results


def plot_metrics(grouped_results_list, labels, x_name):
    # Define the metrics to plot
    metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define some colors for different grouped results

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot each metric
    for ax, metric in zip(axes.flatten(), metrics):
        width = 0.2  # Width of each bar
        x = np.arange(len(grouped_results_list[0][x_name]))  # The label locations
        for i, grouped_results in enumerate(grouped_results_list):
            offset = width * i  # Offset for each dataset
            ax.bar(x + offset, grouped_results[metric], width, label=labels[i], color=colors[i % len(colors)], alpha=0.7)
        ax.set_title(metric.capitalize())
        ax.set_xticks(x + width * (len(grouped_results_list) - 1) / 2)
        ax.set_xticklabels(grouped_results_list[0][x_name], rotation=45, ha='right')
        ax.set_xlabel(x_name)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)  # Assuming metric values range between 0 and 1
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
# Show results of csv file by differents columns
def show_csv_results_for_tp(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path,  encoding='latin1')
    
    # Define performance metric calculation
    def calculate_metrics(df, col_name):
        accuracy = accuracy_score(df['label'], df[col_name])
        precision = precision_score(df['label'], df[col_name])
        recall = recall_score(df['label'], df[col_name])
        f1 = f1_score(df['label'], df[col_name])
        return accuracy, precision, recall, f1

    # Get columns with t and p parameters
    params_columns = [col for col in df.columns if col.startswith('t=')]

    # Initialize a list to store the results
    results_list = []

    # Calculate metrics for each combination of t and p
    for col in params_columns:
        t, p = col.split(', ')
        t = t.split('=')[1]
        p = p.split('=')[1]
        accuracy, precision, recall, f1 = calculate_metrics(df, col)
        results_list.append({'t': t, 'p': p, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

    # Convert the list to a dataframe
    results = pd.DataFrame(results_list)

    # Convert columns to appropriate data types
    results['t'] = results['t'].astype(float)
    results['p'] = results['p'].astype(float)

    # Plotting the results
    plt.figure(figsize=(15, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    sns.heatmap(results.pivot('t', 'p', 'accuracy'), annot=True, fmt=".2f")
    plt.title('Accuracy')

    # Plot precision
    plt.subplot(2, 2, 2)
    sns.heatmap(results.pivot('t', 'p', 'precision'), annot=True, fmt=".2f")
    plt.title('Precision')

    # Plot recall
    plt.subplot(2, 2, 3)
    sns.heatmap(results.pivot('t', 'p', 'recall'), annot=True, fmt=".2f")
    plt.title('Recall')

    # Plot F1 score
    plt.subplot(2, 2, 4)
    sns.heatmap(results.pivot('t', 'p', 'f1'), annot=True, fmt=".2f")
    plt.title('F1 Score')   

    plt.tight_layout()
    plt.show()


def text_to_csv(input_file_path, output_file_path):
    # Read the text file and parse its content with utf-8 encoding
    parsed_data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Each line is expected to be a list in string format
            parsed_data.append(eval(line.strip()))

    # Write the parsed data to a CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Writing header
        csv_writer.writerow(['ID', 'text', 'label'])
        # Writing the data rows
        csv_writer.writerows(parsed_data)

    print("CSV file has been created successfully.")


def json_to_csv(input_file_path, output_file_path):
    # Read the JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare the CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['ID', 'Utterance', 'Speaker', 'Context', 'Context Speakers', 'Show', 'Sarcasm'])

        # Write the data rows
        for key, value in data.items():
            csv_writer.writerow([
                key,
                value['utterance'],
                value['speaker'],
                '; '.join(value['context']),
                '; '.join(value['context_speakers']),
                value['show'],
                value['sarcasm']
            ])

    print("CSV file has been created successfully.")
  
    
def multimodal_sarcasm_detection():
    input_multimodal_sarcasm_detection = 'C:/Users/97254/VScode/data/multimodal_sarcasm_detection/raw/train.txt'
    output_multimodal_sarcasm_detection = 'C:/Users/97254/VScode/data/multimodal_sarcasm_detection/data.csv'
    text_to_csv(input_multimodal_sarcasm_detection, output_multimodal_sarcasm_detection)
    # Reorder the columns
    df = pd.read_csv(output_multimodal_sarcasm_detection)
    reordered_df = df[['label', 'text', 'ID'] + [col for col in df.columns if col not in ['label', 'text', 'ID']]]
    # Save the reordered dataframe to a new CSV file
    reordered_df.to_csv(output_multimodal_sarcasm_detection, index=False)
    print("data.csv created for multimodal_sarcasm_detection")


def MUStARD():
    input_MUStARD = 'C:/Users/97254/VScode/data/MUStARD/raw/sarcasm_data.json'
    outputMUStARD = 'C:/Users/97254/VScode/data/MUStARD/data.csv'
    json_to_csv(input_MUStARD, outputMUStARD)
    # Load the CSV file
    df = pd.read_csv(outputMUStARD)

    # Rename the columns
    df = df.rename(columns={'Sarcasm': 'label', 'Utterance': 'text'})

    # Convert the 'label' column to 1 for True and 0 for False
    df['label'] = df['label'].apply(lambda x: 1 if x else 0)

    # Reorder the columns
    reordered_df = df[['label', 'text', 'ID'] + [col for col in df.columns if col not in ['label', 'text', 'ID']]]

    # Save the reordered dataframe to a new CSV file
    reordered_file_path = 'path_to_save/reordered_your_csv_file.csv'  # Change this to where you want to save the new file
    reordered_df.to_csv(outputMUStARD, index=False)
    print("data.csv created for MUStARD")


def create_random_mixed_data(data_files_path, samples_number, filtered=False):    
    combined_data = []
    label_0_data = []
    label_1_data = []
    
    min_samples_per_dataset = float('inf')
    
    for data_name in datasets:
        if filtered:
            file_path = f'{data_files_path}/{data_name}/test_data_filtered.csv'
        else:
            file_path = f'{data_files_path}/{data_name}/test_data.csv'
            
        
        try:
            with open(file_path, encoding='latin1') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check necessary columns
                if 'label' not in reader.fieldnames or 'text' not in reader.fieldnames:
                    print(f"File {file_path} does not have the required columns.")
                    continue
                
                # Add 'context' column if missing
                context_present = 'context' in reader.fieldnames
                
                data = [row for row in reader]
                
                # Separate data by label and add dataset name
                for row in data:
                    row['dataset'] = data_name
                    if not context_present:
                        row['context'] = ""
                    
                    if row['label'] == '0':
                        label_0_data.append(row)
                    elif row['label'] == '1':
                        label_1_data.append(row)
                
                label_0_dataset = [row for row in data if row['label'] == '0']
                label_1_dataset = [row for row in data if row['label'] == '1']
                
                min_samples_per_dataset = min(min_samples_per_dataset, len(label_0_dataset), len(label_1_dataset))
        
        except UnicodeDecodeError as e:
            print(f"Could not read {file_path} due to encoding error: {e}")
        except Exception as e:
            print(f"Could not process {file_path} due to error: {e}")
    
    # Determine the number of samples to use per dataset
    num_samples_per_label = samples_number // 2
    
    if num_samples_per_label > min_samples_per_dataset:
        print(f"Warning: Requested samples per label ({num_samples_per_label}) exceeds the smallest dataset size ({min_samples_per_dataset}). Adjusting to {min_samples_per_dataset} samples per label.")
        num_samples_per_label = min_samples_per_dataset
    
    sampled_label_0_data = []
    sampled_label_1_data = []
    
    # Sample the specified number of rows for each label from each dataset
    for data_name in datasets:
        label_0_data_for_dataset = [row for row in label_0_data if row['dataset'] == data_name]
        label_1_data_for_dataset = [row for row in label_1_data if row['dataset'] == data_name]
        
        if len(label_0_data_for_dataset) < num_samples_per_label or len(label_1_data_for_dataset) < num_samples_per_label:
            print(f"Warning: Not enough data in dataset {data_name} to sample {num_samples_per_label} rows per label.")
            continue
        
        sampled_label_0_data.extend(random.sample(label_0_data_for_dataset, num_samples_per_label))
        sampled_label_1_data.extend(random.sample(label_1_data_for_dataset, num_samples_per_label))
    
    combined_data = sampled_label_0_data + sampled_label_1_data
    
    # Shuffle the combined data
    random.shuffle(combined_data)
    
    # Define fieldnames with 'label', 'text', 'context', and 'dataset'
    fieldnames = ['label', 'text', 'context', 'dataset']
    
    # Write combined data to the output CSV file
    if filtered:
        output_path = f'{data_files_path}/random/random_{num_samples_per_label}_filtered.csv'
    else:
        output_path = f'{data_files_path}/random/random_{num_samples_per_label}.csv'
        
    if combined_data:
        with open(output_path, 'w', newline='', encoding='latin1') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in combined_data:
                filtered_row = {key: row.get(key, '') for key in fieldnames}
                writer.writerow(filtered_row)
        
        print(f"Data successfully written to {output_path}")
    else:
        print("No data to write to the output CSV.")


def create_random_mixed_data_with_context(data_files_path, samples_number):    
    combined_data = []
    label_0_data = []
    label_1_data = []
    
    min_samples_per_dataset = float('inf')
    
    for data_name in context_datasets:
        file_path = f'{data_files_path}/{data_name}/test_data.csv'
        
        try:
            with open(file_path, encoding='latin1') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check necessary columns
                if 'label' not in reader.fieldnames or 'text' not in reader.fieldnames:
                    print(f"File {file_path} does not have the required columns.")
                    continue
                
                # Add 'context' column if missing
                context_present = 'context' in reader.fieldnames
                
                data = [row for row in reader]
                
                # Separate data by label and add dataset name
                for row in data:
                    row['dataset'] = data_name
                    if not context_present:
                        row['context'] = ""
                    
                    if row['label'] == '0' and row['context']:
                        label_0_data.append(row)
                    elif row['label'] == '1' and row['context']:
                        label_1_data.append(row)
                
                label_0_dataset = [row for row in data if row['label'] == '0' and row['context']]
                label_1_dataset = [row for row in data if row['label'] == '1' and row['context']]
                                
                min_samples_per_dataset = min(min_samples_per_dataset, len(label_0_dataset), len(label_1_dataset))
        
        except UnicodeDecodeError as e:
            print(f"Could not read {file_path} due to encoding error: {e}")
        except Exception as e:
            print(f"Could not process {file_path} due to error: {e}")
    
    # Determine the number of samples to use per dataset
    num_samples_per_label = samples_number // 2
    
    if num_samples_per_label > min_samples_per_dataset:
        print(f"Warning: Requested samples per label ({num_samples_per_label}) exceeds the smallest dataset size ({min_samples_per_dataset}). Adjusting to {min_samples_per_dataset} samples per label.")
        num_samples_per_label = min_samples_per_dataset
    
    sampled_label_0_data = []
    sampled_label_1_data = []
    
    # Sample the specified number of rows for each label from each dataset
    for data_name in context_datasets:
        label_0_data_for_dataset = [row for row in label_0_data if row['dataset'] == data_name]
        label_1_data_for_dataset = [row for row in label_1_data if row['dataset'] == data_name]
        
        if len(label_0_data_for_dataset) < num_samples_per_label or len(label_1_data_for_dataset) < num_samples_per_label:
            print(f"Warning: Not enough data in dataset {data_name} to sample {num_samples_per_label} rows per label.")
            continue
        
        sampled_label_0_data.extend(random.sample(label_0_data_for_dataset, num_samples_per_label))
        sampled_label_1_data.extend(random.sample(label_1_data_for_dataset, num_samples_per_label))
    
    combined_data = sampled_label_0_data + sampled_label_1_data
    
    # Shuffle the combined data
    random.shuffle(combined_data)
    
    # Define fieldnames with 'label', 'text', 'context', and 'dataset'
    fieldnames = ['label', 'text', 'context', 'dataset']
    
    # Write combined data to the output CSV file
    output_path = f'{data_files_path}/compare/random/random_{num_samples_per_label}_dataset.csv'
    if combined_data:
        with open(output_path, 'w', newline='', encoding='latin1') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in combined_data:
                filtered_row = {key: row.get(key, '') for key in fieldnames}
                writer.writerow(filtered_row)
        
        print(f"Data successfully written to {output_path}")
    else:
        print("No data to write to the output CSV.")





