import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time

# Create directories if they don't exist
export_directory = "Export"
plot_directory = "Plot"
histogram_directory = "Filtered_Config_hist"
if not os.path.exists(export_directory):
    os.makedirs(export_directory)
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
if not os.path.exists(histogram_directory):
    os.makedirs(histogram_directory)

# Function to count matches between two lists
def match_count(predicted, actual):
    matches = sum(1 for p, a in zip(predicted, actual) if p == a)
    return matches

# Function to transform the data into the required JSON-like format
def transform_to_json_like_format(data):
    transformed_data = []
    for row in data:
        json_like_entry = {
            "first_layer": int(row[0]),  # Convert layer values to integer
            "second_layer": int(row[1]),  # Convert layer values to integer
            "dense_layer": int(row[2]),  # Convert layer values to integer
            "learning_rate": float(row[3])  # Keep learning_rate as a float
        }
        transformed_data.append(json_like_entry)
    return transformed_data

# Function to process number matches and categorize them based on match count
def process_number_matches(df):
    filtered_7_7 = []
    filtered_6_7 = []
    filtered_5_7 = []
    filtered_4_7 = []

    for index, row in df.iterrows():
        if row['learning_rate'] == 0.0:
            continue

        matches = match_count(row['predicted_series'], row['actual_series'])
        if matches == 7:
            filtered_7_7.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])
        elif matches == 6:
            filtered_6_7.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])
        elif matches == 5:
            filtered_5_7.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])
        elif matches == 4:
            filtered_4_7.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])

    return filtered_7_7, filtered_6_7, filtered_5_7, filtered_4_7

# Function to process trend matches and categorize them based on match count
def process_trend_matches(df):
    filtered_7_7_trend = []
    filtered_6_7_trend = []

    for index, row in df.iterrows():
        if row['learning_rate'] == 0.0:
            continue

        matches = match_count(row['prediction_trend'], row['actual_trend'])
        if matches == 7:
            filtered_7_7_trend.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])
        elif matches == 6:
            filtered_6_7_trend.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])

    return filtered_7_7_trend, filtered_6_7_trend

# Function to get user input for R² filter
def user_input():
    while True:
        choice = input("Do you want to export [0] All configurations or [1] R2-score positive configurations? Enter 0 or 1: ")
        if choice in ['0', '1']:
            return int(choice)
        else:
            print("Invalid input. Please enter 0 or 1.")

# Function to export configurations filtered by a specific number
def export_filtered_by_number(df, number, filter_r2_positive=False):
    filtered_data = []
    for index, row in df.iterrows():
        if row['learning_rate'] == 0.0 or (filter_r2_positive and row['r2'] <= 0):
            continue
        if row['predicted_series'][number - 1] == row['actual_series'][number - 1]:
            filtered_data.append([int(row['first_layer']), int(row['second_layer']), int(row['dense_layer']), float(row['learning_rate'])])

    if filtered_data:
        transformed_data = transform_to_json_like_format(filtered_data)
        timestamp = time.strftime("%y%m%d%H%M%S")
        file_name = os.path.join(export_directory, f"Export_{number}_{timestamp}.json")
        with open(file_name, 'w') as f:
            json.dump(transformed_data, f, indent=4)
        print(f"\nData for predicted number {number} exported to {file_name}")
    else:
        print(f"\nNo matching data for predicted number {number} found. Export file not created.")

# Function to export filtered data in JSON-like format
def export_to_json_like(filtered_data, file_name_prefix="Export"):
    all_data = []
    for data in filtered_data:
        if data:
            all_data.extend(data)
    
    if all_data:
        transformed_data = transform_to_json_like_format(all_data)
        timestamp = time.strftime("%y%m%d%H%M%S")
        file_name = os.path.join(export_directory, f"{file_name_prefix}_{timestamp}.json")
        with open(file_name, 'w') as f:
            json.dump(transformed_data, f, indent=4)
        print(f"\nData exported to {file_name}")
    else:
        print("\nNo data match filters - no data to export... Export files not created.")

# Function to create histograms from exported LSTM configurations
def create_histograms_from_exports():
    for number in range(1, 8):  # Loop through 1 to 7 for Export_1.json to Export_7.json
        export_files = [f for f in os.listdir(export_directory) if f.startswith(f"Export_{number}_") and f.endswith(".json")]

        all_configs = []
        for file_name in export_files:
            file_path = os.path.join(export_directory, file_name)
            with open(file_path, 'r') as f:
                configs = json.load(f)
                all_configs.extend(configs)
        
        # Count unique LSTM configurations (without learning rate)
        config_counts = {}
        for config in all_configs:
            config_key = (config['first_layer'], config['second_layer'], config['dense_layer'])
            if config_key in config_counts:
                config_counts[config_key] += 1
            else:
                config_counts[config_key] = 1

        # Generate histogram data
        config_labels = [f"{k[0]}-{k[1]}-{k[2]}" for k in config_counts.keys()]
        config_values = list(config_counts.values())

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.bar(config_labels, config_values, color='skyblue')
        plt.xlabel('LSTM Configuration (First Layer - Second Layer - Dense Layer)')
        plt.ylabel('Frequency')
        plt.title(f"Histogram of LSTM Configurations for Export_{number}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the histogram
        histogram_file = os.path.join(histogram_directory, f"Histogram_Export_{number}.png")
        plt.savefig(histogram_file)
        plt.close()
        print(f"Histogram saved: {histogram_file}")

# Function to load JSON data from files (for plotting)
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Function to parse xResults_bckp_All.txt and return the required metrics
def parse_xresults(file_path):
    results = []
    with open(file_path, 'r') as file:
        current_record = {}
        for line in file:
            line = line.strip()
            if line.startswith("# LSTM Input:"):
                if current_record:
                    if current_record.get('learning_rate', 0) != 0:
                        results.append(current_record)
                current_record = {}
                parts = line.split('(')[1].replace(')', '').split(',')
                current_record['first_layer'] = int(float(parts[0].strip()))
                current_record['second_layer'] = int(float(parts[1].strip()))
                current_record['dense_layer'] = int(float(parts[2].strip()))
                current_record['learning_rate'] = float(parts[3].strip())
            elif line.startswith("Huber Loss:"):
                current_record['huber'] = float(line.split(":")[1].strip())
            elif line.startswith("R-squared:"):
                current_record['r2'] = float(line.split(":")[1].strip())
            elif line.startswith("SMAPE:"):
                current_record['smape'] = float(line.split(":")[1].strip())
            elif line.startswith("Mean Absolute Error (MAE):"):
                current_record['mae'] = float(line.split(":")[1].strip())
        if current_record and current_record.get('learning_rate', 0) != 0:
            results.append(current_record)
    return pd.DataFrame(results)

# Function to find and match configurations from export files with xResults_bckp_All.txt
def match_and_extract_metrics(export_file, xresults_df, metric):
    export_data = load_json_data(os.path.join(export_directory, export_file))
    merged_data = pd.merge(export_data, xresults_df, on=['first_layer', 'second_layer', 'dense_layer', 'learning_rate'])
    return merged_data[['learning_rate', metric]]

# Function to plot metrics vs learning_rate
def plot_metric_vs_learning_rate(metric_name, metric_column, export_file_pattern, ylabel, color):
    # Find the latest export file for the metric
    files = [f for f in os.listdir(export_directory) if f.startswith(export_file_pattern) and f.endswith(".json")]
    if not files:
        print(f"No {metric_name} export files found.")
        return
    latest_file = sorted(files)[-1]
    
    # Parse xResults_bckp_All.txt to get the metrics
    xresults_df = parse_xresults('xResults_bckp_All.txt')
    
    # Match configurations and extract the metric
    metric_data = match_and_extract_metrics(latest_file, xresults_df, metric_column)
    
    if metric_data.empty:
        print(f"No matching data found for {metric_name}.")
        return
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(metric_data['learning_rate'], metric_data[metric_column], color=color, label=f'{metric_name}', alpha=0.7)
    plt.xscale('log')  # Logarithmic scale for learning rate
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name} vs Learning Rate')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Save the plot
    plot_file = os.path.join(plot_directory, f"{metric_name}_vs_LearningRate.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"{metric_name} vs Learning Rate plot saved: {plot_file}")

# Function to plot multiple metrics in one plot using dual y-axis (Huber Loss and R²)
def plot_huber_and_r2_vs_learning_rate():
    # Load the latest Huber and R² export files
    huber_files = [f for f in os.listdir(export_directory) if f.startswith("Export_Huber") and f.endswith(".json")]
    r2_files = [f for f in os.listdir(export_directory) if f.startswith("Export_R2") and f.endswith(".json")]
    
    if not huber_files or not r2_files:
        print("No Export_Huber or Export_R2 files found.")
        return
    
    latest_huber_file = sorted(huber_files)[-1]
    latest_r2_file = sorted(r2_files)[-1]
    
    xresults_df = parse_xresults('xResults_bckp_All.txt')
    
    huber_data = match_and_extract_metrics(latest_huber_file, xresults_df, 'huber')
    r2_data = match_and_extract_metrics(latest_r2_file, xresults_df, 'r2')
    
    if huber_data.empty or r2_data.empty:
        print("No matching data found for Huber or R².")
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Huber Loss', color='b')
    ax1.scatter(huber_data['learning_rate'], huber_data['huber'], color='b', label='Huber Loss')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('R² Score', color='g')
    ax2.scatter(r2_data['learning_rate'], r2_data['r2'], color='g', label='R² Score')
    ax2.tick_params(axis='y', labelcolor='g')
    
    fig.tight_layout()
    plt.title("Huber Loss and R² Score vs Learning Rate")
    plt.grid(True, which="both", ls="--")
    
    plot_file = os.path.join(plot_directory, "Huber_and_R2_vs_LearningRate.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Huber Loss and R² Score vs Learning Rate plot saved: {plot_file}")

# Function to plot combined MAE and Huber loss vs learning_rate
def plot_mae_and_huber_vs_learning_rate():
    # Parse xResults_bckp_All.txt to get all configurations
    xresults_df = parse_xresults('xResults_bckp_All.txt')
    
    if xresults_df.empty:
        print("No data found in xResults_bckp_All.txt.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(xresults_df['learning_rate'], xresults_df['mae'], color='r', label='MAE', alpha=0.7)
    plt.scatter(xresults_df['learning_rate'], xresults_df['huber'], color='b', label='Huber Loss', alpha=0.7)
    plt.xscale('log')  # Logarithmic scale for learning rate
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Error Value')
    plt.title('MAE and Huber Loss vs Learning Rate')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plot_file = os.path.join(plot_directory, "MAE_and_Huber_vs_LearningRate.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"MAE and Huber Loss vs Learning Rate plot saved: {plot_file}")

# Plot Huber vs learning_rate
def plot_huber_vs_learning_rate():
    plot_metric_vs_learning_rate('Huber Loss', 'huber', 'Export_Huber', 'Huber Loss', 'b')

# Plot R² vs learning_rate
def plot_r2_vs_learning_rate():
    plot_metric_vs_learning_rate('R² Score', 'r2', 'Export_R2', 'R² Score', 'g')

# Plot SMAPE vs learning_rate
def plot_smape_vs_learning_rate():
    plot_metric_vs_learning_rate('SMAPE', 'smape', 'Export_SMAPE', 'SMAPE', 'r')

# Function to export top configurations based on minimal Huber loss
def export_huber_configs(df):
    timestamp = time.strftime("%y%m%d%H%M%S")
    top_huber = df.nsmallest(10, 'huber')[['first_layer', 'second_layer', 'dense_layer', 'learning_rate']].values.tolist()

    # Ensure proper formatting: layers as integers and learning rate as float
    formatted_huber = [[int(config[0]), int(config[1]), int(config[2]), float(config[3])] for config in top_huber]
    
    print("\nTop 10 LSTM configurations for minimal Huber:")
    print(formatted_huber)

    file_name = os.path.join(export_directory, f"Export_Huber_{timestamp}.json")
    with open(file_name, 'w') as f:
        json_data = transform_to_json_like_format(formatted_huber)
        json.dump(json_data, f, indent=4)
    print(f"\nTop 10 Huber configurations exported to {file_name}")

# Function to export top configurations based on R²
def export_r2_configs(df):
    timestamp = time.strftime("%y%m%d%H%M%S")
    top_r2 = df.nlargest(10, 'r2')[['first_layer', 'second_layer', 'dense_layer', 'learning_rate']].values.tolist()

    # Ensure proper formatting: layers as integers and learning rate as float
    formatted_r2 = [[int(config[0]), int(config[1]), int(config[2]), float(config[3])] for config in top_r2]
    
    print("\nTop 10 LSTM configurations for highest R²:")
    print(formatted_r2)

    file_name = os.path.join(export_directory, f"Export_R2_{timestamp}.json")
    with open(file_name, 'w') as f:
        json_data = transform_to_json_like_format(formatted_r2)
        json.dump(json_data, f, indent=4)
    print(f"\nTop 10 R² configurations exported to {file_name}")

# Function to export top configurations based on SMAPE
def export_smape_configs(df):
    timestamp = time.strftime("%y%m%d%H%M%S")
    top_smape = df.nsmallest(10, 'smape')[['first_layer', 'second_layer', 'dense_layer', 'learning_rate']].values.tolist()

    # Ensure proper formatting: layers as integers and learning rate as float
    formatted_smape = [[int(config[0]), int(config[1]), int(config[2]), float(config[3])] for config in top_smape]

    print("\nTop 10 LSTM configurations for lowest SMAPE:")
    print(formatted_smape)

    file_name = os.path.join(export_directory, f"Export_SMAPE_{timestamp}.json")
    with open(file_name, 'w') as f:
        json_data = transform_to_json_like_format(formatted_smape)
        json.dump(json_data, f, indent=4)
    print(f"\nTop 10 SMAPE configurations exported to {file_name}")

# Function to load and parse the results file
def parse_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        current_record = {}
        for line in file:
            line = line.strip()
            if line.startswith("# LSTM Input:"):
                if current_record:
                    if current_record.get('learning_rate', 0) != 0:
                        results.append(current_record)
                current_record = {'predicted_series': [], 'actual_series': [], 'prediction_trend': [], 'actual_trend': []}
                parts = line.split('(')[1].replace(')', '').split(',')
                current_record['first_layer'] = float(parts[0].strip())
                current_record['second_layer'] = float(parts[1].strip())
                current_record['dense_layer'] = float(parts[2].strip())
                current_record['learning_rate'] = float(parts[3].strip())
            elif line.startswith("| Prediction |"):
                current_record['predicted_series'] = [float(x.strip()) for x in line.split('|')[2:-1]]
            elif line.startswith("| Actual Series |"):
                current_record['actual_series'] = [float(x.strip()) for x in line.split('|')[2:-1]]
            elif line.startswith("| Prediction Trend |"):
                current_record['prediction_trend'] = [x.strip() for x in line.split('|')[2:-1]]
            elif line.startswith("| Actual Trend |"):
                current_record['actual_trend'] = [x.strip() for x in line.split('|')[2:-1]]
            elif line.startswith("Mean Squared Error (MSE):"):
                current_record['mean_squared_error'] = float(line.split(":")[1].strip())
            elif line.startswith("Mean Absolute Error (MAE):"):
                current_record['mean_absolute_error'] = float(line.split(":")[1].strip())
            elif line.startswith("Huber Loss:"):
                current_record['huber'] = float(line.split(":")[1].strip())
            elif line.startswith("R-squared:"):
                current_record['r2'] = float(line.split(":")[1].strip())
            elif line.startswith("SMAPE:"):
                current_record['smape'] = float(line.split(":")[1].strip())
            elif line.startswith("Training Time:"):
                current_record['training_time'] = float(line.split(":")[1].strip().replace(' seconds', ''))
        if current_record and current_record.get('learning_rate', 0) != 0:
            results.append(current_record)
    return pd.DataFrame(results)

# Main function to process and display results
def main(file_path):
    df = parse_results(file_path)
    r2_filter = user_input()

    filtered_7_7, filtered_6_7, filtered_5_7, filtered_4_7 = process_number_matches(df)
    filtered_7_7_trend, filtered_6_7_trend = process_trend_matches(df)

    print("\nFiltered 7/7 number matches:")
    if filtered_7_7:
        print(pd.DataFrame(filtered_7_7, columns=["First Layer", "Second Layer", "Dense Layer", "Learning Rate"]))
    else:
        print("No records found")

    print("\nFiltered 6/7 number matches:")
    if filtered_6_7:
        print(pd.DataFrame(filtered_6_7, columns=["First Layer", "Second Layer", "Dense Layer", "Learning Rate"]))
    else:
        print("No records found")

    print("\nFiltered 5/7 number matches:")
    if filtered_5_7:
        print(pd.DataFrame(filtered_5_7, columns=["First Layer", "Second Layer", "Dense Layer", "Learning Rate"]))
    else:
        print("No records found")

    print("\nFiltered 4/7 number matches:")
    if filtered_4_7:
        print(pd.DataFrame(filtered_4_7, columns=["First Layer", "Second Layer", "Dense Layer", "Learning Rate"]))
    else:
        print("No records found")

    print("\nFiltered 7/7 trend matches:")
    if filtered_7_7_trend:
        print(pd.DataFrame(filtered_7_7_trend, columns=["First Layer", "Second Layer", "Dense Layer", "Learning Rate"]))
    else:
        print("No records found")

    print("\nFiltered 6/7 trend matches:")
    if filtered_6_7_trend:
        print(pd.DataFrame(filtered_6_7_trend, columns=["First Layer", "Second Layer", "Dense Layer", "Learning Rate"]))
    else:
        print("No records found")

    export_to_json_like([filtered_7_7, filtered_6_7, filtered_5_7, filtered_4_7], file_name_prefix="Export_Num")
    export_to_json_like([filtered_7_7_trend, filtered_6_7_trend], file_name_prefix="Export_Trends")

    export_huber_configs(df)
    export_r2_configs(df)
    export_smape_configs(df)

    for i in range(1, 8):
        export_filtered_by_number(df, i, filter_r2_positive=bool(r2_filter))

    # Create histograms for filtered LSTM configurations
    create_histograms_from_exports()

    # Generate plots
    plot_huber_vs_learning_rate()
    plot_r2_vs_learning_rate()
    plot_smape_vs_learning_rate()

    # New combined plot for Huber and R²
    plot_huber_and_r2_vs_learning_rate()

    # New combined plot for MAE and Huber Loss
    plot_mae_and_huber_vs_learning_rate()

# Entry point
if __name__ == "__main__":
    file_path = 'xResults_bckp_All.txt'
    main(file_path)
