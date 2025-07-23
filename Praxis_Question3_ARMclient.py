"""
Copyright Raymond Soto Jr. D.Eng(c).
From Edge to Enterprise:
Federated Learning Threat Classification with Heterogeneous Devices in Converged Energy Sector Networks
Revised July 9th, 2025
"""
import requests
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report, f1_score # Added f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import datetime
import sys

# Configuration for the ARM client (Raspberry Pi 5)
CLIENT_ID = "ARM"
SERVER_URL = "http://192.168.1.198:5000"
TRAIN_CSV = "data/TRAIN.csv" # Training CSV for ARM
TEST_CSV = "data/TEST.csv"    # Test CSV for ARM
LOCAL_EPOCHS = 8
BATCH_SIZE = 256
FED_ROUNDS = 100 # Federated rounds control the loop
CLASS_WEIGHTS = {0: 1.0, 1: 5.0} # To adjust focus on minority class

# --- Prepare matplotlib for interactive plotting (1x2) AND saving ---
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
loss_history = []
acc_history = []
test_acc_history = []

def create_model():
    model = models.Sequential([
        layers.Input(shape=(47,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(csv_file):
    try:
        if not os.path.exists(csv_file): raise FileNotFoundError(f"Data file not found at {csv_file}")
        if os.path.isdir(csv_file): raise ValueError(f"CSV path '{csv_file}' is a directory.")
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns: raise ValueError(f"'label' column not found in {csv_file}")
        X = df.drop(columns=['label']).values.astype('float32')
        y = df['label'].astype('int').values
        expected_features = 47
        if X.shape[1] != expected_features: raise ValueError(f"Data Error: Expected {expected_features} features, found {X.shape[1]} in {csv_file}.")
        return X, y
    except Exception as e: print(f"Error loading data from {csv_file}: {e}"); sys.exit(1)

# --- UPDATED plot_metrics function for overlay plot ---
def plot_metrics(round_num, loss, train_acc, test_acc):
    loss_history.append(loss)
    acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    # Plot Training Loss (ax[0])
    ax[0].clear()
    ax[0].plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-')
    ax[0].set_title("Local Training Loss per Round")
    ax[0].set_xlabel("Federated Round")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    # Plot Combined Training & Testing Accuracy (ax[1])
    ax[1].clear()
    ax[1].plot(range(1, len(acc_history) + 1), acc_history, marker='o', linestyle='-', label='Train Accuracy')
    ax[1].plot(range(1, len(test_acc_history) + 1), test_acc_history, marker='x', linestyle='--', color='g', label='Test Accuracy')
    ax[1].set_title("Training & Testing Accuracy") # Shortened title
    ax[1].set_xlabel("Federated Round")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True)
    ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax[1].legend()

    fig.suptitle(f"Client {CLIENT_ID} - Metrics up to Round {round_num}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.1)

# --- (get_global_weights function) ---
def get_global_weights(expected_round=None):
    max_wait_time = 300
    start_time = time.time()
    poll_interval = 5
    while True:
        if time.time() - start_time > max_wait_time:
             raise TimeoutError(f"Server did not reach round {expected_round} within {max_wait_time}s.")
        try:
            response = requests.get(f"{SERVER_URL}/global_model", timeout=15)
            response.raise_for_status()
            data = response.json()
            if 'global_weights' in data and 'round' in data:
                current_server_round = data['round']
                if expected_round is None or current_server_round >= expected_round:
                    print(f"Received model from server round {current_server_round}.")
                    return data['global_weights'], current_server_round
                else: print(f"Waiting for server round {expected_round} (current: {current_server_round})... polling in {poll_interval}s."); time.sleep(poll_interval)
            else: print("Server response invalid/model not ready. Waiting..."); time.sleep(poll_interval * 2)
        except requests.exceptions.Timeout: print("Warning: Timeout fetching weights. Retrying..."); time.sleep(10)
        except requests.exceptions.RequestException as e: print(f"Warning: Connection error fetching weights ({e}). Retrying..."); time.sleep(10)
        except json.JSONDecodeError: print("Warning: JSON decode error fetching weights. Retrying..."); time.sleep(10)

# --- (submit_update function) ---
def submit_update(round_num, weights, num_samples):
    payload = {"client_id": CLIENT_ID, "round": round_num, "weights": weights, "num_samples": num_samples}
    try:
        response = requests.post(f"{SERVER_URL}/submit_update", json=payload, timeout=45)
        if response.status_code == 409: return {"status": "Error", "message": "Round mismatch", "http_status": 409, "response_text": response.text}
        elif response.status_code == 400: return {"status": "Error", "message": "Bad request", "http_status": 400, "response_text": response.text}
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout: return {"status": "Error", "message": "Timeout during submission"}
    except requests.exceptions.HTTPError as e: return {"status": "Error", "message": f"HTTP Error {response.status_code}", "http_status": response.status_code, "response_text": response.text}
    except requests.exceptions.RequestException as e: return {"status": "Error", "message": str(e)}
    except json.JSONDecodeError: return {"status": "Error", "message": "Invalid JSON response", "response_text": response.text}

# --- (model_to_list function remains the same) ---
def model_to_list(model):
    return [w.tolist() for w in model.get_weights()]

# --- (list_to_weights function remains the same) ---
def list_to_weights(weights_list):
    if not isinstance(weights_list, list): raise TypeError(f"Expected list, got {type(weights_list)}")
    return [np.array(w) for w in weights_list]

def main():
    # --- Setup Results Directory ---
    results_dir = 'results'
    try: os.makedirs(results_dir, exist_ok=True); print(f"Results dir: '{results_dir}'")
    except OSError as e: print(f"Error creating results dir '{results_dir}': {e}"); sys.exit(1)

    # --- Load Data ---
    print(f"\n--- Client {CLIENT_ID}: Loading Data ---")
    X_train, y_train = load_data(TRAIN_CSV); print(f"Train data: {X_train.shape[0]} samples.")
    X_test, y_test = load_data(TEST_CSV); print(f"Test data: {X_test.shape[0]} samples.")

    # --- Initialize Model ---
    print(f"\n--- Client {CLIENT_ID}: Initializing Model ---")
    model = create_model()

    # --- Initial Synchronization ---
    print("\nFetching initial global model...")
    client_current_round = -1
    try:
        global_weights, server_round = get_global_weights(expected_round=0)
        model.set_weights(list_to_weights(global_weights))
        client_current_round = server_round
        print(f"Initialized model with weights from server round {client_current_round}.")
    except Exception as e: print(f"Initial sync error: {e}"); sys.exit(1)

    # --- Federated Learning Loop ---
    print(f"\n--- Client {CLIENT_ID}: Starting FL ({FED_ROUNDS} Rounds) ---")
    if client_current_round >= FED_ROUNDS:
         print(f"Target rounds ({FED_ROUNDS}) already met by server ({client_current_round}). Exiting.")
         plt.ioff(); fig.suptitle(f"Client {CLIENT_ID} - No rounds run", fontsize=12); sys.exit(0)

    for round_num in range(client_current_round, FED_ROUNDS):
        print(f"\n--- Round {round_num}/{FED_ROUNDS - 1} ---")

        # Record start time for the round
        round_start_time = time.time()

        # 1. Local Training
        print(f"Starting local training ({LOCAL_EPOCHS} epochs)...")
        train_loss, train_acc = None, None
        training_succeeded = False
        try:
            history = model.fit(X_train, y_train, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, class_weight=CLASS_WEIGHTS, verbose=1)
            if history and history.history and 'loss' in history.history and 'accuracy' in history.history:
                 train_loss = history.history['loss'][-1]
                 train_acc = history.history['accuracy'][-1]
                 print(f"Training complete. Final Epoch - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                 training_succeeded = True
            else: print("Warning: Invalid training history object.")
        except Exception as e: print(f"!!! Training error round {round_num}: {e} !!!")

        # Calculate round duration
        round_end_time = time.time()
        round_duration_seconds = round_end_time - round_start_time
        print(f"Round {round_num} completed in {round_duration_seconds:.2f} seconds.")

        if not training_succeeded:
            print("Skipping eval/submit. Attempting recovery fetch for next round.")
            next_round_num_recovery = round_num + 1
            if next_round_num_recovery < FED_ROUNDS:
                 try:
                      print(f"Recovery fetch for round {next_round_num_recovery}...")
                      global_weights, server_round = get_global_weights(expected_round=next_round_num_recovery)
                      model.set_weights(list_to_weights(global_weights)); client_current_round = server_round
                      print(f"Recovered weights for round {client_current_round}.")
                      continue
                 except Exception as fetch_e: print(f"Recovery fetch failed: {fetch_e}. Exiting."); sys.exit(1)
            else: print("Training failed on last round. Exiting."); sys.exit(1)

        # 2. Evaluate Model, Report Metrics, Plot
        print("Evaluating model on local test set...")
        test_loss, test_acc = None, None
        # --- Metrics calculation and CM plotting updated ---
        try:
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"Round {round_num} Test Eval - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = (y_pred_probs > 0.5).astype(int)
            class_labels = ["Negative", "Positive"]

            # --- Calculate all metrics ---
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Round {round_num} Test Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=0);
            print("\nClassification Report:"); print(report)

            # Call plot_metrics function (for interactive plot)
            plot_metrics(round_num, train_loss, train_acc, test_acc)

            # --- Generate Confusion Matrix with Metrics Text ---
            cm = confusion_matrix(y_test, y_pred)
            cm_fig, cm_ax = plt.subplots(figsize=(7, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
            disp.plot(ax=cm_ax, cmap=plt.cm.Blues, values_format='d')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_ax.set_title(f"{CLIENT_ID} - CM - Rd {round_num} ({timestamp})", fontsize=10)

            # Format metrics text with round duration
            metrics_text = (
                f"Accuracy:  {test_acc:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall:    {recall:.4f}\n"
                f"F1 Score:  {f1:.4f}\n"
                f"Round Duration: {round_duration_seconds:.2f}s" # Added duration here
            )
            # Adjust layout to make space and add text
            plt.subplots_adjust(bottom=0.25)
            cm_fig.text(0.5, 0.1, metrics_text, ha="center", va="bottom", fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.5))

            # Save the figure
            cm_filename = os.path.join(results_dir, f"confusion_matrix_{CLIENT_ID}_round_{round_num}_{timestamp}.png")
            cm_fig.savefig(cm_filename, bbox_inches='tight')
            plt.close(cm_fig)
            print(f"Saved CM plot with metrics to {cm_filename}")
            # --- End CM generation ---

        except Exception as e: print(f"!!! Error during eval/report/plot: {e} !!!")

        # 3. Submit Update to Server
        print("Submitting update to server...")
        try:
            weights_list = model_to_list(model)
            update_response = submit_update(round_num, weights_list, int(X_train.shape[0]))
            print(f"Server response: {update_response}")
        except Exception as e:
             print(f"Crit err pre-submission: {e}"); update_response = {"status": "Error", "message": "Client pre-submission err"}

        # 4. Handle Server Response & Prepare for Next Round
        if round_num < FED_ROUNDS - 1:
            next_round_num = round_num + 1
            print(f"Processing response & preparing for round {next_round_num}...")
            weights_successfully_set = False
            try:
                if not isinstance(update_response, dict):
                    print(f"Err: Invalid resp type: {type(update_response)}. Resp: {update_response}")
                    update_response = {"status": "Error", "message": "Invalid response type"}

                status = update_response.get("status")
                message = update_response.get("message", "")
                new_round = update_response.get("new_round")

                if status == "Error":
                    print(f"Submission/Server err: {message}.")
                    if update_response.get("http_status") == 409:
                         print("Round mismatch. Fetching current server state..."); global_weights, server_round = get_global_weights(expected_round=None); print(f"Re-synced to server round {server_round}.")
                    else: print("Attempting recovery fetch for next round."); global_weights, server_round = get_global_weights(expected_round=next_round_num)
                elif new_round is not None:
                    print(f"Server aggregated round {round_num}. New round {new_round}.")
                    server_round = new_round
                    if new_round < next_round_num: print(f"Warning: Server new round {new_round} < expected {next_round_num}. Polling..."); global_weights, server_round = get_global_weights(expected_round=next_round_num)
                    elif 'global_weights' in update_response: print("Using weights from submit response."); global_weights = update_response['global_weights']
                    else: print("Fetching weights via /global_model..."); global_weights, _ = get_global_weights(expected_round=server_round)
                else: print(f"Unexp server response: {update_response}. Recovery fetch for next round."); global_weights, server_round = get_global_weights(expected_round=next_round_num)

                model.set_weights(list_to_weights(global_weights))
                client_current_round = server_round
                weights_successfully_set = True
                print(f"Set weights for round {client_current_round}.")

            except Exception as e: print(f"!!! Error processing response/getting weights for {next_round_num}: {e} !!! Exiting."); sys.exit(1)
            if not weights_successfully_set: print("Error setting weights for next round. Exiting."); sys.exit(1)
        else:
            print(f"\nFinished final training round {round_num}.")
            break

    # --- End of Loop ---
    print(f"\n--- Client {CLIENT_ID}: FL process complete ---")

    # --- Save Final Plot ---
    print("Saving final metrics plot...")
    final_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rounds_completed = len(loss_history)
    fig.suptitle(f"Client {CLIENT_ID} - Final Metrics ({rounds_completed} Rounds Completed) - {final_timestamp}", fontsize=14)
    metrics_filename = os.path.join(results_dir, f"combined_metrics_{CLIENT_ID}_final_{final_timestamp}.png")
    try: fig.savefig(metrics_filename, bbox_inches='tight'); print(f"Saved final metrics plot to {metrics_filename}")
    except Exception as e: print(f"Error saving final plot: {e}")

    # --- Cleanup ---
    print("\nExecution complete. Close plot window to exit."); plt.ioff(); plt.show()

# --- Script Entry Point ---
if __name__ == '__main__':
    main()