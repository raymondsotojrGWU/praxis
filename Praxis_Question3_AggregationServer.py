"""
Copyright Raymond Soto Jr. D.Eng(c).
From Edge to Enterprise:
Federated Learning Threat Classification with Heterogeneous Devices in Converged Energy Sector Networks
Revised July 9th, 2025
"""
# Praxis Aggregation Server
from flask import Flask, request, jsonify
import numpy as np
import threading
import tensorflow as tf
import os
import pickle
import logging
import sys

# --- Configuration ---
HOST = os.environ.get("FL_HOST", "192.168.1.198")
PORT = int(os.environ.get("FL_PORT", 5000))
INITIAL_MODEL_PATH = os.environ.get("FL_MODEL_PATH", "models/M11.h5")
EXPECTED_CLIENTS = int(os.environ.get("FL_EXPECTED_CLIENTS", 2))
STATE_FILE = 'fl_server_state.pkl'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Add FileHandler for persistence

# --- Flask App ---
app = Flask(__name__)

# --- Custom Layer (if needed) ---
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super(CustomInputLayer, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super(CustomInputLayer, cls).from_config(config)

# --- Global State ---
global_weights = None
current_round = 0
updates = {}
lock = threading.Lock()

# --- Helper Functions ---
def weighted_average(weights_updates):
    if not weights_updates:
        logging.warning("weighted_average called with empty updates list.")
        return []

    # Calculate total number of samples
    total_samples = sum(update['num_samples'] for update in weights_updates)

    # Handle edge case: division by zero if total samples is 0
    if total_samples == 0:
        logging.warning("Total samples is zero during weighted average. Cannot compute average. Returning weights from the first client as fallback.")
        # Return weights from the first client as a fallback, ensuring they are numpy arrays
        try:
            # Assuming weights are list-of-lists from JSON
            first_weights = [np.array(w) for w in weights_updates[0]['weights']]
            return first_weights
        except Exception as e:
             logging.error(f"Error processing first client's weights during zero-sample fallback: {e}", exc_info=True)
             return []

    try:
        # Convert first client's weights to numpy arrays to get shapes
        first_client_np_weights = [np.array(w) for w in weights_updates[0]['weights']]
        num_layers = len(first_client_np_weights)
        # Initialize structure to hold the sum (using zeros like the first client's shapes)
        weighted_sum_accumulator = [np.zeros_like(w, dtype=np.float64) for w in first_client_np_weights] # Use float64 for precision
    except Exception as e:
         logging.error(f"Error initializing accumulator based on first client's weights: {e}", exc_info=True)
         return []

    # Iterate through each layer
    for layer_idx in range(num_layers):
        # Accumulate weighted sum for the current layer from all clients
        try:
            for update in weights_updates:
                # Convert current client's layer weights to numpy array
                client_weight_layer = np.array(update['weights'][layer_idx], dtype=np.float64)
                # Check shape consistency
                if client_weight_layer.shape != weighted_sum_accumulator[layer_idx].shape:
                    logging.error(f"Shape mismatch detected in layer {layer_idx} from client update. Expected {weighted_sum_accumulator[layer_idx].shape}, got {client_weight_layer.shape}. Aborting aggregation.")
                    return []

                # Add weighted contribution to the sum for this layer
                weighted_sum_accumulator[layer_idx] += update['num_samples'] * client_weight_layer

        except Exception as e:
             logging.error(f"Error processing layer {layer_idx} during weighted average: {e}", exc_info=True)
             return []

    # Calculate the final averaged weights by dividing the sum by total samples
    try:
        # Perform the division for each layer's accumulated sum
        averaged_weights = [(layer_sum / total_samples) for layer_sum in weighted_sum_accumulator]
    except Exception as e:
         logging.error(f"Error calculating final average from sums: {e}", exc_info=True)
         return []

    logging.info(f"Successfully calculated weighted average over {total_samples} total samples.")
    # This variable now holds the result as a list of NumPy arrays
    return averaged_weights

def load_initial_model(path):
    if not os.path.exists(path):
         logging.error(f"Fatal error: Initial model file not found at {path}")
         sys.exit(1) # Use sys.exit
    try:
        logging.info(f"Loading global model from {path}")
        with tf.keras.utils.custom_object_scope({'CustomInputLayer': CustomInputLayer}):
            model = tf.keras.models.load_model(path)

        # Extract weights as JSON-serializable lists.
        weights = [w.tolist() for w in model.get_weights()]
        logging.info("Global model loaded and weights extracted.")
        return weights # Returns list of lists
    except Exception as e:
        logging.error(f"Fatal error loading global model from {path}: {e}", exc_info=True)
        sys.exit(1)

def save_state(weights, round_num):
    try:
        # Ensure weights are JSON-serializable (list of lists) before saving
        weights_to_save = weights
        # Check if weights are numpy arrays (as returned by weighted_average)
        if weights and isinstance(weights[0], np.ndarray):
             logging.debug("Converting numpy weights to lists for saving state.")
             weights_to_save = [w.tolist() for w in weights]
        elif not isinstance(weights, list) or (weights and not isinstance(weights[0], list)):
              logging.warning(f"Attempting to save weights in unexpected format: {type(weights)}. Converting to list if possible.")

              if hasattr(weights, 'tolist'):
                 weights_to_save = weights.tolist()
              elif isinstance(weights, list):
                 weights_to_save = [np.array(w).tolist() for w in weights]
              else:
                 raise TypeError("Weights format is not list of lists or list of ndarrays.")


        with open(STATE_FILE, 'wb') as f:
            pickle.dump({'weights': weights_to_save, 'round': round_num}, f)
        logging.info(f"Server state saved for round {round_num}")
    except Exception as e:
        # Log warning
        logging.warning(f"Could not save server state to {STATE_FILE}: {e}", exc_info=True)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                saved_state = pickle.load(f)
                # Basic validation
                if 'weights' in saved_state and 'round' in saved_state and isinstance(saved_state['weights'], list):
                    logging.info(f"Loaded server state from {STATE_FILE}. Resuming at round {saved_state['round']}.")
                    # Weights are loaded as list of lists from pickle file
                    return saved_state['weights'], saved_state['round']
                else:
                    logging.warning(f"State file {STATE_FILE} has invalid format or missing keys. Starting fresh.")
                    return None, 0
        except Exception as e:
            logging.warning(f"Error loading state file {STATE_FILE}: {e}. Will load initial model.", exc_info=True)
            return None, 0
    else:
        logging.info("No state file found. Starting fresh.")
        return None, 0

# --- Flask Routes ---
@app.route('/global_model', methods=['GET'])
def get_global_model():
    global global_weights, current_round
    with lock:
        if global_weights is None:
            logging.warning("GET /global_model called before model initialized.")
            return jsonify({'message': 'Global model not initialized yet', 'round': current_round}), 503 # Service Unavailable

        logging.info(f"Sending global model for round {current_round}")
        # Ensure weights sent are in list format for JSON
        weights_to_send = global_weights
        # Check if global_weights are stored as numpy arrays
        if global_weights and isinstance(global_weights[0], np.ndarray):
             logging.debug("Converting numpy weights to lists for JSON response.")
             weights_to_send = [w.tolist() for w in global_weights]
        elif not isinstance(weights_to_send, list) or (weights_to_send and not isinstance(weights_to_send[0], list)):
             logging.warning("Global weights are in an unexpected format before sending. Attempting conversion.")
             try:
                 weights_to_send = [np.array(w).tolist() for w in global_weights]
             except Exception:
                 logging.error("Failed to convert global weights to list format for sending.", exc_info=True)
                 return jsonify({'message': 'Internal server error: Cannot format weights.'}), 500

        return jsonify({'global_weights': weights_to_send, 'round': current_round})


@app.route('/submit_update', methods=['POST'])
def submit_update():
    global global_weights, current_round, updates
    client_id = None
    try:
        # --- 1. Parse Request ---
        data = request.get_json()
        if not data:
            logging.warning("Received empty JSON payload.")
            return jsonify({'message': 'Bad Request: No JSON payload.'}), 400

        client_id = data.get('client_id')
        round_num = data.get('round')
        client_weights = data.get('weights') # Expected format: list of lists
        num_samples = data.get('num_samples')

        # Basic validation of received data
        if not client_id:
             logging.warning("Received update with missing client_id.")
             return jsonify({'message': 'Bad Request: Missing client_id.'}), 400
        if round_num is None:
             logging.warning(f"Received update from {client_id} with missing round number.")
             return jsonify({'message': 'Bad Request: Missing round number.'}), 400
        if not isinstance(client_weights, list):
             logging.warning(f"Received update from {client_id} where 'weights' is not a list.")
             return jsonify({'message': 'Bad Request: Invalid weights format (must be a list).'}), 400
        if num_samples is None or not isinstance(num_samples, int) or num_samples < 0:
             logging.warning(f"Received update from {client_id} with invalid num_samples: {num_samples}")
             return jsonify({'message': 'Bad Request: Invalid or missing num_samples.'}), 400

    except Exception as e:
        # --- Catch errors during parsing/initial validation ---
        logging.error(f"Error parsing request JSON or initial field access: {e}", exc_info=True)
        return jsonify({'message': 'Bad Request: Could not parse JSON or invalid format.'}), 400

    with lock:
        # --- 2. Validate Round ---
        if round_num != current_round:
            logging.warning(f"Client {client_id} submitted for wrong round ({round_num}), server is at {current_round}.")
            return jsonify({'message': f'Round mismatch. Server round: {current_round}. Client round: {round_num}.'}), 409 # Conflict

        # --- 3. Check for Duplicate Update ---
        if client_id in updates:
             logging.warning(f"Client {client_id} submitted update multiple times for round {current_round}. Ignoring duplicate.")
             # Return success but don't re-process or count again
             return jsonify({'message': 'Duplicate update ignored. Waiting for other client(s).'})

        # --- 4. Store Valid Update ---
        # Store weights as received (list of lists)
        updates[client_id] = {'weights': client_weights, 'num_samples': num_samples}
        logging.info(f"Received valid update from client {client_id} for round {current_round} ({len(updates)}/{EXPECTED_CLIENTS} received).")

        # --- 5. Check if Aggregation is Ready ---
        if len(updates) >= EXPECTED_CLIENTS:
            logging.info(f"Starting aggregation for round {current_round} with updates from: {list(updates.keys())}")
            try:
                # --- 6. Perform Aggregation ---
                # weighted_average expects list of dicts {'weights': list_of_lists, 'num_samples': int}
                # It returns weights as list of numpy arrays
                aggregated_weights_np = weighted_average(list(updates.values()))

                # Check if aggregation failed (returned empty list)
                if not aggregated_weights_np:
                     logging.error(f"Weighted average function returned empty list for round {current_round}. Aggregation failed.")
                     updates = {}
                     return jsonify({'message': 'Internal server error during aggregation (weighted_average failed).'}), 500

                # Store weights globally as list of NumPy arrays
                global_weights = aggregated_weights_np
                # Increment round *before* saving state for that new round
                current_round += 1
                # Clear updates for the completed round
                updates = {}
                logging.info(f"Aggregation complete. Global model generated for new round {current_round}.")

                # --- 7. Save State (includes new round number and weights) ---
                save_state(global_weights, current_round)

                # --- 8. Send Success Response ---
                response_weights = [w.tolist() for w in global_weights]
                return jsonify({
                    'message': 'Update received and global model aggregated',
                    'new_round': current_round,
                    'global_weights': response_weights
                })

            except Exception as e:
                # --- Catch errors during aggregation or state saving ---
                logging.error(f"Unexpected error during aggregation/state saving block for round {current_round} (before increment): {e}", exc_info=True)
                updates = {}
                return jsonify({'message': 'Internal server error during aggregation processing.'}), 500
        else:
            # --- Not enough updates yet, tell client to wait ---
            return jsonify({'message': f'Update received. Waiting for {EXPECTED_CLIENTS - len(updates)} more client(s) for round {current_round}.'})


# --- Main Execution ---
if __name__ == '__main__':
    logging.info("--- Federated Learning Server Starting ---")
    logging.info(f"Host: {HOST}, Port: {PORT}")
    logging.info(f"Initial Model Path: {INITIAL_MODEL_PATH}")
    logging.info(f"Expected clients per round: {EXPECTED_CLIENTS}")
    logging.info(f"State file: {STATE_FILE}")

    # Attempt to load state first
    loaded_weights, loaded_round = load_state()
    if loaded_weights is not None:
        global_weights = loaded_weights
        current_round = loaded_round
    else:
        # If no state, load initial model
        global_weights = load_initial_model(INITIAL_MODEL_PATH)
        current_round = 0

    # Ensure global_weights is not None before starting server
    if global_weights is None:
        logging.error("FATAL: Failed to initialize global_weights from state file or initial model.")
        sys.exit(1)

    logging.info(f"Server ready. Initial round: {current_round}")

    try:
        from waitress import serve
        logging.info(f"Starting server using Waitress on http://{HOST}:{PORT}")
        serve(app, host=HOST, port=PORT)
    except ImportError:
        logging.warning("Waitress not found, using Flask development server (not recommended for production).")
        logging.warning("Install waitress for a production-ready server: pip install waitress")
        app.run(host=HOST, port=PORT, debug=False)
    except Exception as e:
         logging.error(f"Failed to start server: {e}", exc_info=True)
         sys.exit(1)