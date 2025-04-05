# app/main.py
import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import components - initialization happens within these calls now
from app.rag_component import get_rag_chain, initialize_vector_store, GOOGLE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)
log.info("Flask-Cors enabled for all origins.")

# --- Application State (Initialization Status) ---
# We don't pre-initialize here anymore for local run clarity,
# it will happen on the first call to get_rag_chain()
initialization_error = None
rag_chain_instance = None # Will be populated by get_rag_chain

# --- API Endpoints ---
@app.route("/recommend", methods=['POST'])
def recommend_assessment():
    global rag_chain_instance, initialization_error
    log.debug("'/recommend' endpoint hit.")

    # Try to get/initialize the chain on the first request if not done yet
    # This block also handles checking the initialization_error state implicitly
    try:
        if rag_chain_instance is None and initialization_error is None:
             log.info("RAG chain not initialized yet, attempting initialization now...")
             rag_chain_instance = get_rag_chain() # This will build DB if needed
             log.info("RAG chain ready.")
    except Exception as e:
         log.exception("ERROR during on-demand RAG initialization!")
         initialization_error = f"On-Demand Init Failed: {type(e).__name__} - Check logs."
         # Fall through to error handling below

    # Check for initialization failure state
    if initialization_error:
         log.error(f"Initialization error detected. Returning 503. Error: {initialization_error}")
         return jsonify({"error": f"Service Unavailable: {initialization_error}"}), 503
    if not rag_chain_instance: # Should not happen if error is None, but safety check
         log.error("RAG chain is still None after initialization attempt. Unexpected state.")
         return jsonify({"error": "Service Unavailable: RAG components not ready."}), 503

    # --- Request Validation and Processing (Same as before) ---
    if not request.is_json:
        # ... (return 415 error) ...
        log.warning("Request Content-Type is not application/json.")
        return jsonify({"error": "Request must be JSON."}), 415
    try:
        data = request.get_json()
        if not data or 'question' not in data or not isinstance(data['question'], str) or not data['question'].strip():
            # ... (return 400 error) ...
            log.warning(f"Invalid request payload received: {data}")
            return jsonify({"error": "Invalid request body. Required: {'question': 'your non-empty query'}."}), 400
        question = data['question']
        log.info(f"Processing recommendation for question: '{question[:100]}...'")
    except Exception as e:
         # ... (return 400 error) ...
         log.error(f"Error parsing request JSON: {e}", exc_info=True)
         return jsonify({"error": "Invalid JSON format in request body."}), 400
    try:
        result = rag_chain_instance.invoke(question)
        log.info(f"Successfully generated recommendation for question: '{question[:100]}...'")
        return jsonify({"answer": result}), 200
    except Exception as e:
        # ... (return 500 error) ...
        log.exception(f"Error invoking RAG chain for question: '{question[:100]}...'")
        return jsonify({"error": "An internal error occurred while processing the request."}), 500


@app.route("/health", methods=['GET'])
def health_check():
    """Basic health check. Returns unhealthy if initialization failed."""
    log.debug("'/health' endpoint hit.")
    # Check the stored initialization error state
    if initialization_error:
         log.warning(f"Health check reporting unhealthy due to initialization error: {initialization_error}")
         return jsonify({"status": "unhealthy", "reason": initialization_error}), 503
    # If no error stored, assume basic OK for now
    # Could add a non-blocking check here later if needed
    return jsonify({"status": "ok"}), 200


# --- Main execution block for LOCAL TESTING ONLY ---
if __name__ == "__main__":
    log.info("--- Starting Flask Development Server for Local Testing ---")
    log.info("NOTE: Vector DB will be BUILT on the first '/recommend' request if it doesn't exist.")
    # Ensure API key is loaded locally (dotenv should handle this)
    if not GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not found in environment. API calls will likely fail.")
        print("\n*** WARNING: GOOGLE_API_KEY not found in .env file or environment! ***\n")

    # Define port for local dev server
    local_port = int(os.environ.get("PORT", 5001)) # Use 5001 to avoid conflicts
    log.info(f"Attempting to start server on http://0.0.0.0:{local_port}")
    # Run the app
    # debug=False is safer even for local testing with sensitive operations
    app.run(host='0.0.0.0', port=local_port, debug=False)