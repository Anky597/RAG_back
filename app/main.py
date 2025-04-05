# app/main.py (Gradio Version)

import logging
import os
import gradio as gr # Import Gradio

# Import the function that gets the chain and handles initialization implicitly
# Ensure RAG components handles GOOGLE_API_KEY loading via os.getenv
from app.rag_component import get_rag_chain, GOOGLE_API_KEY # Use absolute import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

# --- Application Initialization & State ---
# Attempt to initialize components eagerly when the module is loaded
rag_chain_instance = None
initialization_error = None
log.info("Gradio app module loaded. Attempting RAG component initialization (will build DB if needed)...")
try:
    # Check if API Key was loaded by rag_components (optional check here)
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set or empty.")

    # Initialize components via get_rag_chain. This triggers DB build if needed.
    # This can be slow on startup!
    rag_chain_instance = get_rag_chain()
    log.info("RAG components initialization attempt finished successfully.")

except Exception as e:
    log.exception("FATAL ERROR during application startup initialization.")
    initialization_error = f"Initialization Failed: {type(e).__name__} - Check server logs."
    # Gradio app will still launch, but the function will fail.


# --- Define the Core Processing Function ---
def get_recommendation(user_question: str) -> str:
    """Takes user question and returns RAG recommendation or error message."""
    log.info(f"Processing request via Gradio function for: '{user_question[:100]}...'")

    # Check initialization status on each call (important!)
    if initialization_error:
        log.error(f"Returning error due to initialization failure: {initialization_error}")
        # You might want to raise an exception here for Gradio to catch
        # raise gr.Error(f"Service Initialization Failed: {initialization_error}")
        return f"Error: Service Initialization Failed - Check application logs. ({initialization_error})" # Return error string

    if not rag_chain_instance:
        log.error("RAG chain instance is not available. Initialization might have silently failed.")
        # raise gr.Error("Service Unavailable: RAG components not ready.")
        return "Error: Service is not ready. Please try again later or check logs."

    if not user_question or not isinstance(user_question, str) or not user_question.strip():
        log.warning("Received empty or invalid question.")
        # raise gr.Error("Please enter a valid question.")
        return "Error: Please enter a question."

    try:
        # Invoke the RAG chain
        result = rag_chain_instance.invoke(user_question)
        log.info("Successfully generated recommendation.")
        return result # Return the answer string
    except Exception as e:
        log.exception(f"Error invoking RAG chain for question: '{user_question[:100]}...'")
        # raise gr.Error(f"An internal error occurred: {e}") # Raise exception for Gradio UI
        return f"Error: An internal processing error occurred. Check logs. ({type(e).__name__})" # Return error string


# --- Create Gradio Interface ---
log.info("Creating Gradio Interface...")
# Use Blocks for more layout control if needed, Interface is simpler
iface = gr.Interface(
    fn=get_recommendation, # The function to call
    inputs=gr.Textbox(lines=5, label="Your Question / Role Description", placeholder="e.g., Need cognitive tests for graduate engineers focusing on problem solving..."),
    outputs=gr.Markdown(label="SHL Assessment Recommendation"), # Use Markdown for better text formatting
    title="SHL RAG Assessment Recommender",
    description="Enter your requirements below to get AI-powered SHL assessment recommendations based on product data. Initialization (including DB build) happens on startup and can take several minutes.",
    allow_flagging="never", # Disable flagging unless you set it up
    # examples=[ # Optionally add example queries
    #     ["What is the OPQ?"],
    #     ["Which assessments help evaluate suitability for remote work?"],
    #     ["I need to assess numerical reasoning and problem-solving for graduate engineers."]
    # ]
)
log.info("Gradio Interface created.")

# --- Launch the Gradio App ---
# When running in Hugging Face Spaces, HF usually handles the launch.
# This block is mainly for local testing.
if __name__ == "__main__":
    log.info("Attempting to launch Gradio app locally...")
    # HF Spaces usually uses port 7860
    # Use share=False for local testing unless you need public link
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
    log.info("Gradio app launched.")
