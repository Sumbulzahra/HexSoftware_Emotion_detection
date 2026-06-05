import gradio as gr
from transformers import pipeline
from PIL import Image

# 1. Load the pre-trained Facial Emotion Recognition AI model
try:
    classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# 2. Process the continuous streaming camera frames
def predict_live_emotion(frame):
    if not model_loaded:
        return "Model failed to load. Please check your connection."
   
    if frame is None:
        return "Waiting for live camera stream feed..."
   
    try:
        # Convert the incoming webcam numpy array frame to PIL format
        pil_img = Image.fromarray(frame)
       
        # Run the Vision Transformer prediction
        results = classifier(pil_img)
       
        # Grab top prediction parameters
        top_prediction = results[0]['label']
        score = results[0]['score'] * 100
       
        emoji_dict = {
            "sad": "😢 Sadness",
            "disgust": "🤢 Disgust",
            "angry": "😡 Anger",
            "fear": "📁 Fear / Anxiety",
            "happy": "😊 Happiness / Joy",
            "surprise": "😮 Surprise",
            "neutral": "😐 Neutral / Calm"
        }
       
        detected_emotion = emoji_dict.get(top_prediction.lower(), top_prediction.capitalize())
        return f"Live Status: {detected_emotion} ({score:.2f}% Confidence)"
       
    except Exception as err:
        return "Processing live stream... Keep looking at the camera."

# 3. Build the Automated Live Streaming Interface
# Note: Theme configuration moved to launch() to support Gradio 6.0 standards
with gr.Blocks() as demo:
    gr.Markdown("# 👁️ Live Automated Facial Emotion Detection AI")
    gr.Markdown("This system establishes a continuous live video loop. It reads your facial expressions frame-by-frame instantly without requiring manual capture clicking.")
   
    with gr.Row():
        with gr.Column():
            # streaming=True tells Gradio to continuously send frames
            webcam_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Front Camera Feed", type="numpy")
           
        with gr.Column():
            output_text = gr.Textbox(label="Real-Time AI Output:", interactive=False, placeholder="Awaiting camera transmission stream...")

    # FIX: Removed the deprecated time_per_frame argument to support Gradio 6.0 structure
    webcam_stream.stream(
        fn=predict_live_emotion,
        inputs=webcam_stream,
        outputs=output_text
    )
   
    gr.Markdown("---")
    gr.Markdown("_Fully Automated Video Stream Version built for Internship Submission._")

# Launch with the chosen theme applied directly
demo.launch(theme=gr.themes.Soft()) 
