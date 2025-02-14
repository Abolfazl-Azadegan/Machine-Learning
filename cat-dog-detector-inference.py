from fastai.vision.all import *
from pathlib import Path

# Re-define the missing function
def is_cat(x): 
    return x[0].isupper()  # This is the function used in training

# Load the trained model
model_path = Path("C:/Users/EFE/.fastai/models/pet_classifier.pkl")  # Update with the correct path
learn_loaded = load_learner(model_path)

# Path to the image you want to classify
img_path = Path("C:/Users/EFE/Desktop/download (1).jpeg")  # Change this to your actual image path

# Check if the image file exists
if not img_path.exists():
    raise FileNotFoundError(f"Error: Image not found at {img_path}")

# Predict the class of the image
pred, pred_idx, probs = learn_loaded.predict(img_path)

# Print the result
print(f"Predicted: {'Cat' if pred else 'Dog'}, Probability: {probs[pred_idx]:.4f}")
