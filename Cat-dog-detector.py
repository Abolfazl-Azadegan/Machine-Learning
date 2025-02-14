# CLICK ME
from fastai.vision.all import *
# path = untar_data(URLs.PETS)/'images'
path ='C:\\Users\\EFE\\.fastai\\data\\oxford-iiit-pet\\images'
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224), num_workers=0  # Disable multiprocessing
)
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
# Save the trained model
model_path = Path('C:\\Users\\EFE\\.fastai\\models')
model_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
learn.export(model_path / 'pet_classifier.pkl')

print("Model saved successfully!")
