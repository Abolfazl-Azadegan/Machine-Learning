# CLICK ME
from fastai.vision.all import *
# path = untar_data(URLs.PETS)/'images'
path ='C:\\Users\\EFE\\.fastai\\data\\oxford-iiit-pet\\images'
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224), num_workers=0  # Disable multiprocessing
)


# dls = ImageDataLoaders.from_name_func(
#     path,                          # Path where images are stored
#     get_image_files(path),         # Get all image file paths
#     valid_pct=0.2,                 # 20% of data will be used for validation
#     seed=42,                       # Seed for reproducibility
#     label_func=is_cat,             # Function to generate labels
#     item_tfms=Resize(224),         # Image transformation: Resize to 224x224
#     num_workers=0                  # Disable multiprocessing (useful for Windows)
# )

# ImageDataLoaders.from_name_func()
# This automatically creates a DataLoader from images, using a function to extract labels based on filenames.

# 📌 Equivalent Alternative:

# ImageDataLoaders.from_name_re() – Uses regex patterns instead of a function.
# ImageDataLoaders.from_folder() – Uses folder names as labels.


# 🔹 2. path
# 📌 Purpose: The directory containing image files.
# 📌 Example:

# 🔹 Options:

# Can be an absolute (C:/Users/...) or relative (./data/images) path.
# Must contain images in supported formats (.jpg, .png, etc.).



# 🔹 3. get_image_files(path)
# 📌 Purpose: Retrieves all image file paths from path.
# 📌 Returns: A list of image file paths.

# ✅ Example Output:


# ['C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Birman_12.jpg', 
#  'C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Siamese_45.jpg']

# 🔹 Alternative Options:
# If images are nested in folders:

# get_image_files(path, recurse=True)  # Searches subdirectories
# If you want only .png images:

# get_image_files(path, extensions=['.png'])


# 🔹 4. valid_pct=0.2
# 📌 Purpose: Sets aside 20% of the images for validation.

# ✅ Options:

# valid_pct=0.1 → 10% validation, 90% training
# valid_pct=0.3 → 30% validation, 70% training
# valid_pct=0.0 → No validation set
# 🔹 Fixed Splitting: If you want to specify the validation set manually, use:

# splitter = GrandparentSplitter(valid_name='valid')
# dls = ImageDataLoaders.from_name_func(path, get_image_files(path), label_func=is_cat, splitter=splitter)



# 🔹 5. seed=42
# 📌 Purpose: Ensures that the train-validation split is reproducible.
# 📌 Example:

# seed=42 (default in machine learning for reproducibility)
# seed=123 (use any integer)
# 🔹 Why is this important?

# If seed is not set, every time you run the code, FastAI randomly selects different validation images.
# Setting seed=42 ensures consistent validation images across runs.



# 🔹 6. label_func=is_cat
# 📌 Purpose: Defines how to label images.
# 📌 Example:


# def is_cat(x): 
#     return x[0].isupper()  # Cats have filenames starting with uppercase letters
# ✅ Other ways to label images:

# Using a Regex Pattern (from_name_re)

# label_func = r'(.+)_\d+.jpg'  # Extracts 'Birman' from 'Birman_12.jpg'
# Using Folder Names (from_folder)

# dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)



# 🔹 7. item_tfms=Resize(224)
# 📌 Purpose: Resizes every image to 224×224 pixels.
# 📌 Example:


# item_tfms=Resize(224)
# 🔹 Options:

# Resize(128) → Shrinks all images to 128x128
# Resize(224, method='crop') → Crops the center instead of stretching
# Resize(224, method='pad') → Pads images instead of cropping
# ✅ Why resize?

# Different images have different sizes (e.g., 500x600, 800x300).
# To train a model, all images must be the same size.
# 🔹 Alternative: Augmentations (More Powerful)


# item_tfms = RandomResizedCrop(224, min_scale=0.8)  # Crop & Resize dynamically



# 🔹 8. num_workers=0
# 📌 Purpose: Number of CPU cores used for data loading.
# 📌 Example:


# num_workers=4  # Uses 4 CPU cores (faster)
# ✅ Options:

# num_workers=0 → Single-threaded (safe for Windows)
# num_workers=2 → Uses 2 CPU cores
# num_workers=4 → Uses 4 CPU cores (Recommended for Linux/macOS)
# 📌 Why use 0 on Windows?
# Multiprocessing can cause errors on Windows, so setting num_workers=0 avoids crashes.





# 🔹 Understanding valid_pct (Validation Percentage) in FastAI
# In FastAI, valid_pct (validation percentage) is a parameter that controls how much of the dataset is set aside for validation.

# python
# Copy
# Edit
# dls = ImageDataLoaders.from_name_func(
#     path, get_image_files(path), valid_pct=0.2, seed=42,  # 20% for validation
#     label_func=is_cat, item_tfms=Resize(224)
# )
# 📌 valid_pct=0.2 means:

# 80% of the images → Used for training (model learns from these).
# 20% of the images → Used for validation (model is tested on these).
# You can change valid_pct to any value between 0 and 1:

# valid_pct=0.1 → 10% validation, 90% training
# valid_pct=0.3 → 30% validation, 70% training
# valid_pct=0.5 → 50% validation, 50% training
# ✔️ You are not limited to predefined values. You can enter any percentage.




learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
# Save the trained model
model_path = Path('C:\\Users\\EFE\\.fastai\\models')
model_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
learn.export(model_path / 'pet_classifier.pkl')

print("Model saved successfully!")
