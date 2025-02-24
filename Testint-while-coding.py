import torch
print(f"Optimal num_workers: {torch.get_num_threads() // 2}")


# **********************************************************************************************************************************


from fastai.vision.all import *
path = Path("C:/Users/EFE/.fastai/data/oxford-iiit-pet/images")

files = get_image_files(path)
print(files[:5])  # Print first 5 files


# [Path('C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Abyssinian_1.jpg'), 
# Path('C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Abyssinian_10.jpg'), 
# Path('C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Abyssinian_100.jpg'), 
# Path('C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Abyssinian_101.jpg'), 
# Path('C:/Users/EFE/.fastai/data/oxford-iiit-pet/images/Abyssinian_102.jpg')]




# **********************************************************************************************************************************


# ✅ Yes! You must set the seed every time if you want to get the same random values.

# If you don’t set the seed again, the randomness will change in every run.

# 🔹 Example: What Happens Without Resetting the Seed?
import random

random.seed(42)  
print(random.randint(1, 100))  # Output: 81

# # No seed set here, so the next number is random
print(random.randint(1, 100))  # Output: 14 (random every run)

# # Set seed again
random.seed(42)  
print(random.randint(1, 100))  # Output: 81 (same as first time!)
# ✔ First and last values are the same because we reset the seed to 42.
# ✔ Middle value is different every time because we didn’t reset the seed.

# 🔹 Rule of Thumb
# ✅ If you want the same results every time → Set the seed before generating random values.
# ❌ If you don’t set the seed again → The sequence will continue unpredictably.




# **********************************************************************************************************************************

# 2️⃣ Understanding for _ in range(5)

[random.randint(1, 100) for _ in range(5)]
# 📌 This is called a list comprehension. It creates a list of 5 random numbers.
# 🔹 The _ is a throwaway variable that means we don’t need its value.

# 📌 What Does _ Mean?
# Normally, when you use a loop, you write:


for i in range(5):
    print(i)
# This prints 0, 1, 2, 3, 4 because i holds values.

# But in:
for _ in range(5):
    pass
# We don’t need to use the loop variable (_).
# It just runs the loop 5 times.
# 🔹 _ is just a placeholder to indicate that we are looping but don’t care about the loop variable.




# **********************************************************************************************************************************


# 3️⃣ What Happens Without random.seed(10)?

import random

print([random.randint(1, 100) for _ in range(5)])  
# Output: [Random numbers, different every time!]

print([random.randint(1, 100) for _ in range(5)])  
# Output: [Another random sequence, different from above!]
# ❌ Since we didn’t set random.seed(), the numbers are different every time we run the code.

# ✔ With random.seed(10), we always get the same result!




# 4️⃣ Trying Different Seed Values
# If you change the seed, the numbers will be different, but still repeatable.

import random

random.seed(42)  
print([random.randint(1, 100) for _ in range(5)])  
# Output: [82, 15, 4, 95, 36]

random.seed(42)  
print([random.randint(1, 100) for _ in range(5)])  
# Output: [82, 15, 4, 95, 36] (same numbers!)

random.seed(99)  
print([random.randint(1, 100) for _ in range(5)])  
# Output: [9, 81, 64, 30, 78] (different numbers!)
# ✔ Different seeds → Different fixed random sequences!
# ✔ Same seed → Same sequence every time!




