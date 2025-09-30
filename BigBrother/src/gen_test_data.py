import numpy as np
import os

# Create folders if they don't exist
os.makedirs("data/data", exist_ok=True)
os.makedirs("data/label", exist_ok=True)

num_samples = 900

for i in range(num_samples):
    # Random 2-input sample
    x1 = np.random.rand()
    x2 = np.random.rand()
    x = np.array([x1, x2])

    # Compute sum and difference
    y = np.array([x1 + x2, x1 - x2]) # sum and difference

    # Save input and label with matching filenames
    np.save(f"data/data/sample_{100+i}.npy", x)
    np.save(f"data/label/sample_{100+i}.npy", y)

print("100 samples generated in 'data' and 'label' folders.")