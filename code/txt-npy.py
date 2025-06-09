import numpy as np

# Load the text file (assuming space-separated values)
data = np.loadtxt("param.txt")  # Replace 'data.txt' with your actual file name

# Save as .npy file
np.save("params.npy", data)

# Verify the shape
loaded_data = np.load("params.npy")
print("Shape:", loaded_data.shape)  # Should print (1000, 6)
