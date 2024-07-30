import torch
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matrix size (increase for more GPU load)
matrix_size = 2048

# Create random matrices
A = torch.randn(matrix_size, matrix_size, device=device)
B = torch.randn(matrix_size, matrix_size, device=device)

# Record the start time
start_time = time.time()

# Run for approximately 30 seconds
while time.time() - start_time < 30:
    C = torch.mm(A, B)  # Matrix multiplication

# Print out a final message
print("Finished GPU stress test.")
