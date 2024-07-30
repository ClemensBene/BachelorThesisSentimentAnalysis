#PyTorch
import torch

# Überprüfen, ob eine GPU verfügbar ist
if torch.cuda.is_available():
    print("CUDA ist verfügbar")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA ist nicht verfügbar")

# Ein Beispiel-Tensor erstellen und auf die GPU verschieben
x = torch.randn(5, 3).to(torch.device("cuda"))
print(x)
print("Dimensionen des Tensors:", x.size())