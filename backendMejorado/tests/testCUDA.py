import torch


print("¿CUDA disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
else:
    print("No se detectó GPU compatible con CUDA.")
    