import os; os.add_dll_directory(r"C:\CUDA_Manual\bin"); import tensorflow as tf; print("\n\n=== GPU STATUS ==="); print("Available:", tf.config.list_physical_devices("GPU"))  
