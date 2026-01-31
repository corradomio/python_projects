import torch

if torch.cuda.is_available():
    # Get the total number of available devices
    num_gpus = torch.cuda.device_count()
    print(f"Total number of available GPUs: {num_gpus}")

    # List the name of each GPU
    for i in range(num_gpus):
        print(f"GPU Device {i}: {torch.cuda.get_device_name(i)}")

    # Print the name of the *current* default device (usually device 0)
    print(f"Current default GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print(f"NO GPUs available!!")