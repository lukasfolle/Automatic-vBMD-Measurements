def device_check(volume):
    if (device := str(volume.device)) != "gpu":
        print(f"Warning: Tensor is on device {device}. Consider moving tensor to gpu first.")
