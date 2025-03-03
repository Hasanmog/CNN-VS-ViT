import os
import torch
import onnx
from model import Segmentor
from inference import load_model

def convert_to_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)

    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    onnx_dir = "weights"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "best_checkpoint.onnx")

    dynamic_axes = {"image_input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image_input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"Model successfully converted and saved to {onnx_path}")

if __name__ == "__main__":
    convert_to_onnx()
