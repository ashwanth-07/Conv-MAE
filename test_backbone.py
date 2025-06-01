# example_backbone_test.py

import torch
from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

def main():
    # Instantiate the EfficientViT‐B0 backbone
    model = efficientvit_backbone_b0()
    model.eval()

    # Create a dummy input tensor: batch size = 1, 3 channels, 224×224 pixels
    dummy_input = torch.randn(1, 3, 224, 224)

    # Forward pass through the backbone
    with torch.no_grad():
        features = model(dummy_input)

    # Print out the names and shapes of the returned feature maps
    print("Feature‐map shapes from EfficientViT‐B0 backbone:")
    for stage_name, fmap in features.items():
        print(f"  {stage_name}: {tuple(fmap.shape)}")

if __name__ == "__main__":
    main()
