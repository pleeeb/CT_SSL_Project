import torch
import os
from collections import OrderedDict


def extract_yolo_backbone_from_checkpoint(
        checkpoint_path: str,
        cfg_path: str,
        num_classes: int,
        output_dir: str,
        output_name: str = "yolov5_extracted_backbone.pt",
        verbose: bool = True
):
    """
    Extracts YOLOv5 backbone weights from the custom model checkpoint.
    """
    from LUNA16.yolov5.models.yolo import Model

    # Load YOLOv5 model
    if verbose: print(f"Loading YOLOv5 model config from {cfg_path}...")
    yolo_model = Model(cfg_path, ch=3, nc=num_classes)
    yolo_state = yolo_model.state_dict()

    # Load checkpoint
    if verbose: print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        custom_state = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        phase = checkpoint.get('phase', 'unknown')
        if verbose: print(f"  Checkpoint info: Epoch {epoch}, Phase {phase}")
    else:
        custom_state = checkpoint

    # Analyze checkpoint structure
    if verbose:
        print("\nCheckpoint structure...")
        prefixes = {}
        for key in custom_state.keys():
            prefix = key.split('.')[0]
            if prefix not in prefixes:
                prefixes[prefix] = 0
            prefixes[prefix] += 1

        print(" Modules found in checkpoint:")
        for prefix, count in prefixes.items():
            print(f"    - {prefix}: {count} parameters")

    # Create mapping from custom model to YOLOv5
    # Model uses lower_layers (0-4) and upper_layers (5-9) from YOLOv5
    mapping = {}

    yolo_layers = list(yolo_model.model.children())

    # Map lower_layers (first 5 layers of YOLOv5)
    lower_layer_mapping = {}
    upper_layer_mapping = {}

    # Extract keys that belong to the backbone
    backbone_keys = [k for k in custom_state.keys() if k.startswith('backbone.')]

    if verbose:
        print(f"\n  Backbone keys: {len(backbone_keys)}")

    # Create mapping for lower layers (0-4 in YOLOv5)
    for key in backbone_keys:
        parts = key.split('.', 2)
        if len(parts) >= 3:
            layer_idx = int(parts[1])
            param_name = parts[2]
            yolo_key = f"model.{layer_idx}.{param_name}"
            mapping[key] = yolo_key

    # Create lists for missing and mismatched keys for logging
    matched_keys = []
    mismatched_shapes = []
    missing_in_checkpoint = []

    # Create new state dict with YOLOv5 keys
    new_state_dict = OrderedDict()

    for custom_key, yolo_key in mapping.items():
        if yolo_key in yolo_state:
            custom_tensor = custom_state[custom_key]
            yolo_tensor = yolo_state[yolo_key]

            if custom_tensor.shape == yolo_tensor.shape:
                new_state_dict[yolo_key] = custom_tensor
                matched_keys.append(yolo_key)
            else:
                mismatched_shapes.append((yolo_key, custom_tensor.shape, yolo_tensor.shape))
        else:
            if verbose: print(f"YOLOv5 key not found: {yolo_key}")

    for yolo_key in yolo_state.keys():
        if yolo_key.startswith('model.') and yolo_key not in new_state_dict:
            # Extract layer number
            try:
                layer_num = int(yolo_key.split('.')[1])
                if layer_num < 10:  # Only backbone layers
                    missing_in_checkpoint.append(yolo_key)
            except:
                pass

    if verbose:
        print(f"\nSuccessfully mapped: {len(matched_keys)} parameters")
        if mismatched_shapes:
            print(f"⚠Shape mismatches: {len(mismatched_shapes)}")
            for key, custom_shape, yolo_shape in mismatched_shapes[:5]:
                print(f"    {key}: {custom_shape} vs {yolo_shape}")
        if missing_in_checkpoint:
            print(f"Missing backbone layers: {len(missing_in_checkpoint)}")
            # Only show first few
            for key in missing_in_checkpoint[:5]:
                print(f"    {key}")

    # Load the mapped weights into YOLOv5 model
    yolo_model.load_state_dict(new_state_dict, strict=False)

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    if verbose: print(f"\naving YOLOv5 model to {output_path}")
    torch.save({'model': yolo_model}, output_path)

    if verbose:
        print(f"   Loaded {len(matched_keys)}/{len(yolo_state)} total parameters")

    return output_path


def inspect_custom_checkpoint(checkpoint_path):
    """
    Helper function to inspect custom checkpoint structure
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\nTotal parameters in state_dict: {len(state_dict)}")

            # Group by module
            modules = {}
            for key in state_dict.keys():
                module = key.split('.')[0]
                if module not in modules:
                    modules[module] = []
                modules[module].append(key)

            print("\nParameter distribution:")
            for module, keys in modules.items():
                print(f"  {module}: {len(keys)} parameters")
                # Show first few keys as examples
                for key in keys[:3]:
                    print(f"    - {key} (shape: {state_dict[key].shape})")
                if len(keys) > 3:
                    print(f"    ... and {len(keys) - 3} more")


# Example usage
if __name__ == "__main__":
    checkpoint_path = "C:\\Users\\peter\\Masters\\Project\\checkpoints\\checkpoint_epoch_150_cl_full.pth"
    cfg_path = "C:\\Users\\peter\\Masters\\Project\\yolo\\yolov5\\models\\yolov5s.yaml"
    num_classes = 1
    output_dir = "converted_models"

    inspect_custom_checkpoint(checkpoint_path)
    print("=" * 60)

    # Then extract the backbone
    extracted_path = extract_yolo_backbone_from_checkpoint(
        checkpoint_path,
        cfg_path,
        num_classes,
        output_dir,
        output_name="yolov5_custom_backbone_sequential_full.pt",
        verbose=True
    )