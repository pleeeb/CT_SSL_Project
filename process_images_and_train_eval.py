import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import json
import logging
from tqdm import tqdm
import yaml
import shutil
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LUNA16Preprocessor:
    """Preprocessor for LUNA16 CT scans"""

    # Window sizes set to preferred size for chest CT scans
    def __init__(self, target_spacing=(1.0, 1.0, 1.0),
                 window_center=-600, window_width=1500):
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.window_min = window_center - window_width / 2
        self.window_max = window_center + window_width / 2

    def load_itk_image(self, filename):
        """Load the ITK image from .mhd file"""
        try:
            itkimage = sitk.ReadImage(filename)
            ct_scan = sitk.GetArrayFromImage(itkimage)
            origin = np.array(list(reversed(itkimage.GetOrigin())))
            spacing = np.array(list(reversed(itkimage.GetSpacing())))

            logger.debug(f"Loaded scan shape: {ct_scan.shape}, spacing: {spacing}, origin: {origin}")

            return ct_scan, origin, spacing
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise

    def resample_scan(self, scan, spacing, new_spacing=(1.0, 1.0, 1.0)):
        """Resample scan to new spacing, used spacing of 1 as per standard in chest CTs"""
        # Convert to numpy arrays to ensure proper operations
        spacing = np.array(spacing)
        new_spacing = np.array(new_spacing)

        resize_factor = spacing / new_spacing
        new_real_shape = scan.shape * resize_factor
        new_shape = np.round(new_real_shape).astype(int)

        # Ensure new shape is valid
        new_shape = np.maximum(new_shape, 1)

        # Use scipy for 3D resampling
        from scipy.ndimage import zoom

        # Calculate zoom factors for each dimension
        zoom_factors = new_shape / scan.shape

        # Resample the scan
        scan_resampled = zoom(scan, zoom_factors, order=1)  # bilinear interpolation

        # Calculate actual new spacing
        real_resize_factor = np.array(scan_resampled.shape) / np.array(scan.shape)
        new_spacing = spacing / real_resize_factor

        return scan_resampled, new_spacing

    def apply_windowing(self, scan):
        """Apply windowing to CT scan"""
        scan = np.clip(scan, self.window_min, self.window_max)
        return scan

    def normalize_scan(self, scan):
        """Normalize scan to 0-255 range"""
        scan = (scan - self.window_min) / (self.window_max - self.window_min)
        scan = (scan * 255).astype(np.uint8)
        return scan

    def world_to_voxel_coord(self, world_coord, origin, spacing):
        """Convert world coordinates to voxel coordinates using the new origin and spacing"""
        voxel_coord = (world_coord - origin) / spacing
        return voxel_coord.astype(int)

    def preprocess_scan(self, filename):
        """Complete preprocessing pipeline applying windowing, resampling, normalization"""
        # Load scan
        scan, origin, spacing = self.load_itk_image(filename)

        # Resample to 1mm spacing
        scan_resampled, new_spacing = self.resample_scan(scan, spacing, self.target_spacing)

        # Apply windowing
        scan_windowed = self.apply_windowing(scan_resampled)

        # Normalize
        scan_normalized = self.normalize_scan(scan_windowed)

        return scan_normalized, origin, spacing, new_spacing


class LUNA16YOLODataset:
    """Dataset class for LUNA16 in YOLO format"""

    def __init__(self, base_path, annotations_path, subset_ids=None):
        self.base_path = base_path
        self.annotations_df = pd.read_csv(annotations_path)
        self.subset_ids = subset_ids if subset_ids else list(range(10))
        self.preprocessor = LUNA16Preprocessor()

    # This function extracts slices containing nodules from the scan and returns them in a format suitable for YOLO training.
    def get_nodule_slices(self, scan, nodule_coords, spacing, origin, diameter_mm):
        """Extract slices containing nodules with bounding boxes in YOLO format to use for training and testing"""
        nodule_slices = []

        # Convert world coordinates to voxel coordinates
        voxel_coord = self.preprocessor.world_to_voxel_coord(
            np.array([nodule_coords[2], nodule_coords[1], nodule_coords[0]]),
            origin, spacing
        )

        # Calculate radius in voxels for each dimension
        radius_voxels = (diameter_mm / 2.0) / spacing

        # Ensure coordinates are within the scan bounds so that nodules are valid and can be used
        voxel_coord = np.clip(voxel_coord, [0, 0, 0],
                              [scan.shape[0] - 1, scan.shape[1] - 1, scan.shape[2] - 1])

        # Get slice range
        z_min = max(0, int(voxel_coord[0] - radius_voxels[0]))
        z_max = min(scan.shape[0], int(voxel_coord[0] + radius_voxels[0] + 1))

        for z in range(z_min, z_max):
            # Calculate 2D bounding box for this slice
            x_center = voxel_coord[2]
            y_center = voxel_coord[1]

            # Project 3D nodule to 2D slice
            distance_from_center = abs(z - voxel_coord[0])
            if distance_from_center <= radius_voxels[0]:
                # Calculate radius at this slice (circular cross-section)
                slice_radius = np.sqrt(max(0, radius_voxels[0] ** 2 - distance_from_center ** 2))

                # Use the maximum of x and y radius for 2D projection
                radius_2d = max(radius_voxels[1], radius_voxels[2]) * (slice_radius / radius_voxels[0])

                # Calculate bounding box
                x_min = max(0, x_center - radius_2d)
                y_min = max(0, y_center - radius_2d)
                x_max = min(scan.shape[2], x_center + radius_2d)
                y_max = min(scan.shape[1], y_center + radius_2d)

                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center_norm = (x_min + x_max) / 2.0 / scan.shape[2]
                y_center_norm = (y_min + y_max) / 2.0 / scan.shape[1]
                width_norm = (x_max - x_min) / scan.shape[2]
                height_norm = (y_max - y_min) / scan.shape[1]

                # Only add if bounding box is valid to ensure slices with no nodules or out of bound nodules do not skew training
                if width_norm > 0 and height_norm > 0:
                    nodule_slices.append({
                        'slice_idx': z,
                        'yolo_bbox': [x_center_norm, y_center_norm, width_norm, height_norm],
                        'slice_data': scan[z, :, :]
                    })

        return nodule_slices

    # This function retrieves random slices from the scan that do not contain nodules. This is to balance
    # the dataset and ensure that the model does not overfit to nodules and predict that every scan has a nodule.
    def get_negative_slices(self, scan, num_slices=10):
        """Get random slices without nodules"""
        negative_slices = []
        total_slices = scan.shape[0]

        # Skip top and bottom 10% of slices (usually no nodules there and the CT images look a lot different at these points)
        start_slice = int(total_slices * 0.1)
        end_slice = int(total_slices * 0.9)
        valid_range = end_slice - start_slice

        # Sample fewer negative slices for better balance as nodules are a lot rarer in this dataset
        # compared to slices without nodules
        num_to_sample = min(num_slices, valid_range)
        slice_indices = np.random.choice(
            range(start_slice, end_slice),
            num_to_sample,
            replace=False
        )

        for idx in slice_indices:
            negative_slices.append({
                'slice_idx': idx,
                'yolo_bbox': [],
                'slice_data': scan[idx, :, :]
            })

        return negative_slices

    # This function prepares the dataset in YOLO format, extracting images and labels from the scans.
    # This is only used for the first run to reduce computation time and avoid reprocessing the dataset.
    # Can be forced to reload the dataset by setting force_reload=True.
    def prepare_yolo_dataset(self, subset_ids, output_dir, split_name='train', force_reload=False):
        """Prepare dataset in YOLO format"""
        import gc

        # Create output directories
        images_dir = os.path.join(output_dir, 'images', split_name)
        labels_dir = os.path.join(output_dir, 'labels', split_name)

        # Check if already processed
        manifest_path = os.path.join(output_dir, f'{split_name}_manifest.json')
        if os.path.exists(manifest_path) and not force_reload:
            logger.info(f"Loading existing {split_name} dataset manifest")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            logger.info(f"Found existing {split_name} dataset with {len(manifest['image_paths'])} images")
            return images_dir, labels_dir, manifest

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        image_paths = []
        image_metadata = []

        for subset_id in tqdm(subset_ids, desc=f"Processing {split_name} subsets"):
            subset_path = os.path.join(self.base_path, f'subset{subset_id}')

            # Get all .mhd files in subset
            mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]

            for mhd_file in tqdm(mhd_files, desc=f"Processing subset{subset_id}", leave=False):
                # Extract the series uid from the filename
                series_uid = mhd_file.replace('.mhd', '')
                file_path = os.path.join(subset_path, mhd_file)

                try:
                    # Preprocess scan
                    scan, origin, original_spacing, new_spacing = self.preprocessor.preprocess_scan(file_path)

                    # Get annotations for this scan
                    scan_annotations = self.annotations_df[self.annotations_df['seriesuid'] == series_uid]

                    if len(scan_annotations) > 0:
                        # Process positive slices (with nodules)
                        for _, row in scan_annotations.iterrows():
                            nodule_coords = [row['coordX'], row['coordY'], row['coordZ']]
                            diameter_mm = row['diameter_mm']

                            nodule_slices = self.get_nodule_slices(
                                scan, nodule_coords, new_spacing, origin, diameter_mm
                            )

                            for slice_info in nodule_slices:
                                # Save image
                                img_name = f"{series_uid}_slice_{slice_info['slice_idx']}_pos.png"
                                img_path = os.path.join(images_dir, img_name)

                                # Convert to 3-channel for YOLO
                                slice_3ch = cv2.cvtColor(slice_info['slice_data'], cv2.COLOR_GRAY2RGB)
                                cv2.imwrite(img_path, slice_3ch)

                                # Save label
                                label_name = img_name.replace('.png', '.txt')
                                label_path = os.path.join(labels_dir, label_name)

                                with open(label_path, 'w') as f:
                                    # Class 0 (nodule), followed by normalized bbox
                                    bbox = slice_info['yolo_bbox']
                                    f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

                                image_paths.append(img_path)
                                image_metadata.append({
                                    'image_path': img_path,
                                    'label_path': label_path,
                                    'series_uid': str(series_uid),
                                    'slice_idx': int(slice_info['slice_idx']),
                                    'subset_id': int(subset_id),
                                    'has_nodule': True,
                                    'bbox': bbox,
                                    'diameter_mm': diameter_mm
                                })

                    # Add negative slices (without nodules)
                    negative_slices = self.get_negative_slices(scan, num_slices=5)
                    for slice_info in negative_slices:
                        img_name = f"{series_uid}_slice_{slice_info['slice_idx']}_neg.png"
                        img_path = os.path.join(images_dir, img_name)

                        # Convert to 3-channel
                        slice_3ch = cv2.cvtColor(slice_info['slice_data'], cv2.COLOR_GRAY2RGB)
                        cv2.imwrite(img_path, slice_3ch)

                        # Create empty label file
                        label_name = img_name.replace('.png', '.txt')
                        label_path = os.path.join(labels_dir, label_name)
                        open(label_path, 'w').close()  # Empty file for negative samples

                        image_paths.append(img_path)
                        image_metadata.append({
                            'image_path': img_path,
                            'label_path': label_path,
                            'series_uid': str(series_uid),
                            'slice_idx': int(slice_info['slice_idx']),
                            'subset_id': int(subset_id),
                            'has_nodule': False,
                            'bbox': None,
                            'diameter_mm': None
                        })

                    # Clear scan from memory otherwise out of memory errors can occur after processing many scans
                    del scan
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

        # Save manifest
        manifest = {
            'split_name': split_name,
            'subset_ids': subset_ids,
            'image_paths': image_paths,
            'image_metadata': image_metadata,
            'total_images': len(image_paths),
            'positive_images': sum(1 for m in image_metadata if m['has_nodule']),
            'negative_images': sum(1 for m in image_metadata if not m['has_nodule'])
        }

        tmp_manifest_path = manifest_path + '.tmp'
        with open(tmp_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest_path, manifest_path)

        logger.info(f"Created {split_name} dataset with {len(image_paths)} images")
        logger.info(f"Saved manifest to {manifest_path}")

        # Clear memory
        gc.collect()

        return images_dir, labels_dir, manifest


# This function creates the YOLO dataset configuration file in YAML format for each dataset split.
def create_yolo_yaml(data_dir, num_classes=1):
    """Create YOLO dataset configuration file"""
    yaml_content = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': num_classes,
        'names': ['nodule']
    }

    yaml_path = os.path.join(data_dir, 'luna16.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    return yaml_path

# This function trains the YOLOv5 model using the provided dataset configuration and parameters.
# Pretrained weights can be specified to fine-tune the model, if empty string is passed, the model will be trained from scratch.
def train_yolov5(data_yaml, fold_num, output_dir, epochs=50, batch_size=16, img_size=512,
                 pretrained_weights='yolov5s.pt'):
    """Train YOLOv5s model"""
    import subprocess
    import gc
    import torch

    # Clone YOLOv5 if not exists, gets the base yolov5 model and the training script
    yolov5_dir = "C:\\Users\peter\Masters\Project\LUNA16\yolov5"
    if not os.path.exists(yolov5_dir):
        logger.info("Cloning YOLOv5 repository...")
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', yolov5_dir])

    # Training command, has to be ran outside of a notebook to allow for multiple workers to be used at once.
    project_dir = os.path.join(output_dir, f'fold_{fold_num}')
    cmd = [
        'python', os.path.join(yolov5_dir, 'train.py'),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', data_yaml,
        '--weights', pretrained_weights,
        '--cfg', os.path.join(yolov5_dir, 'models', 'yolov5s.yaml'),
        '--hyp', 'C:\\Users\peter\Masters\Project\hyp.finetune-luna-16.yaml',
        '--project', project_dir,
        '--name', 'luna16_nodule_detection',
        '--save-period', '10',
        '--patience', '30',
        '--multi-scale',
        '--sync-bn',
        '--workers', '8',
        '--exist-ok',
    ]

    logger.info(f"Training fold {fold_num} with command: {' '.join(cmd)}")
    logger.info(f"Using pretrained weights: {pretrained_weights}")

    # Run training
    subprocess.run(cmd)

    # Clear memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Cleared GPU memory after training")

    # Return path to best weights
    best_weights = os.path.join(project_dir, 'luna16_nodule_detection', 'weights', 'best.pt')
    return best_weights


# This function runs predictions on the dataset using the trained YOLOv5 model.
# Checkpoint support is added to allow for resuming predictions in case of interruptions.
# Interruptions can happen due to memory issues or other errors, so this function saves the state after each batch.
def predict_with_checkpoint(weights_path, images_dir, output_dir, checkpoint_file=None,
                            batch_size=16, img_size=512, conf_thres=0.001):
    """
    Run predictions with checkpoint support for recovery
    """
    import subprocess
    from pathlib import Path

    yolov5_dir = os.path.join(os.path.dirname(output_dir), 'yolov5')
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Load or create checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Resuming from checkpoint: {checkpoint['processed_images']} images already processed")
    else:
        checkpoint = {
            'processed_images': 0,
            'total_images': 0,
            'predictions': [],
            'failed_images': []
        }

    # Get list of images to process
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    checkpoint['total_images'] = len(image_files)

    # Skip already processed images
    remaining_images = image_files[checkpoint['processed_images']:]

    if len(remaining_images) == 0:
        logger.info("All images already processed")
        return checkpoint

    # Process in batches to avoid memory issues
    for i in range(0, len(remaining_images), batch_size):
        batch_images = remaining_images[i:i + batch_size]
        batch_paths = [os.path.join(images_dir, img) for img in batch_images]

        try:
            # Create temporary file with batch paths
            batch_file = os.path.join(predictions_dir, f'batch_{i}.txt')
            with open(batch_file, 'w') as f:
                for path in batch_paths:
                    f.write(f"{path}\n")

            # Run detection on batch
            detect_output = os.path.join(predictions_dir, f'batch_{i}_results')
            cmd = [
                'python', os.path.join("C:\\Users\peter\Masters\Project\LUNA16\yolov5", 'detect.py'),
                '--weights', weights_path,
                '--img', str(img_size),
                '--conf', str(conf_thres),
                '--source', batch_file,
                '--save-txt',
                '--save-conf',
                '--nosave',  # Don't save images to save space
                '--project', detect_output,
                '--name', 'detect',
                '--exist-ok'
            ]

            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(remaining_images) + batch_size - 1) // batch_size}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Detection failed: {result.stderr}")
                checkpoint['failed_images'].extend(batch_images)
            else:
                # Parse results
                labels_dir = os.path.join(detect_output, 'detect', 'labels')
                if os.path.exists(labels_dir):
                    for img_name in batch_images:
                        label_file = os.path.join(labels_dir, img_name.replace('.png', '.txt'))
                        predictions = []

                        if os.path.exists(label_file):
                            with open(label_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 6:  # class, x, y, w, h, conf
                                        predictions.append({
                                            'class': int(parts[0]),
                                            'x_center': float(parts[1]),
                                            'y_center': float(parts[2]),
                                            'width': float(parts[3]),
                                            'height': float(parts[4]),
                                            'confidence': float(parts[5])
                                        })

                        checkpoint['predictions'].append({
                            'image_name': img_name,
                            'predictions': predictions
                        })

                checkpoint['processed_images'] += len(batch_images)

            # Clean up batch files
            os.remove(batch_file)
            if os.path.exists(detect_output):
                shutil.rmtree(detect_output)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            checkpoint['failed_images'].extend(batch_images)

        # Save checkpoint after each batch
        if checkpoint_file:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(
                f"Checkpoint saved: {checkpoint['processed_images']}/{checkpoint['total_images']} images processed")

    return checkpoint


# This function uses the standard froc calculations to evaluate the YOLOv5 model predictions on the LUNA16 dataset.
def evaluate_yolov5(weights_path, data_yaml, fold_num, output_dir, manifest, img_size=512):
    """Evaluate YOLOv5 model with checkpoint support"""
    import subprocess

    yolov5_dir = os.path.join(os.path.dirname(output_dir), 'yolov5')
    project_dir = os.path.join(output_dir, f'fold_{fold_num}_eval')
    os.makedirs(project_dir, exist_ok=True)

    # Check if predictions already exist
    predictions_checkpoint = os.path.join(project_dir, 'predictions_checkpoint.json')

    # Load data yaml to get validation images directory
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    val_images_dir = os.path.join(data_config['path'], data_config['val'])

    # Run predictions with checkpoint support
    checkpoint = predict_with_checkpoint(
        weights_path,
        val_images_dir,
        project_dir,
        predictions_checkpoint,
        batch_size=16,
        img_size=img_size,
        conf_thres=0.001
    )

    # Save final predictions
    final_predictions_path = os.path.join(project_dir, 'final_predictions.json')
    with open(final_predictions_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    # Calculate FROC metrics using saved predictions and manifest
    froc_results = calculate_froc_metrics(checkpoint, manifest)

    # Save evaluation results
    eval_results = {
        'fold': fold_num,
        'weights_path': weights_path,
        'total_images': checkpoint['total_images'],
        'processed_images': checkpoint['processed_images'],
        'failed_images': len(checkpoint['failed_images']),
        'froc_results': froc_results,
        'manifest': manifest
    }

    eval_results_path = os.path.join(project_dir, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    return froc_results

# Helper function to calculate FROC metrics from predictions and manifest.
def calculate_froc_metrics(predictions_checkpoint, manifest):
    """Calculate FROC sensitivity at different FP rates"""
    from collections import defaultdict

    # FP rates to evaluate
    fp_per_scan_levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    # Group predictions by scan (series_uid)
    scan_predictions = defaultdict(list)
    scan_ground_truths = defaultdict(list)

    # Create mapping from image name to metadata
    image_to_metadata = {
        os.path.basename(m['image_path']): m
        for m in manifest['image_metadata']
    }

    # Process predictions
    for pred_item in predictions_checkpoint['predictions']:
        img_name = pred_item['image_name']
        if img_name in image_to_metadata:
            metadata = image_to_metadata[img_name]
            series_uid = metadata['series_uid']

            # Add predictions for this scan
            for pred in pred_item['predictions']:
                scan_predictions[series_uid].append({
                    'confidence': pred['confidence'],
                    'bbox': [pred['x_center'], pred['y_center'],
                             pred['width'], pred['height']],
                    'slice_idx': metadata['slice_idx']
                })

    # Process ground truths from manifest
    for metadata in manifest['image_metadata']:
        if metadata['has_nodule']:
            series_uid = metadata['series_uid']
            scan_ground_truths[series_uid].append({
                'bbox': metadata['bbox'],
                'slice_idx': metadata['slice_idx'],
                'diameter_mm': metadata['diameter_mm']
            })

    # Calculate total number of scans
    all_scans = set(scan_predictions.keys()) | set(scan_ground_truths.keys())
    num_scans = len(all_scans)

    # Sort all predictions by confidence (descending)
    all_predictions = []
    for series_uid, preds in scan_predictions.items():
        for pred in preds:
            all_predictions.append({
                'series_uid': series_uid,
                'confidence': pred['confidence'],
                'bbox': pred['bbox'],
                'slice_idx': pred['slice_idx']
            })

    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    # Calculate sensitivities at different FP rates
    sensitivities = []

    for fp_rate in fp_per_scan_levels:
        # Calculate confidence threshold for this FP rate
        max_fps = int(fp_rate * num_scans)

        # Find threshold that gives approximately this FP rate
        fps_seen = 0
        threshold = 0.0
        fps_per_scan = defaultdict(int)

        for pred in all_predictions:
            # Check if this is a false positive (simplified check)
            is_fp = True  # Assume FP unless matched with GT

            if pred['series_uid'] in scan_ground_truths:
                # Check if prediction matches any ground truth in same slice
                for gt in scan_ground_truths[pred['series_uid']]:
                    if gt['slice_idx'] == pred['slice_idx']:
                        # Simple IoU check (you may want to improve this)
                        is_fp = False
                        break

            if is_fp:
                fps_per_scan[pred['series_uid']] += 1
                fps_seen = sum(fps_per_scan.values())

                if fps_seen >= max_fps:
                    threshold = pred['confidence']
                    break

        # Calculate sensitivity at this threshold
        true_positives = 0
        total_nodules = sum(len(gts) for gts in scan_ground_truths.values())

        matched_gts = set()
        for pred in all_predictions:
            if pred['confidence'] < threshold:
                break

            gt_list = scan_ground_truths.get(pred['series_uid'], [])
            for i, gt in enumerate(gt_list):
                gt_key = (pred['series_uid'], gt['slice_idx'], i)
                if pred['slice_idx'] == gt['slice_idx'] and gt_key not in matched_gts:
                    matched_gts.add(gt_key)
                    true_positives += 1
                    break

        sensitivity = true_positives / max(total_nodules, 1)
        sensitivities.append(sensitivity)

    # Calculate average sensitivity
    avg_sensitivity = np.mean(sensitivities)

    # Prepare FROC metrics dictionary
    froc_metrics = {
        'sensitivity_at_0.125_fp': sensitivities[0] if len(sensitivities) > 0 else 0.0,
        'sensitivity_at_0.25_fp': sensitivities[1] if len(sensitivities) > 1 else 0.0,
        'sensitivity_at_0.5_fp': sensitivities[2] if len(sensitivities) > 2 else 0.0,
        'sensitivity_at_1_fp': sensitivities[3] if len(sensitivities) > 3 else 0.0,
        'sensitivity_at_2_fp': sensitivities[4] if len(sensitivities) > 4 else 0.0,
        'sensitivity_at_4_fp': sensitivities[5] if len(sensitivities) > 5 else 0.0,
        'sensitivity_at_8_fp': sensitivities[6] if len(sensitivities) > 6 else 0.0,
        'average_sensitivity': avg_sensitivity,
        'total_scans': num_scans,
        'total_nodules': total_nodules,
        'total_predictions': len(all_predictions)
    }

    # Log results for easy viewing
    logger.info(f"FROC Results: Average sensitivity = {avg_sensitivity:.2%}")
    return froc_metrics

# This function runs the 10-fold cross-validation for the LUNA16 dataset using YOLOv5.
# Each fold is held out to be used for testing whilst the other 9 are used to train the model.
# This is standard for the LUNA16 dataset to ensure that the model is robust and generalizes well.
def run_cross_validation(base_path, annotations_path, num_folds=10, force_reload=False,
                         experiment_name="default", pretrained_weights='yolov5s.pt'):
    """Run 10-fold cross-validation with YOLOv5"""
    import gc
    import torch

    # Initialize dataset
    dataset = LUNA16YOLODataset(base_path, annotations_path)

    # Create output directory with experiment name
    if experiment_name == "default":
        output_dir = os.path.join(base_path, 'yolov5_output')
    else:
        output_dir = os.path.join(base_path, f'yolov5_output_{experiment_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Results storage
    all_results = []

    # Save cross-validation state
    cv_state_file = os.path.join(output_dir, 'cv_state.json')
    if os.path.exists(cv_state_file) and not force_reload:
        with open(cv_state_file, 'r') as f:
            cv_state = json.load(f)
        logger.info(f"Resuming cross-validation from fold {cv_state['last_completed_fold'] + 1}")
    else:
        cv_state = {
            'last_completed_fold': -1,
            'completed_folds': [],
            'results': [],
            'experiment_name': experiment_name,
            'pretrained_weights': pretrained_weights
        }

    # 10-fold cross-validation (leave-one-subset-out)
    for test_subset in range(num_folds):
        # Skip if already completed
        if test_subset <= cv_state['last_completed_fold'] and not force_reload:
            logger.info(f"Skipping completed fold {test_subset}")
            # Load existing results
            fold_results_file = os.path.join(output_dir, f'fold_{test_subset}_complete.json')
            if os.path.exists(fold_results_file):
                with open(fold_results_file, 'r') as f:
                    fold_results = json.load(f)
                all_results.append(fold_results)
            continue

        logger.info(f"\n--- Fold {test_subset + 1}/{num_folds} ---")
        logger.info(f"Test subset: subset{test_subset}")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Pretrained weights: {pretrained_weights}")

        fold_dir = os.path.join(base_path, f'yolov5_output', f'fold_{test_subset}_data')
        data_yaml = os.path.join(fold_dir, 'luna16.yaml')
        train_manifest_file = os.path.join(fold_dir, 'train_manifest.json')
        val_manifest_file = os.path.join(fold_dir, 'val_manifest.json')

        if not os.path.exists(data_yaml):
            logger.error(f"Missing luna16.yaml at {data_yaml}")
            continue

            # Load manifests
        with open(train_manifest_file, 'r') as f:
            train_manifest = json.load(f)
        with open(val_manifest_file, 'r') as f:
            val_manifest = json.load(f)

        # Define train and test subsets
        # Only for the first run, we prepare the datasets as the experiments all use the same splits for efficiency and
        # reproducibility.
        #train_subsets = [i for i in range(num_folds) if i != test_subset]
        #test_subsets = [test_subset]

        # Uncomment the following lines if you want to prepare datasets for the first run

        # Create fold directory
        #fold_dir = os.path.join(output_dir, f'fold_{test_subset}_data')

        # Prepare datasets
        #logger.info("Preparing training dataset...")
        #train_images_dir, train_labels_dir, train_manifest = dataset.prepare_yolo_dataset(
        #    train_subsets, fold_dir, 'train', force_reload
        #)

        #logger.info("Preparing validation dataset...")
        #val_images_dir, val_labels_dir, val_manifest = dataset.prepare_yolo_dataset(
        #    test_subsets, fold_dir, 'val', force_reload
        #)

        # Create YOLO yaml configuration
        #data_yaml = create_yolo_yaml(fold_dir)

        # Check if model already exists
        weights_path = os.path.join(output_dir, f'fold_{test_subset}',
                                    'luna16_nodule_detection', 'weights', 'best.pt')

        if not os.path.exists(weights_path) or force_reload:
            # Train model
            logger.info("Training YOLOv5s model...")
            weights_path = train_yolov5(
                data_yaml, test_subset, output_dir,
                epochs=50, batch_size=16, img_size=512,
                pretrained_weights=pretrained_weights
            )
        else:
            logger.info(f"Using existing model: {weights_path}")

        # Backup model
        backup_dir = os.path.join(output_dir, 'model_backups', f'fold_{test_subset}')
        os.makedirs(backup_dir, exist_ok=True)
        backup_weights = os.path.join(backup_dir, 'best.pt')
        if os.path.exists(weights_path) and not os.path.exists(backup_weights):
            shutil.copy2(weights_path, backup_weights)
            logger.info(f"Backed up model to {backup_weights}")

        # Evaluate model
        logger.info("Evaluating model...")
        try:
            froc_results = evaluate_yolov5(weights_path, data_yaml, test_subset, output_dir, val_manifest)

            # Save fold results
            fold_results = {
                'fold': test_subset,
                'froc_results': froc_results,
                'weights_path': weights_path,
                'backup_weights': backup_weights,
                'experiment_name': experiment_name,
                'pretrained_weights': pretrained_weights,
                'train_manifest': {
                    'total_images': train_manifest['total_images'],
                    'positive_images': train_manifest['positive_images'],
                    'negative_images': train_manifest['negative_images']
                },
                'val_manifest': {
                    'total_images': val_manifest['total_images'],
                    'positive_images': val_manifest['positive_images'],
                    'negative_images': val_manifest['negative_images']
                }
            }

            all_results.append(fold_results)

            # Save completed fold results
            fold_results_file = os.path.join(output_dir, f'fold_{test_subset}_complete.json')
            with open(fold_results_file, 'w') as f:
                json.dump(fold_results, f, indent=2)

            # Update CV state
            cv_state['last_completed_fold'] = test_subset
            cv_state['completed_folds'].append(test_subset)
            cv_state['results'].append(fold_results)

            with open(cv_state_file, 'w') as f:
                json.dump(cv_state, f, indent=2)

        except Exception as e:
            logger.error(f"Error evaluating fold {test_subset}: {str(e)}")
            # Save error state
            error_file = os.path.join(output_dir, f'fold_{test_subset}_error.json')
            with open(error_file, 'w') as f:
                json.dump({'error': str(e), 'fold': test_subset}, f, indent=2)

        # Clear memory after each fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared memory after fold completion")

    # Calculate average FROC sensitivity
    if all_results:
        avg_sensitivity = np.mean([r['froc_results']['average_sensitivity'] for r in all_results])
        logger.info(f"\nAverage FROC sensitivity across all folds: {avg_sensitivity:.2%}")

    return all_results


def resume_predictions(base_path, fold_num, experiment_name="default"):
    """Resume predictions for a specific fold if interrupted"""
    if experiment_name == "default":
        output_dir = os.path.join(base_path, 'yolov5_output')
    else:
        output_dir = os.path.join(base_path, f'yolov5_output_{experiment_name}')

    # Load fold data
    fold_dir = os.path.join(output_dir, f'fold_{fold_num}_data')
    data_yaml = os.path.join(fold_dir, 'luna16.yaml')

    # Load validation manifest
    val_manifest_path = os.path.join(fold_dir, 'val_manifest.json')
    with open(val_manifest_path, 'r') as f:
        val_manifest = json.load(f)

    # Get weights path
    weights_path = os.path.join(output_dir, f'fold_{fold_num}',
                                'luna16_nodule_detection', 'weights', 'best.pt')

    # Check for backup weights if main weights missing
    if not os.path.exists(weights_path):
        backup_weights = os.path.join(output_dir, 'model_backups', f'fold_{fold_num}', 'best.pt')
        if os.path.exists(backup_weights):
            logger.info(f"Restoring model from backup: {backup_weights}")
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            shutil.copy2(backup_weights, weights_path)
        else:
            raise FileNotFoundError(f"No model weights found for fold {fold_num}")

    # Resume evaluation
    logger.info(f"Resuming predictions for fold {fold_num}")
    froc_results = evaluate_yolov5(weights_path, data_yaml, fold_num, output_dir, val_manifest)

    return froc_results


def main():
    # Paths
    base_path = r"C:\Users\peter\Masters\Project\LUNA16"
    annotations_path = r"C:\Users\peter\Masters\Project\LUNA16\annotations.csv"

    # Set parameters
    force_reload = False
    experiment_name = 'custom_backbone_sequential_0.002lr'
    pretrained_weights = 'C:\\Users\peter\Masters\Project\converted_models\yolov5_custom_backbone_sequential.pt'

    # Run cross-validation
    results = run_cross_validation(
        base_path,
        annotations_path,
        force_reload=force_reload,
        experiment_name=experiment_name,
        pretrained_weights=pretrained_weights
    )

    # Save overall results
    if experiment_name == "default":
        output_dir = os.path.join(base_path, 'yolov5_output')
    else:
        output_dir = os.path.join(base_path, f'yolov5_output_{experiment_name}')

    with open(os.path.join(output_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Training completed!")

    # Print summary of results
    print("\n=== Cross-Validation Results Summary ===")
    print(f"Experiment: {experiment_name}")
    print(f"Pretrained weights: {pretrained_weights}")
    print("=" * 40)

    for fold_result in results:
        fold = fold_result['fold']
        froc = fold_result['froc_results']
        print(f"Fold {fold}: Average FROC Sensitivity = {froc['average_sensitivity']:.2%}")
        print(f"  - Sensitivity @ 0.125 FP/scan: {froc['sensitivity_at_0.125_fp']:.2%}")
        print(f"  - Sensitivity @ 1 FP/scan: {froc['sensitivity_at_1_fp']:.2%}")
        print(f"  - Sensitivity @ 4 FP/scan: {froc['sensitivity_at_4_fp']:.2%}")

    # Calculate and print overall average
    if results:
        avg_sensitivity = np.mean([r['froc_results']['average_sensitivity'] for r in results])
        print(f"\nOverall Average FROC Sensitivity: {avg_sensitivity:.2%}")


if __name__ == "__main__":
    main()