import os
import json
import numpy as np

from process_images_and_train_eval_subset_train import run_cross_validation


def main_rerun():
    """
    A dedicated function to re-run the evaluation for a specific,
    already-trained experiment.
    """
    base_path = r"C:\Users\peter\Masters\Project\LUNA16"
    annotations_path = r"C:\Users\peter\Masters\Project\LUNA16\annotations.csv"

    experiment_name = 'custom_full_backbone_0.002lr'
    data_percentage = 100

    pretrained_weights = 'C:\\Users\peter\Masters\Project\converted_models\yolov5_custom_backbone_sequential_full.pt'

    print(f"RE-RUNNING EVALUATION FOR: {experiment_name} ({data_percentage}%)")

    # This will find the existing 'best.pt' files and skip straight to evaluation.
    results = run_cross_validation(
        base_path=base_path,
        annotations_path=annotations_path,
        num_folds=10,
        force_reload=False,
        experiment_name=experiment_name,
        pretrained_weights=pretrained_weights,
        data_percentage=data_percentage,
        seed=42
    )

    if results:
        print(f"\nRe-evaluation Summary ({data_percentage}% data)")
        print(f"Experiment: {experiment_name}")
        for fold_result in results:
            fold = fold_result['fold']
            froc = fold_result['froc_results']
            print(f"Fold {fold}: Average FROC Sensitivity = {froc['average_sensitivity']:.2%}")

        avg_sensitivity = np.mean([r['froc_results']['average_sensitivity'] for r in results])
        print(f"\nOverall Average FROC Sensitivity: {avg_sensitivity:.2%}")
        print("\nRe-evaluation complete. Check the updated JSON files for full details.")
    else:
        print("\nNo results generated. Please check that the model weight paths are correct.")


if __name__ == "__main__":
    main_rerun()