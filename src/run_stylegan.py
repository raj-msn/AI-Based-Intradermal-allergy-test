import argparse
import os
from stylegan_helpers import (
    setup_stylegan_environment, 
    prepare_dataset_for_stylegan, 
    create_stylegan_training_set, 
    train_stylegan, 
    generate_stylegan_images
)

def main(args):
    """Main script to run the StyleGAN2-ADA workflow."""
    
    # Always set up the environment first
    setup_stylegan_environment()

    if args.action == 'prepare':
        print("Preparing dataset for StyleGAN...")
        if not args.input_dir or not args.output_dir:
            print("Error: --input-dir and --output-dir are required for 'prepare' action.")
            return
        prepare_dataset_for_stylegan(args.input_dir, args.output_dir)
        print("Dataset preparation complete.")

    elif args.action == 'create_dataset':
        print("Creating StyleGAN training set (.zip)...")
        if not args.input_dir or not args.zip_path:
            print("Error: --input-dir and --zip-path are required for 'create_dataset' action.")
            return
        create_stylegan_training_set(args.input_dir, args.zip_path)
        print("Training set creation complete.")
    
    elif args.action == 'train':
        print("Training StyleGAN2-ADA...")
        if not args.zip_path or not args.output_dir:
            print("Error: --zip-path and --output-dir are required for 'train' action.")
            return
        train_stylegan(args.zip_path, args.output_dir, resume_from=args.resume_from)
        print("StyleGAN training complete.")
        
    elif args.action == 'generate':
        print("Generating images with StyleGAN2-ADA...")
        if not args.weights or not args.output_dir:
            print("Error: --weights and --output-dir are required for 'generate' action.")
            return
        generate_stylegan_images(args.weights, args.output_dir, num_images=args.num_images)
        print(f"Image generation complete. Images saved to {args.output_dir}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the StyleGAN2-ADA data augmentation workflow.")
    parser.add_argument('action', type=str, 
                        choices=['prepare', 'create_dataset', 'train', 'generate'],
                        help="The StyleGAN action to perform.")
    
    # --- Arguments for data preparation ---
    parser.add_argument('--input-dir', type=str,
                        help="[prepare, create_dataset] Path to the directory of raw images.")
    parser.add_argument('--zip-path', type=str,
                        help="[create_dataset, train] Path to save or load the dataset .zip file.")

    # --- Arguments for training ---
    parser.add_argument('--resume-from', type=str, default='ffhq1024',
                        help="[train] Path to a network pickle to resume training from, or an official alias (e.g., 'ffhq1024').")

    # --- Arguments for generation ---
    parser.add_argument('--weights', type=str,
                        help="[generate] Path to the trained network pickle (.pkl) for image generation.")
    parser.add_argument('--num-images', type=int, default=100,
                        help="[generate] Number of images to generate.")
    
    # --- General arguments ---
    parser.add_argument('--output-dir', type=str, default='stylegan_results',
                        help="[prepare, train, generate] Path to the directory to save outputs.")

    args = parser.parse_args()
    main(args) 