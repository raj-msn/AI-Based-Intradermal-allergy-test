import os
import shutil
import subprocess
import glob
from PIL import Image

def setup_stylegan_environment():
    """Install required packages and clone StyleGAN2-ADA repository."""
    print("Setting up StyleGAN2-ADA environment...")
    subprocess.run("pip install ninja", shell=True, check=True)
    if not os.path.exists('stylegan2-ada-pytorch'):
        subprocess.run("git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git", shell=True, check=True)

def prepare_dataset_for_stylegan(input_path, output_path, size=1024):
    """Resize images and create a directory for StyleGAN training."""
    print("Preparing dataset for StyleGAN...")
    os.makedirs(output_path, exist_ok=True)
    image_files = glob.glob(os.path.join(input_path, '*.[jp][pn][g]'))
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize while maintaining aspect ratio
            w, h = img.size
            ratio = min(size/w, size/h)
            new_size = (int(w*ratio), int(h*ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
            # Create a new square image with padding
            new_img = Image.new('RGB', (size, size), (255, 255, 255))
            paste_pos = ((size - new_size[0])//2, (size - new_size[1])//2)
            new_img.paste(img, paste_pos)
            
            output_file = os.path.join(output_path, os.path.basename(img_path))
            new_img.save(output_file, 'JPEG', quality=95)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def create_stylegan_training_set(data_path, dataset_zip_path):
    """Create a .zip file dataset in the format required by StyleGAN2-ADA."""
    print("Creating StyleGAN training dataset...")
    subprocess.run([
        'python', 'stylegan2-ada-pytorch/dataset_tool.py',
        '--source', data_path,
        '--dest', dataset_zip_path,
        '--width', '1024',
        '--height', '1024'
    ], check=True)

def train_stylegan(dataset_zip_path, output_dir, resume_from='ffhq1024', gpus=1, snap=10):
    """Train StyleGAN2-ADA."""
    print("Starting StyleGAN2-ADA training...")
    subprocess.run([
        'python', 'stylegan2-ada-pytorch/train.py',
        '--outdir', output_dir,
        '--data', dataset_zip_path,
        '--gpus', str(gpus),
        '--batch', '4',
        '--gamma', '100',
        '--mirror', '1',
        '--snap', str(snap),
        '--metrics', 'none',
        '--aug', 'ada',
        '--target', '0.7',
        '--augpipe', 'bgc',
        '--cond', '0',
        '--resume', resume_from,
        '--freezed', '10'
    ], check=True)

def generate_stylegan_images(network_pkl, output_dir, num_images=100):
    """Generate new images using the trained StyleGAN2-ADA model."""
    print("Generating images with StyleGAN2-ADA...")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        'python', 'stylegan2-ada-pytorch/generate.py',
        '--outdir', output_dir,
        '--trunc', '0.7',
        '--seeds', f'0-{num_images-1}',
        '--network', network_pkl
    ], check=True) 