# Import the necessary libraries
import os
import zipfile
import glob
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Define the download folder containing the ZIP files
downloads_folder = "/Users/minotikedare/Downloads/"
zip_files = [
  "10048-20250912T021750Z-1-001.zip",
  "10043-20250912T021748Z-1-001.zip",
  "10040-20250912T021747Z-1-001.zip",
  "10005-20250912T021745Z-1-001.zip",
]


# Define folder to extract all ZIP files
extract_folder = os.path.join(downloads_folder, "data")
os.makedirs(extract_folder, exist_ok=True)


# Extract all provided ZIP files into the extraction folder
for z in zip_files:
  with zipfile.ZipFile(os.path.join(downloads_folder, z), "r") as zip_ref:
      zip_ref.extractall(extract_folder)


print("Extracted all files into:", extract_folder)


# Create a results folder inside the current project directory
project_results_folder = os.path.join(os.getcwd(), "results")
os.makedirs(project_results_folder, exist_ok=True)


# Function to locate files with a given filename pattern inside a directory
def find_file(base_dir, pattern):
  files = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
  if not files:
      raise FileNotFoundError(f"No file found for {pattern}")
  return files[0]


# Function to apply a binary mask to an image and enhance visibility of the region
def mask_prostate(image, mask):


  mask_bin = (mask > 0).astype(np.uint8)
  masked = cv2.bitwise_and(image, image, mask=mask_bin)


  if np.any(masked > 0):
      # Apply CLAHE to improve local contrast
      clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
      enhanced = clahe.apply(masked)


      # Re-apply mask to keep only prostate region
      enhanced_masked = cv2.bitwise_and(enhanced, enhanced, mask=mask_bin)


      # Apply gamma correction to brighten the prostate region
      gamma = 0.6
      enhanced_gamma = np.power(enhanced_masked.astype(np.float64) / 255.0, gamma) * 255.0
      enhanced_gamma = enhanced_gamma.astype(np.uint8)


      # Final mask application to clean boundaries
      final_result = cv2.bitwise_and(enhanced_gamma, enhanced_gamma, mask=mask_bin)
      return final_result


  return masked


# List of patient IDs to be processed
patient_ids = ["10005", "10048", "10043", "10040"]


# Process each patient one by one
for pid in patient_ids:
  print(f"\nPatient {pid}")


  # Locate the T2-weighted image and gland mask for the patient
  t2w_file = find_file(extract_folder, f"{pid}_t2w.nii.gz")
  gland_file = find_file(extract_folder, f"{pid}_gland.nii.gz")


  # Load the 3D medical images using SimpleITK
  t2w_img = sitk.GetArrayFromImage(sitk.ReadImage(t2w_file))
  gland_img = sitk.GetArrayFromImage(sitk.ReadImage(gland_file))


  # Extract the middle slice from the 3D volume for visualization
  mid_slice = t2w_img.shape[0] // 2
  t2w_slice = t2w_img[mid_slice, :, :]
  mask_slice = gland_img[mid_slice, :, :]


  # Normalize the original T2W slice to 0–255 range for display
  t2w_normalized = cv2.normalize(t2w_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


  # Apply prostate mask with enhancement
  masked_slice = mask_prostate(t2w_normalized, mask_slice.astype(np.uint8))


  # Ensure correct orientation
  if t2w_slice.shape[1] < t2w_slice.shape[0]:
      t2w_disp = t2w_normalized.T
      masked_disp = masked_slice.T
      mask_disp = mask_slice.T
  else:
      t2w_disp = t2w_normalized
      masked_disp = masked_slice
      mask_disp = mask_slice


  # Create a figure with 3 subplots: original, masked, and contour overlay
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))


  # Display original T2W image
  axes[0].imshow(t2w_disp, cmap="gray", aspect='auto', vmin=0, vmax=255)
  axes[0].set_title("Original T2W Image", fontsize=12, fontweight='bold')
  axes[0].axis("off")


  # Display masked prostate image
  axes[1].imshow(masked_disp, cmap="gray", aspect='auto', vmin=0, vmax=255)
  axes[1].set_title("Masked Prostate", fontsize=12, fontweight='bold')
  axes[1].axis("off")


  # Display prostate contour overlay on original image
  axes[2].imshow(t2w_disp, cmap="gray", aspect='auto', vmin=0, vmax=255)
  axes[2].contour(mask_disp, levels=[0.5], colors="red", linewidths=2.0)
  axes[2].set_title("Prostate Contour Overlay", fontsize=12, fontweight='bold')
  axes[2].axis("off")


  # Add patient ID as figure title
  plt.suptitle(f"Patient {pid}", fontsize=16, fontweight='bold')
  plt.tight_layout()
  plt.subplots_adjust(top=0.85)


  # Save the final figure as a PNG image
  out_path = os.path.join(project_results_folder, f"{pid}_all_in_one.png")
  fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor='white')
  plt.close(fig)
  print(f"Image for {pid} -> {out_path}")


print("\nAll patient images saved in:", project_results_folder)