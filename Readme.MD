# Prostate MRI Segmentation and Visualization

This project focuses on the processing and visualization of prostate magnetic resonance imaging (MRI) data.  
The objective is to extract patient MRI datasets, apply prostate gland masks, enhance the gland region, and generate visual representations that allow qualitative assessment.

---

## Objectives
- Extract patient MRI datasets from local ZIP files into a structured `data/` folder.  
- Load T2-weighted MRI volumes and corresponding prostate gland masks using **SimpleITK**.  
- Preprocess images by normalizing intensity values to 0–255 and selecting the middle slice from each 3D volume.  
- Enhance the prostate gland region using:
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** from **OpenCV** to improve local contrast.  
  - **Gamma correction** to brighten the masked region.  
- Generate comparative visualizations using **Matplotlib**. 
- Save all figures in PNG format in the `results/` folder for further review and analysis.

---

## Libraries Used
- **SimpleITK** - Loading and handling 3D medical images (`.nii.gz`).  
- **NumPy** - Array manipulation and processing.  
- **OpenCV (cv2)** - Image normalization, masking, CLAHE, gamma correction.  
- **Matplotlib** - Visualization and contour overlay plotting.  
- **zipfile** - Extracting ZIP archives.  
- **glob** - File search.  
- **os** - Directory and file path handling.

---

## Methodology 
1. **Data Extraction**:
- Patient MRI data is provided as ZIP files listed in the `zip_files`.  
- All ZIP files are extracted into a `data` folder using Python’s `zipfile` module.  
- Extracted files include T2-weighted MRI images (`_t2w.nii.gz`) and prostate masks (`_gland.nii.gz`). 

2. **File Location**:  
   - Search for T2-weighted images and gland masks using **`glob`**.  

3. **Image Loading**:  
   - Read 3D medical images (`.nii.gz`) using **SimpleITK**.  
   - Convert volumes to NumPy arrays for processing.  

4. **Slice Selection & Normalization**:  
   - Extract the middle axial slice for each patient.  
   - Normalize pixel intensities to the range 0–255 using **OpenCV**.  

5. **Masking & Enhancement**:  
   - Apply binary masking to isolate the prostate region.  
   - Enhance the masked region with **CLAHE** and **gamma correction**.  
   - Apply the mask again to clean boundaries.  

6. **Visualization**:  
   - Create a figure with three subplots using **Matplotlib**:  
     - Original slice  
     - Masked/enhanced prostate  
     - Original slice with **prostate contour overlay** using `plt.contour()`.  
   - Adjust figure layout and add titles for clarity.  

7. **Output Saving**:  
   - Save the final figure as a PNG in the `results/` folder.  
   - Repeat for all patients specified in the `patient_ids` list.

---
 
## Prostate Segmentation Results

### Patient 10005
<img src="https://github.com/minotikedare/Prostate_Segmentation/blob/main/results/10005_all_in_one.png?raw=true" alt="Patient 10005 Segmentation" width="600"/>

### Patient 10040
<img src="https://github.com/minotikedare/Prostate_Segmentation/blob/main/results/10040_all_in_one.png?raw=true" alt="Patient 10040 Segmentation" width="600"/>

### Patient 10043
<img src="https://github.com/minotikedare/Prostate_Segmentation/blob/main/results/10043_all_in_one.png?raw=true" alt="Patient 10043 Segmentation" width="600"/>

### Patient 10048
<img src="https://github.com/minotikedare/Prostate_Segmentation/blob/main/results/10048_all_in_one.png?raw=true" alt="Patient 10048 Segmentation" width="600"/>

---