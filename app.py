import streamlit as st
import segmentationmetrics as sm
import pandas as pd
import pydicom
import numpy as np
import cv2
from difflib import get_close_matches

st.set_page_config(layout="wide")

def get_image_info(ds, rt_struct):
    zoom = [float(ds.SliceThickness), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
    depth = calculate_z_depth(rt_struct)
    img_shape = (ds.Rows, ds.Columns, depth)
    return zoom, img_shape

def calculate_z_depth(ds):
    z_coords = set()

    for roi_contour_sequence in ds.ROIContourSequence:
        for contour in roi_contour_sequence.ContourSequence:
            contour_data = contour.ContourData
            coords = np.array(contour_data).reshape(-1, 3)
            z_coords.update(coords[:, 2])
    
    return len(z_coords)

def create_mask_for_contour(contour_data, img_shape, spacing):
    mask = np.zeros(img_shape, dtype=np.uint8)
    coords = np.array(contour_data).reshape(-1, 3)#.astype(int)
    z_coords = (coords[:, 2]/spacing[0]).astype(np.int32)

    for z in np.unique(z_coords):
        if z < 0 or z >= img_shape[0]:  # Check if z is within bounds
            continue  # Skip slices that are out of bounds
        
        mask_slice = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        slice_coords = (coords[z_coords == z, :2]/spacing[1]).astype(np.int32)
        cv2.fillPoly(mask_slice, [slice_coords], 1)
        mask[:,:,z] += mask_slice

    return mask

def get_contour(ds, img_shape, spacing):
    contours = {}
    
    for roi_contour_sequence in ds.ROIContourSequence:
        roi_name = ds.StructureSetROISequence[roi_contour_sequence.ReferencedROINumber].ROIName.lower()
        mask = np.zeros(img_shape, dtype=np.uint8)

        for contour in roi_contour_sequence.ContourSequence:
            contour_data = contour.ContourData
            mask += create_mask_for_contour(contour_data, img_shape, spacing)

        contours[roi_name] = mask

    return contours


def match_contours(rois1, rois2):
    matches = {}
    for roi in rois1:
        match = get_close_matches(roi, rois2, n=1, cutoff=0.6)
        print(roi,match)
        if match:
            matches[roi] = match[0]
        else:
            matches[roi] = None
    return matches

def compare_contours(manuals, inferences, zoom):
    rois1 = sorted(list(manuals.keys()))
    rois2 = sorted(list(inferences.keys()))

    matched_rois = match_contours(rois1, rois2)
    result = pd.DataFrame()

    for roi1, roi2 in matched_rois.items():
        if roi2 is None:
            continue  # Skip if no close match was found

        mask_manual = manuals[roi1]
        mask_inference = inferences[roi2]

        if mask_manual is None or mask_inference is None:
            st.warning(f"No valid masks for ROI: {roi1} or {roi2}. Skipping this ROI.")
            continue  # Skip if either mask is None

        try:
            metrics = sm.SegmentationMetrics(mask_inference, mask_manual, zoom, symmetric=False)
            df = metrics.get_df().rename(columns={'Score': roi1})[roi1]
            result = pd.concat([result, df], axis=1)
        except KeyError as e:
            st.warning(f"Error processing ROI: {roi1}. Metric {e} not found.")
            continue
        except Exception as e:
            st.warning(f"Error processing ROI: {roi1}. Error: {str(e)}")
            continue

    return result.T, matched_rois

def main():
    st.title("RT Structure DICOM Comparison Tool")

    st.sidebar.header("Upload Files")
    image_file = st.sidebar.file_uploader("Upload One of Image DICOMs", type=["dcm"])
    manual_rtstruct_file = st.sidebar.file_uploader("Upload Manual RT Structure DICOM", type=["dcm"])
    infer_rtstruct_file = st.sidebar.file_uploader("Upload Inference RT Structure DICOM", type=["dcm"])

    if image_file and manual_rtstruct_file and infer_rtstruct_file:
        image = pydicom.dcmread(image_file)
        manual_rtstructure = pydicom.dcmread(manual_rtstruct_file)
        infer_rtstruct = pydicom.dcmread(infer_rtstruct_file)
        
        # Calculate the number of slices (z-depth) based on unique z-coordinates
        zoom, img_shape = get_image_info(image, manual_rtstruct)
        st.write(f"Calculated image shape: {img_shape}")

        # Load contours from the uploaded files
        manuals = get_contour(manual_rtstruct, img_shape, zoom)
        inferences = get_contour(infer_rtstruct, img_shape, zoom)

        # Compare the contours and display results
        result_df, matched_rois = compare_contours(manuals, inferences, zoom)

        if result_df.empty:
            st.warning("No matching contours found or the comparison could not be performed.")
        else:
            st.write("### Comparison Results")
            st.dataframe(result_df)

            # Provide a download option for the results
            csv = result_df.to_csv(index=True).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name='rt_struct_comparison.csv')

if __name__ == "__main__":
    main()
