import streamlit as st
import segmentationmetrics as sm
import pandas as pd
import pydicom
import numpy as np
import cv2
from difflib import get_close_matches
from tqdm import tqdm
import glob
import sys
from shapely import Polygon
from shapely.geometry import MultiPolygon
from skimage.draw import polygon as draw_polygon
from datetime import datetime

st.set_page_config(page_title = "RT Structure Comparison Tool", page_icon="dgi_tab.ico",)

theme = st.get_option("theme.base")

if theme == "dark":
    st.logo('logo_light.png')
else:
    st.logo('logo_dark.png')

st.markdown(f"""
    <style>
    /* Hide the configuration menu (three dots) */
    [data-testid="stToolbar"] {{
        visibility: hidden;
        height: 0px;
    }}
    /* Add a footer */
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #333;
    }}
    </style>
    <div class="footer">
        © 2024 Wonyoung Cho. All rights reserved. |
        Contact: <a href="mailto:wycho@oncosoft.io" style="text-decoration: none; color: DodgerBlue;">
        wycho@oncosoft.io</a>
    </div>
""", unsafe_allow_html=True)

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

def create_mask_for_contour(contour_data, img_shape, spacing, affine):
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

def world_to_pixel_coordinates(world_coords, affine_matrix, slice_index):
    """Convert world coordinates to pixel coordinates"""
    # Compute the inverse of the affine matrix
    inverse_affine = np.linalg.inv(affine_matrix)

    pixel_points = []
    
    for point in world_coords:
        # Convert world coordinates to homogeneous coordinates
        homogeneous_world_coords = np.append(point, 1)

        # Apply the inverse affine matrix to get pixel coordinates
        pixel_coords_homogeneous = inverse_affine @ homogeneous_world_coords

        # Extract the pixel coordinates
        pixel_coords = pixel_coords_homogeneous[:3]
        pixel_coords = list(np.round(pixel_coords).astype(int))
        pixel_points.append(pixel_coords)

    pixel_points_t = np.array(pixel_points).T
    pixel_points_t[2] = slice_index
    pixel_points = pixel_points_t.T
    return pixel_points

def get_mask(image_shape, organ_contour_points, contour_name):
    """Create binary mask from contour points"""
    mask_volume = np.zeros(image_shape, dtype=np.uint8)
    contours_by_plane = {}
    
    for contour_points in organ_contour_points:
        z_coord = contour_points[0, 2]
        slice_index = int(round(z_coord))
        
        # Skip if slice_index is out of bounds
        if slice_index < 0 or slice_index >= image_shape[2]:
            continue
            
        x_points = contour_points[:, 0]
        y_points = contour_points[:, 1]

        # Skip if any points are out of bounds
        if np.any(x_points < 0) or np.any(x_points >= image_shape[0]) or \
           np.any(y_points < 0) or np.any(y_points >= image_shape[1]):
            continue

        coords = zip(x_points, y_points)
        polygon = Polygon(coords)
        
        if slice_index not in contours_by_plane:
            contours_by_plane[slice_index] = []
        contours_by_plane[slice_index].append(polygon)

    # Process each slice
    for slice_index, polygons in contours_by_plane.items():
        # Double-check slice index is within bounds
        if slice_index < 0 or slice_index >= image_shape[2]:
            continue
            
        slice_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
        
        outer_contours = []
        inner_contours = []

        for polygon in polygons:
            is_inner = any(outer.contains(polygon) for outer in outer_contours)
            if is_inner:
                inner_contours.append(polygon)
            else:
                outer_contours.append(polygon)

        # Draw outer contours
        for outer_polygon in outer_contours:
            if outer_polygon.is_valid:
                rr, cc = draw_polygon(*outer_polygon.exterior.coords.xy)
                # Check if polygon points are within image bounds
                valid_points = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
                rr = rr[valid_points]
                cc = cc[valid_points]
                if len(rr) > 0 and len(cc) > 0:
                    slice_mask[cc, rr] = 1
            else:
                fixed_polygon = outer_polygon.buffer(0)
                if isinstance(fixed_polygon, MultiPolygon):
                    for poly in fixed_polygon.geoms:
                        if poly.is_valid and poly.exterior and len(poly.exterior.coords) > 0:
                            rr, cc = draw_polygon(*poly.exterior.coords.xy)
                            valid_points = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
                            rr = rr[valid_points]
                            cc = cc[valid_points]
                            if len(rr) > 0 and len(cc) > 0:
                                slice_mask[cc, rr] = 1
                elif fixed_polygon.is_valid:
                    if fixed_polygon.exterior and len(fixed_polygon.exterior.coords) > 0:
                        rr, cc = draw_polygon(*fixed_polygon.exterior.coords.xy)
                        valid_points = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
                        rr = rr[valid_points]
                        cc = cc[valid_points]
                        if len(rr) > 0 and len(cc) > 0:
                            slice_mask[cc, rr] = 1

        # Subtract inner contours (holes)
        for inner_polygon in inner_contours:
            if inner_polygon.is_valid:
                rr, cc = draw_polygon(*inner_polygon.exterior.coords.xy)
                valid_points = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
                rr = rr[valid_points]
                cc = cc[valid_points]
                if len(rr) > 0 and len(cc) > 0:
                    slice_mask[cc, rr] = 0

        mask_volume[:, :, slice_index] = np.maximum(mask_volume[:, :, slice_index], slice_mask)

    mask = mask_volume.transpose(2, 0, 1)
    return mask

def get_contour(ds, img_shape, spacing, affine):
    """Enhanced get_contour function using the improved mask creation"""
    contours = {}
    sop_to_index = {}
    
    # Create SOP Instance UID to slice index mapping
    for roi_contour in ds.ROIContourSequence:
        for contour in roi_contour.ContourSequence:
            sop_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            if hasattr(contour, 'ContourImageSequence'):
                z_coord = np.array(contour.ContourData).reshape(-1, 3)[0, 2]
                sop_to_index[sop_uid] = int(round(z_coord))

    # Create a mapping of ROI numbers to ROI names
    roi_number_to_name = {}
    for roi_struct in ds.StructureSetROISequence:
        roi_number_to_name[roi_struct.ROINumber] = roi_struct.ROIName.lower()

    for roi_contour_sequence in ds.ROIContourSequence:
        # Safely get the ROI name using the mapping
        roi_number = roi_contour_sequence.ReferencedROINumber
        if roi_number not in roi_number_to_name:
            continue  # Skip if ROI number is not found
        
        roi_name = roi_number_to_name[roi_number]
        contour_points_set = []

        for contour in roi_contour_sequence.ContourSequence:
            contour_data = np.array(contour.ContourData).reshape(-1, 3)
            sop_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            
            if sop_uid in sop_to_index:
                slice_index = sop_to_index[sop_uid]
                pixel_points = world_to_pixel_coordinates(contour_data, affine, slice_index)
                contour_points_set.append(pixel_points)

        if contour_points_set:
            mask = get_mask(img_shape, contour_points_set, roi_name)
            contours[roi_name] = mask

    return contours

def match_contours(rois1, rois2):
    matches = {}
    rois2l = [word.lower() for word in rois2]
    r2d = {rois2l[i]:rois2[i] for i in range(len(rois2))}

    for roi in rois1:
        roil = roi.lower()
        match = get_close_matches(roil, rois2l, n=1, cutoff=0.6)

        if match:
            matches[roi] = r2d[match[0]]
        else:
            matches[roi] = None
    return matches

def compare_contours(manuals, inferences, zoom):
    """Enhanced comparison function combining functionality from both files"""
    rois1 = sorted(list(manuals.keys()))
    rois2 = sorted(list(inferences.keys()))

    matched_rois = match_contours(rois1, rois2)
    result_dsc = pd.DataFrame()
    result_msd = pd.DataFrame()
    result_hd95 = pd.DataFrame()

    for roi1, roi2 in matched_rois.items():
        if roi2 is None:
            continue

        mask_manual = manuals[roi1]
        mask_inference = inferences[roi2]

        if mask_manual is None or mask_inference is None:
            st.warning(f"No valid masks for ROI: {roi1} or {roi2}. Skipping this ROI.")
            continue

        try:
            metrics = sm.SegmentationMetrics(mask_inference, mask_manual, zoom, symmetric=True)
            df = metrics.get_df().rename(columns={'Score': roi1})[roi1]
            # Split metrics into separate DataFrames
            result_dsc = pd.concat([result_dsc, pd.DataFrame({'dice': df['dice']}, index=[roi1])], axis=0)
            result_msd = pd.concat([result_msd, pd.DataFrame({'mean_surface_distance': df['mean_surface_distance']}, index=[roi1])], axis=0)
            result_hd95 = pd.concat([result_hd95, pd.DataFrame({'hausdorff_distance': df['hausdorff_distance']}, index=[roi1])], axis=0)
            
        except Exception as e:
            st.warning(f"Error processing ROI: {roi1}. Error: {str(e)}")
            continue

    # Calculate averages and totals
    for df in [result_dsc, result_msd, result_hd95]:
        if not df.empty:
            df['Average'] = df.mean(axis=1)
            df.loc['Total'] = df.mean()

    return result_dsc, result_msd, result_hd95, matched_rois

def get_affine_matrix(dicom_ds):
    """
    Extract affine matrix from DICOM dataset
    Args:
        dicom_ds: pydicom dataset object
    Returns:
        4x4 affine matrix as numpy array
    """
    # Get the image position (origin)
    try:
        img_pos = np.array(dicom_ds.ImagePositionPatient, dtype=float)
    except:
        st.warning("ImagePositionPatient not found in DICOM. Using default origin.")
        img_pos = np.array([0, 0, 0])

    # Get the pixel spacing
    try:
        pix_spacing = np.array(dicom_ds.PixelSpacing, dtype=float)
        slice_thickness = float(dicom_ds.SliceThickness)
    except:
        st.warning("PixelSpacing or SliceThickness not found in DICOM. Using default spacing.")
        pix_spacing = np.array([1.0, 1.0])
        slice_thickness = 1.0

    # Get the image orientation
    try:
        img_orient = np.array(dicom_ds.ImageOrientationPatient, dtype=float)
        row_orient = img_orient[:3]
        col_orient = img_orient[3:]
    except:
        st.warning("ImageOrientationPatient not found in DICOM. Using default orientation.")
        row_orient = np.array([1, 0, 0])
        col_orient = np.array([0, 1, 0])

    # Calculate the slice orientation
    slice_orient = np.cross(row_orient, col_orient)

    # Create the affine matrix
    affine = np.zeros((4, 4))
    affine[:3, 0] = row_orient * pix_spacing[0]
    affine[:3, 1] = col_orient * pix_spacing[1]
    affine[:3, 2] = slice_orient * slice_thickness
    affine[:3, 3] = img_pos
    affine[3, 3] = 1.0

    return affine

def main():
    st.title("RT Structure Comparison Tool")

    st.sidebar.header("Upload DICOM Files")
    # Add current time display in sidebar
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #st.sidebar.write(f"Current Time: {current_time}")
    
    image_file = st.sidebar.file_uploader("Upload One of Image file", type=["dcm"])
    manual_rtstruct_file = st.sidebar.file_uploader("Upload Manual RT Structure file", type=["dcm"])
    infer_rtstruct_file = st.sidebar.file_uploader("Upload Inference RT Structure file", type=["dcm"])

    process_button = st.sidebar.button("Process Files", 
                                        disabled=not (image_file and manual_rtstruct_file and infer_rtstruct_file))

    if process_button and image_file and manual_rtstruct_file and infer_rtstruct_file:
        with st.spinner('Calculating metrics...'):
            image = pydicom.dcmread(image_file, force=True)
            affine = get_affine_matrix(image)
            manual_rtstruct = pydicom.dcmread(manual_rtstruct_file, force=True)
            infer_rtstruct = pydicom.dcmread(infer_rtstruct_file, force=True)
            
            zoom, img_shape = get_image_info(image , manual_rtstruct)
            #st.write(f"Calculated image shape: {img_shape}")

            manuals = get_contour(manual_rtstruct, img_shape, zoom, affine)
            inferences = get_contour(infer_rtstruct, img_shape, zoom, affine)

            df_dsc, df_msd, df_hd95, matched_rois = compare_contours(manuals, inferences, zoom)
        
        if df_dsc.empty:
            st.warning("No matching contours found or the comparison could not be performed.")
        else:
            # Create combined results dataframe
            combined_results = pd.DataFrame(index=df_dsc.index)
            combined_results['DSC'] = df_dsc['dice']
            combined_results['MSD (mm)'] = df_msd['mean_surface_distance']
            combined_results['HD95 (mm)'] = df_hd95['hausdorff_distance']
            
            # Format the values
            combined_results['DSC'] = combined_results['DSC'].map('{:.3f}'.format)
            combined_results['MSD (mm)'] = combined_results['MSD (mm)'].map('{:.2f}'.format)
            combined_results['HD95 (mm)'] = combined_results['HD95 (mm)'].map('{:.2f}'.format)

            # Display the combined results
            st.dataframe(combined_results, use_container_width=True)

            # Add a download button for the combined results
            csv_combined = combined_results.to_csv(index=True).encode('utf-8')
            st.download_button(
                "Download Results",
                data=csv_combined,
                file_name='result_metrics.csv',
                mime='text/csv'
            )

            # Show individual metrics if needed (optional)
            with st.expander("Download Metrics Individually"):
                # Individual download buttons
                col0, col1, col2, col3 = st.columns(4)
                with col0:
                    print(current_time)
                with col1:
                    csv_dsc = df_dsc.to_csv(index=True).encode('utf-8')
                    st.download_button("Download DSC Results", data=csv_dsc, file_name='dsc_results.csv')
                with col2:
                    csv_msd = df_msd.to_csv(index=True).encode('utf-8')
                    st.download_button("Download MSD Results", data=csv_msd, file_name='msd_results.csv')
                with col3:
                    csv_hd95 = df_hd95.to_csv(index=True).encode('utf-8')
                    st.download_button("Download HD95 Results", data=csv_hd95, file_name='hd95_results.csv')

if __name__ == "__main__":
    main()
