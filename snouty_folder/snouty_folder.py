from datetime import datetime
from pathlib import Path
from typing import Any
from ome_types import model, from_xml
import numpy as np
import os
import tifffile
try:
    from cupyx.scipy.ndimage import affine_transform as aff_trf
    import cupy as cp
    gpu = True
except ImportError:
    from scipy.ndimage import affine_transform as aff_trf
    cp = None
    gpu = False

class SnoutyFolder:
    """
    A class for processing and converting Snouty microscopy data folders to OME-TIFF format.
    
    This class handles the extraction of metadata and image data from Snouty acquisition
    folders and converts them to standardized OME-TIFF files with proper metadata. It supports
    three different output formats: original, desheared, and traditional (rotated) views.
    
    Attributes:
        dir_path (str): Path to the input Snouty data directory
        dir_name (str): Name of the input directory
        out_dir (str): Directory path for output OME-TIFF files
        
        data_path (str): Path to the data subdirectory containing TIFF files
        data_files (list): List of sorted data file paths
        
        metadata_path (str): Path to the metadata subdirectory
        metadata_file (str): Path to the primary metadata file
        metadata (dict): Parsed metadata from the metadata file
        
        channels (list): List of channel wavelengths/identifiers
        dim_resolutions (dict): Physical resolution information
        dim_order (str): Dimension order string ('TCZYX')
        max_deshear_shift (int): Maximum pixel shift needed for deshearing
        
        im_original_dims (tuple): Original image dimensions (T, C, Z, Y, X)
        im_desheared_dims (tuple): Desheared image dimensions (T, C, Z, Y, X)
        im_traditional_dims (tuple): Traditional view dimensions (T, C, Y, Z, X)
        
        im_original_path (str | None): Path to the original OME-TIFF file
        im_desheared_path (str | None): Path to the desheared OME-TIFF file
        im_traditional_path (str | None): Path to the traditional OME-TIFF file
    
    Example:
        >>> folder = SnoutyFolder("data/2025-07-11_17-43-38_000_ht_sols_acquire")
        >>> folder.write_traditional_ome_tif()
        >>> print(f"Converted files in: {folder.out_dir}")
    """
    
    def __init__(self, dir_path: str, out_dir: str | None = None, remove_timestamp: bool = True):
        """
        Initialize a SnoutyFolder instance.
        
        Args:
            dir_path (str): Path to the Snouty data directory containing 'data' and 'metadata' subdirectories
            out_path (str | None): Optional output path for the OME-TIFF file. 
                                  If None, defaults to '{dir_name}-skewed.ome.tif' in the input directory
            remove_timestamp (bool): Whether to remove the timestamp from the height_px dimension. Defaults to True.
        
        Raises:
            ValueError: If no data files or metadata files are found in the expected locations
        """
        self.dir_path = dir_path
        self.dir_name = Path(self.dir_path).name
        self.out_dir = out_dir or self.dir_path
        os.makedirs(self.out_dir, exist_ok=True)
        
        self._remove_timestamp = remove_timestamp
        self._num_timestamp_px = 8 if self._remove_timestamp else 0
        
        self.data_path = os.path.join(dir_path, "data")
        self.data_files = self._get_data_files()
        
        self.metadata_path = os.path.join(dir_path, "metadata")
        self.metadata_file = self._get_first_metadata_file()
        self.metadata = self._load_metadata()
        
        self.channels = self._load_channels()
        self.dim_resolutions = self._load_im_dim_res()
        self.dim_order = 'TCZYX'
        self.max_deshear_shift = self._get_max_deshear_shift()
        
        self.im_original_dims = self._load_original_dims()
        self.im_desheared_dims = self._load_desheared_dims()
        self.final_rotation_angle = 0.0
        self.im_traditional_dims = self._load_traditional_dims()
        
        self.im_original_path: str | None = None
        self.im_desheared_path: str | None = None
        self.im_traditional_path: str | None = None
        
    # ====================================================================
    # ------------ Public Methods ----------------------------------------
    def write_original_ome_tif(self, out_path: str | None = None) -> str:
        """
        Write the original Snouty data as an OME-TIFF file.
        
        This method creates an OME-TIFF file of the original Snouty data with proper 
        metadata including:
        - Physical pixel sizes and time increments
        - Channel information and wavelengths
        - Acquisition date and creator information
        - Original metadata as description
        
        The method first creates a TIFF file with the correct dimensions and data type,
        then loads the data from individual TIFF files and writes them to the OME-TIFF
        file using memory mapping for efficient processing.
        
        Args:
            out_path (str | None, optional): Path where the original OME-TIFF file will be saved.
                If None, uses the default path from the out_dir. Defaults to None.
        
        Returns:
            str: Path to the created OME-TIFF file
        """
        self.im_original_path = out_path or self._get_original_ome_path()
        self._write_ome_tif(self.im_original_path, self.im_original_dims)
        self._append_ome_metadata(self.im_original_path)
        
        # Write in the data - assumes each file is a single timepoint
        ome_memmap = tifffile.memmap(self.im_original_path, mode='r+')
        for i, data_file in enumerate(self.data_files):
            # Load the data file as a numpy array
            data = tifffile.imread(data_file)[..., self._num_timestamp_px:, :]
            # Swap first two dimensions to match OME-TIFF order (TCZYX)
            if len(self.channels) > 1:
                data = np.swapaxes(data, 0, 1)  # Assuming data is in (Z, C, Y, X) order
            # Write the data to the OME-TIFF file
            if len(self.data_files) > 1:
                ome_memmap[i, ...] = data
            else:
                ome_memmap[...] = data
        ome_memmap.flush()
        return self.im_original_path
    
    def write_desheared_ome_tif(self, output_path: str | None = None) -> str:
        """
        Write a desheared OME-TIFF file by correcting for scan-induced shear distortion.
        
        This method creates a desheared version of the original image by applying pixel shifts
        along the y-axis based on the scan step size and z-position. The deshearing process
        corrects for distortions introduced during the scanning acquisition process.
        Args:
            output_path (str | None, optional): Path where the desheared OME-TIFF file will be saved.
                If None, uses the default path from get_desheared_ome_path(). Defaults to None.
        Returns:
            str: The file path of the written desheared OME-TIFF file.
        Notes:
            - Creates the original OME-TIFF file if it doesn't exist
            - Uses memory mapping for efficient processing of large image stacks
            - Applies z-dependent y-shifts based on scan_step_size_px metadata
            - Preserves all original image dimensions and metadata
            - The deshear shift is calculated as: int(round(scan_step_size_px * z))
        """
        
        self.im_desheared_path = output_path or self._get_desheared_ome_path()
        self._write_ome_tif(self.im_desheared_path, self.im_desheared_dims)
        self._append_ome_metadata(self.im_desheared_path)
        
        if not self.im_original_path:
            self.im_original_path = self._get_original_ome_path()
            if not os.path.exists(self.im_original_path):
                print("Original OME-TIFF does not exist, creating it.")
                self.write_original_ome_tif()
                            
        src = tifffile.memmap(self.im_original_path,  mode='r')
        dst = tifffile.memmap(self.im_desheared_path, mode='r+')
        
        scan_step_size_px = self.metadata["scan_step_size_px"]
        
        # This is fast and memory efficient, no need to optimize for now
        self._per_slice_cpu_deshear(scan_step_size_px, src, dst)
        dst.flush()
        return self.im_desheared_path
    
    def write_traditional_ome_tif(self, output_path: str | None = None) -> str:
        """
        Write a traditional OME-TIFF file by applying deshearing and rotation transformations.
        
        This method creates a traditional view of the image data by:
        1. Deshearing the original image data to correct for scan step artifacts
        2. Rotating the desheared data to align with traditional viewing orientation
        3. Cropping the rotated result to fit the traditional dimensions
        The transformation process involves:
        - Loading desheared and original image memory maps
        - Calculating rotation angle based on scan step size
        - Processing each X position by deshearing Z-slices and rotating the result
        - Cropping the rotated planes to match traditional dimensions
        Args:
            output_path (str | None, optional): Custom output file path. If None, 
                generates path using output directory and folder name with 
                "-traditional.ome.tif" suffix.
        Returns:
            str: Path to the written traditional OME-TIFF file.
        Note:
            Requires desheared and original OME-TIFF files to exist. Will create 
            them if they don't exist. The method processes data in TCZYX order
            and uses bilinear interpolation for rotation.
        """
        self.im_traditional_path = output_path or self._get_traditional_ome_path()
        self._write_ome_tif(self.im_traditional_path, self.im_traditional_dims)
        self._append_ome_metadata(self.im_traditional_path)
        
        if not self.im_original_path:  
            self.im_original_path = self._get_original_ome_path()
            if not os.path.exists(self.im_original_path):
                print("Original OME-TIFF does not exist, creating it.")
                self.write_original_ome_tif()
            
        src = tifffile.memmap(self.im_original_path,    mode='r')
        dst = tifffile.memmap(self.im_traditional_path, mode='r+')
        
        self._affine_rotate(src, dst)
        dst.flush()
        return self.im_traditional_path
    
    # ====================================================================
    # ------------ Private Methods ----------------------------------------
    def _write_ome_tif(self, out_path: str, shape: tuple):
        """
        Write an OME-TIFF file with the specified shape and metadata.
        
        Args:
            out_path (str): Path to the output OME-TIFF file
            shape (tuple): Shape of the image data in (T, C, Z, Y, X) order
            
        Returns:
            None
        """
        tifffile.imwrite(
            out_path,
            shape=shape,
            dtype=np.uint16,
            bigtiff=True,
            ome=True,
            metadata=dict(
                axes=self.dim_order,
                PhysicalSizeX=self.dim_resolutions['PhysicalSizeX'], PhysicalSizeXUnit="µm",
                PhysicalSizeY=self.dim_resolutions['PhysicalSizeY'], PhysicalSizeYUnit="µm",
                PhysicalSizeZ=self.dim_resolutions['PhysicalSizeZ'], PhysicalSizeZUnit="µm",
                TimeIncrement=self.dim_resolutions['TimeIncrement'], TimeIncrementUnit="s",
            ),
        )
    
    def _append_ome_metadata(self, tifffile_path: str):
        """
        Modify OME metadata in a TIFF file.
        
        This method reads existing OME XML metadata from a TIFF file, modifies it with
        updated creator information, image name, acquisition date, and channel data,
        then writes the modified metadata back to the file.
        
        Args:
            tifffile_path (str): Path to the TIFF file to modify
            
        Returns:
            None
            
        Side Effects:
            - Modifies the OME XML metadata comment in the specified TIFF file
            - Updates creator field with email and preserves existing creator info
            - Sets image name to the directory name
            - Updates acquisition date to the file's modified datetime
            - Updates channel information from internal channel data
            
        Raises:
            May raise exceptions related to file I/O or XML parsing if the TIFF file
            is corrupted, inaccessible, or contains invalid OME metadata.
        """
        
        # Grab the OME XML and modify it
        ome_xml = tifffile.tiffcomment(tifffile_path)
        ome = from_xml(ome_xml) if ome_xml else model.OME()
        creator_string = 'austin.e.lefebvre@gmail.com'
        if ome.creator is not None:
            creator_string += f" ({ome.creator})"
        ome.creator = creator_string
        ome.images[0].name = self.dir_name
        ome.images[0].acquisition_date = self._get_modified_datetime()
        ome.images[0].pixels.channels = self._get_channel_ome()
        # All the metadata key values
        ome.images[0].description = self._load_metadata(as_string=True)["metadata"]
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(tifffile_path, ome_xml.encode())
    
    def _get_original_ome_path(self) -> str:
        """
        Get the path for the original OME-TIFF file.
        
        Returns:
            str: Path to the original OME-TIFF file
        """
        return os.path.join(self.out_dir, f"{self.dir_name}-original.ome.tif")
    
    def _get_desheared_ome_path(self) -> str:
        """
        Get the path for the desheared OME-TIFF file.
        
        Returns:
            str: Path to the desheared OME-TIFF file
        """
        return os.path.join(self.out_dir, f"{self.dir_name}-desheared.ome.tif")
    
    def _get_traditional_ome_path(self) -> str:
        """
        Get the path for the traditional OME-TIFF file.
        
        Returns:
            str: Path to the traditional OME-TIFF file
        """
        return os.path.join(self.out_dir, f"{self.dir_name}-traditional.ome.tif")
    
    def _get_first_metadata_file(self):
        """
        Get the first metadata file from the metadata directory, sorted by modification time.
        
        Returns:
            str: Path to the first (oldest) metadata file
            
        Raises:
            ValueError: If no metadata files are found in the metadata directory
        """
        # Get Metadata files and sort them by modified time
        metadata_files = [os.path.join(self.metadata_path, f) for f in os.listdir(self.metadata_path) if f.endswith(".txt")]
        metadata_files.sort(key=os.path.getmtime)
        # Only keep the first metadata file to extract metadata from
        if not metadata_files:
            raise ValueError(f"No metadata files found in {self.metadata_path}")
        return metadata_files[0]
    
    def _get_data_files(self):
        """
        Get all TIFF data files from the data directory, sorted by modification time.
        
        Returns:
            list: List of paths to data files sorted by modification time
            
        Raises:
            ValueError: If no data files are found in the data directory
        """
        # Get Data files and sort them by modified time
        data_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".tif")]
        data_files.sort(key=os.path.getmtime)
        if not data_files:
            raise ValueError(f"No data files found in {self.data_path}")
        return data_files
    
    def _load_metadata(self, as_string: bool = False) -> dict[str, Any]:
        """
        Load and parse metadata from the metadata file.
        
        Parses key-value pairs from the metadata file and attempts to convert
        values to appropriate data types (bool, int, float, tuple).
        
        Returns:
            dict: Dictionary containing parsed metadata with appropriately typed values
        """
        metadata = {}
        with open(self.metadata_file, 'r') as f:
            lines = f.readlines()
        if as_string:
            return dict(metadata="\n".join(lines))

        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate data type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.','').isdigit():
                    value = float(value) if '.' in value else int(value)
                elif value.startswith('(') and value.endswith(')'):
                    # Handle tuples
                    try:
                        value = eval(value)
                    except:
                        pass
                
                metadata[key] = value
        else:
            return metadata
                
    def _load_channels(self):
        """
        Load channel information from metadata.
        
        Extracts the channel wavelengths or identifiers from the metadata
        'channels_per_slice' field.
        
        Returns:
            list: List of channel wavelengths/identifiers
        """
        channels = self.metadata["channels_per_slice"]
        return channels

    def _load_original_dims(self) -> tuple[int, int, int, int, int]:
        """
        Calculate image dimensions from metadata.
        
        Returns:
            tuple: Image dimensions in TCZYX order (time, channels, z, y, x)
        """
        size_c = len(self.channels)
        size_t = self.metadata["volumes_per_buffer"] * len(self.data_files)
        size_z = self.metadata["slices_per_volume"]
        size_y = self.metadata["height_px"] - self._num_timestamp_px  # Adjust for timestamp if needed
        size_x = self.metadata["width_px"]
        dims = (size_t, size_c, size_z, size_y, size_x)
        return dims
    
    def _load_desheared_dims(self) -> tuple[int, int, int, int, int]:
        """
        Calculate desheared image dimensions.
        
        The desheared dimensions are the same as the original dimensions except for
        the Y dimension, which is expanded to accommodate the maximum deshear shift
        needed to correct for scan-induced shearing.
        
        Returns:
            tuple: Desheared image dimensions in TCZYX order (time, channels, z, y, x)
                  where Y dimension is expanded by max_deshear_shift pixels
        """
        return (
            self.im_original_dims[0],  # T
            self.im_original_dims[1],  # C
            self.im_original_dims[2],  # Z
            self.im_original_dims[3] + self.max_deshear_shift,  # Y
            self.im_original_dims[4],  # X
        )
    
    def _load_traditional_dims(self) -> tuple[int, int, int, int, int]:
        """
        Calculate the dimensions for a traditional top-down view after rotation.
        
        This method computes the rotated image dimensions based on the scan step size
        to provide a traditional viewing perspective. The rotation is applied to 
        transform the original acquisition geometry into a more intuitive top-down view.
        
        Returns:
            tuple[int, int, int, int, int]: A tuple containing the rotated image shape
                in the format (T, C, Y, Z, X) where:
                - T: Time dimension (unchanged)
                - C: Channel dimension (unchanged) 
                - Y: Rotated Y dimension (originally Z)
                - Z: Rotated Z dimension (originally Y)
                - X: X dimension (unchanged)
        
        Notes:
            - The rotation angle is calculated from the scan step size in pixels
            - Y and Z dimensions are swapped to achieve the traditional top-down view
            - The rotated dimensions account for the geometric transformation to avoid
              data clipping during rotation
        """
        scan_step_size_px = self.metadata["scan_step_size_px"]
        voxel_aspect_ratio = self.metadata["voxel_aspect_ratio"]
        final_rotation_angle = np.arctan(scan_step_size_px/voxel_aspect_ratio)
        initial_rotation_angle = np.arctan(scan_step_size_px)
        z_original = self.im_original_dims[2]
        y_original = self.im_original_dims[3]
        y_rotated = int(np.rint(np.sin(initial_rotation_angle) * y_original))
        z_rotated = int(np.rint(
            (z_original*voxel_aspect_ratio / (np.cos(final_rotation_angle))) + 
            (np.cos(initial_rotation_angle) * y_original/voxel_aspect_ratio)
        ))
        im_rotated_shape = (
            self.im_original_dims[0],  # T
            self.im_original_dims[1],  # C
            y_rotated,                 # Z  we flip z and y to get a traditional top down view
            z_rotated,                 # Y
            self.im_original_dims[4],  # X
        )
        return im_rotated_shape
    
    def _load_im_dim_res(self):
        """
        Calculate physical resolution information from metadata.
        
        Returns:
            dict: Dictionary containing physical sizes and time increment:
                - PhysicalSizeX: X-axis pixel size in micrometers
                - PhysicalSizeY: Y-axis pixel size in micrometers  
                - PhysicalSizeZ: Z-axis pixel size in micrometers
                - TimeIncrement: Time between frames in seconds
        """
        if len(self.data_files) > 1:
            # Determine the slice for the channel dimension based on the number of channels
            channel_slice = 0 if len(self.channels) > 1 else None
            
            # Construct the full slice tuple, omitting the channel slice if only one channel
            slice_v1 = (0, channel_slice, slice(None, 1), slice(None, 14))
            slice_v2 = (0, channel_slice, slice(None, 1), slice(None, 14))

            # Filter out None from the slice tuple
            slice_v1 = tuple(s for s in slice_v1 if s is not None)
            slice_v2 = tuple(s for s in slice_v2 if s is not None)

            v1 = tifffile.memmap(self.data_files[0], mode='r')[slice_v1]
            v2 = tifffile.memmap(self.data_files[1], mode='r')[slice_v2]
            ts1 = decode_timestamp(v1)
            ts2 = decode_timestamp(v2)
            time_res = (ts2['time_us'] - ts1['time_us']) * 1e-6
        else:
            time_res = 0.0
            
        xy_res_um = self.metadata["sample_px_um"]
        z_ratio = self.metadata["voxel_aspect_ratio"]
        z_res_um = xy_res_um * z_ratio
        return {
            'PhysicalSizeX': xy_res_um,
            'PhysicalSizeY': xy_res_um,
            'PhysicalSizeZ': z_res_um,
            'TimeIncrement': time_res,
        }
    
    def _get_channel_ome(self):
        """
        Generate a list of OME Channel objects from the instance's channels.
        
        Creates Channel objects with sequential IDs, names based on channel numbers,
        and emission wavelengths from the channels attribute. Each channel is 
        configured with 1 sample per pixel.
        
        Returns:
            list: A list of model.Channel objects representing the channels in OME format.
            
        Note:
            This is a private method intended for internal use in OME metadata generation.
        """
        channel_ome = []
        for ch_num, channel in enumerate(self.channels):
            channel_ome.append(
                model.Channel(
                    id=f"Channel:{ch_num}",
                    name=str(ch_num),
                    emission_wavelength=channel,
                    samples_per_pixel=1,
                )
            )
        return channel_ome
    
    def _get_max_deshear_shift(self):
        """
        Calculate the maximum deshear shift based on scan step size and number of slices.
        
        Returns:
            int: Maximum deshear shift in pixels
        """
        scan_step_size_px = self.metadata["scan_step_size_px"]
        num_z = self.metadata["slices_per_volume"]
        max_deshear_shift = int(np.rint(scan_step_size_px * (num_z - 1)))
        return max_deshear_shift
    
    def _get_modified_datetime(self):
        """
        Get the last modified datetime of the first data file.

        Returns:
            datetime: The datetime object representing when the first data file
                     in self.data_files was last modified.

        Raises:
            IndexError: If self.data_files is empty.
            OSError: If the file does not exist or cannot be accessed.
        """
        # convert modified time to a datetime object
        modified_time = os.path.getmtime(self.data_files[0])
        return datetime.fromtimestamp(modified_time)
    
    # ====================================================================
    # ------------ Deshearing and Rotation methods -----------------------
    def _per_slice_cpu_deshear(self, scan_step_size_px: float, 
                               src: np.ndarray, dst: np.ndarray):
        """
        Perform CPU-based deshearing operation on image data slice by slice.
        Fast and memory efficient.
        """
        
        T, C, Z, Y, X = self.im_original_dims
        for z in range(Z):
            deshear_shift = int(np.rint(scan_step_size_px * z))
            y_slice = slice(deshear_shift, deshear_shift + Y)
            for c in range(C):
                for t in range(T):
                    dst[t, c, z, y_slice, :] = src[t, c, z, :, :]
                            
    def _affine_rotate(self, src: np.ndarray, dst: np.ndarray):
        """
        Deskew a sheared (TxCxZxYxX) stack (src), rotate it into
        "traditional" orthogonal zyx view, and write the result to dst.
        """
        zoom = self.metadata["voxel_aspect_ratio"]
        scan_step_size_px = self.metadata["scan_step_size_px"]
        rotation_angle = scan_step_size_px/zoom
        M   = np.linalg.inv(_affine_matrix(rotation_angle=rotation_angle, z_zoom=zoom))
        T, C, Z, Y, X = self.im_original_dims
        Tt, Ct, Zt, Yt, Xt = self.im_traditional_dims
        offset = np.zeros(3, dtype=np.float64)
        
        # Determine if src has an explicit channel dimension (C > 1)
        src_has_channel_dim = C > 1
        
        if gpu and cp is not None:
            M = cp.asarray(M)
            offset = cp.asarray(offset)
            desheared_vol = cp.zeros((Z, Y + self.max_deshear_shift, X), dtype=cp.uint16)
        else:
            desheared_vol = np.zeros((Z, Y + self.max_deshear_shift, X), dtype=np.uint16)
        
        for t_idx in range(T):
            for c_idx in range(C):
                print(f"Processing {t_idx=}/{T}, {c_idx=}/{C}")
                for z_idx in range(Z):
                    deshear_shift = int(np.rint(scan_step_size_px * z_idx))
                    y_slice = slice(deshear_shift, deshear_shift + Y)
                    
                    # Extract the current slice from src, handling T and C dimensions
                    if T > 1: # If there's an explicit time dimension
                        if src_has_channel_dim:
                            current_slice = src[t_idx, c_idx, z_idx, :, :]
                        else: # C == 1, src is (T, Z, Y, X)
                            current_slice = src[t_idx, z_idx, :, :]
                    else: # T == 1, src is (C, Z, Y, X) or (Z, Y, X)
                        if src_has_channel_dim:
                            current_slice = src[c_idx, z_idx, :, :]
                        else: # T == 1 and C == 1, src is (Z, Y, X)
                            current_slice = src[z_idx, :, :]

                    if gpu and cp is not None:
                        desheared_vol[z_idx, y_slice, :] = cp.asarray(current_slice)
                    else:
                        desheared_vol[z_idx, y_slice, :] = current_slice
                
                # 2. rotate around x‑axis
                vol_out = aff_trf(
                    desheared_vol,
                    matrix=M,
                    offset=offset,  # type: ignore
                    order=0,
                    prefilter=False,
                    output_shape=(Yt, Zt, Xt),       # crop happens here
                )

                # 3. swap axes to get traditional view
                if gpu and cp is not None:
                    vol_out = cp.swapaxes(vol_out, 0, 1)
                    # Flip the Z so bottom is on bottom
                    vol_out = cp.flip(vol_out, axis=0)
                    # Convert back to numpy for writing to dst
                    if T > 1:
                        if src_has_channel_dim:
                            dst[t_idx, c_idx] = cp.asnumpy(vol_out)
                        else:
                            dst[t_idx] = cp.asnumpy(vol_out)
                    else: # T == 1
                        if src_has_channel_dim:
                            dst[c_idx] = cp.asnumpy(vol_out)
                        else:
                            dst[:] = cp.asnumpy(vol_out) # Assign to the single volume
                else:
                    vol_out = np.swapaxes(vol_out, 0, 1)
                    # Flip the Z so bottom is on bottom
                    vol_out = np.flip(vol_out, axis=0)
                    if T > 1:
                        if src_has_channel_dim:
                            dst[t_idx, c_idx] = vol_out
                        else:
                            dst[t_idx] = vol_out
                    else: # T == 1
                        if src_has_channel_dim:
                            dst[c_idx] = vol_out
                        else:
                            dst[:] = vol_out # Assign to the single volume
    
def _affine_matrix(rotation_angle, z_zoom: float) -> np.ndarray:
    """Return the 3x3 (z,y,x) rotation matrix for +/- atan(rotation_angle)
    about the x-axis with z scaling applied before rotation."""
    # theta = np.pi/2 - np.arctan(rotation_angle)
    theta = np.arctan(rotation_angle)
        
    c, s = np.cos(theta), np.sin(theta)

    # # Scaling matrix for z
    scale_matrix = np.array([[z_zoom, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1]], dtype=np.float64)
    
    # Rotation matrix about x-axis
    rotation_matrix = np.array([[ c,  s, 0],
                               [ -s,  c, 0],
                               [  0,  0, 1]], dtype=np.float64)
    
    # return scale_matrix @ rotation_matrix
    return rotation_matrix @ scale_matrix
    # return rotation_matrix

def decode_timestamp(image):
    """
    From Alfred:
    https://github.com/amsikking/pco_decode_timestamp/blob/main/pco_decode_timestamp.py
    
    Decode PCO image timestamps from binary-coded decimal (see p94 of
    "pco_camera_control_commands_105.pdf"). In this version of 'packed BCD' each
    pixel contains 2 digits of information in a single byte (8bits). The lower
    and upper nibbles (2 x 4bits) encode the numbers 0-9 which are then combined
    to give a value in the range 0-99.
    """
    assert len(image.shape) == 2 and image.dtype == 'uint16'
    bcd_px = image[0, :14]                      # get BCD pixels
    lower_nibbles =  bcd_px & 0b00001111        # get lower nibbles
    upper_nibbles = (bcd_px & 0b11110000) >> 4  # get upper nibbles and shift
    dec_px = 10 * upper_nibbles + lower_nibbles # convert to decimal
    timestamp = {}
    timestamp['#'] = np.sum(
        dec_px[:4] * np.array((1e6, 1e4, 1e2, 1)), dtype='uint32')
    timestamp['DD'] = dec_px[7].astype('uint32')
    timestamp['MM'] = dec_px[6].astype('uint32')    
    timestamp['YYYY'] = np.sum(
        dec_px[4:6] * np.array((1e2, 1)), dtype='uint32')
    timestamp['h'] = dec_px[8].astype('uint32')
    timestamp['min'] = dec_px[9].astype('uint32')
    timestamp['s'] = dec_px[10].astype('uint32')
    timestamp['us'] = np.sum(
        dec_px[11:14] * np.array((1e4, 1e2, 1), dtype='uint64'))
    timestamp['time_us'] = np.sum(              # total us on a given day
        dec_px[8:14] * np.array((36e8, 60e6, 1e6, 1e4, 1e2, 1)), dtype='uint64')
    return timestamp


if __name__ == "__main__":
    folders = [
        r"C:\test_files_C\john_calcium_single",
        # r"D:\test_files\test_snouty\2025-07-11_17-43-38_000_ht_sols_acquire",
    ]
    out_dir = r"C:\test_files_C\john_calcium_single_out"
    # Example usage
    for folder in folders:
        snouty_folder = SnoutyFolder(folder, out_dir=out_dir)
        output_path = snouty_folder.write_traditional_ome_tif()
        print("OME-TIFF file written successfully.")
