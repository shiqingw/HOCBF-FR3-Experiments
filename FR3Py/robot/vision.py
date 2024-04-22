import threading

import numpy as np
import pyrealsense2 as rs
import cv2

class RealSenseCamera:
    """
    A class for interacting with a RealSense cameras.

    Args:
        callback_fn (callable, optional): The callback function to process frames.
        camera_serial_no (str, optional): The serial number of the camera.
        VGA (bool): Set to True for VGA resolution, False for HD resolution.
        color_fps (int): Frame rate for color stream (frames per second).
        depth_fps (int): Frame rate for depth stream (frames per second).
        enable_imu (bool): Enable or disable IMU stream.
        enable_depth (bool): Enable or disable depth stream.
        enable_color (bool): Enable or disable color stream.
        enable_ir (bool): Enable or disable infrared stream.
        emitter_enabled (bool): Enable or disable emitter for the depth sensor.
        align_to_color (bool): Align depth and IR streams to color.

    Attributes:
        callback_fn (callable, optional): The callback function to process frames.
        camera_serial_no (str): The serial number of the camera.
        VGA (bool): True if VGA resolution, False if HD resolution.
        color_fps (int): Frame rate for color stream (frames per second).
        depth_fps (int): Frame rate for depth stream (frames per second).
        enable_imu (bool): True if IMU stream is enabled.
        enable_depth (bool): True if depth stream is enabled.
        enable_color (bool): True if color stream is enabled.
        enable_ir (bool): True if infrared stream is enabled.
        emitter_enabled (bool): True if emitter is enabled for the depth sensor.
        align_to_color (bool): True if depth and IR streams are aligned to color.
    """

    def __init__(
        self,
        callback_fn=None,
        camera_serial_no=None,
        VGA=False,
        color_fps=60,
        depth_fps=90,
        enable_imu=False,
        enable_depth=True,
        enable_color=True,
        enable_ir=True,
        emitter_enabled=True,
        align_to_color=False,
    ):
        self.callback_fn = callback_fn
        self.camera_serial_no = camera_serial_no
        self.VGA = VGA
        self.color_fps = color_fps
        self.depth_fps = depth_fps
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.align_to_color = align_to_color
        self.enable_ir = enable_ir
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.enable_imu = enable_imu
        if self.camera_serial_no is None:
            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            serial_no = str(device.get_info(rs.camera_info.serial_number))
            self.camera_serial_no = serial_no

        # Enable the streams for the connected device with the requested serial number
        print("Enabling streams for camera: ", self.camera_serial_no)
        self.config.enable_device(self.camera_serial_no)
        if VGA:
            img_size = (640, 480)
            if color_fps > 60:
                self.color_fps = 60
                print("Warning: VGA color fps cannot be higher than 60")
            if depth_fps > 90:
                self.depth_fps = 90
                print("Warning: VGA depth/infrared fps cannot be higher than 90")
        else:
            img_size = (1280, 720)
            if color_fps > 30:
                self.color_fps = 30
                print("Warning: HD color fps cannot be higher than 30")
            if depth_fps > 30:
                self.depth_fps = 30
                print("Warning: HD depth/infrared fps cannot be higher than 30")

        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth, img_size[0], img_size[1], rs.format.z16, self.depth_fps
            )
        if enable_color:
            self.config.enable_stream(
                rs.stream.color,
                img_size[0],
                img_size[1],
                rs.format.bgr8,
                self.color_fps,
            )
        if enable_ir:
            self.config.enable_stream(
                rs.stream.infrared,
                1,
                img_size[0],
                img_size[1],
                rs.format.y8,
                self.depth_fps,
            )
            self.config.enable_stream(
                rs.stream.infrared,
                2,
                img_size[0],
                img_size[1],
                rs.format.y8,
                self.depth_fps,
            )
        if self.enable_imu:
            self.config.enable_stream(rs.stream.accel)
            self.config.enable_stream(rs.stream.gyro)

        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        if emitter_enabled:
            self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
        else:
            self.depth_sensor.set_option(rs.option.emitter_enabled, 0)

        # Start the thread for grabbing frames
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_grab_frames)
        self._thread.start()
        # Get stream profiles
        self.depth_profile = self.profile.get_stream(
            rs.stream.depth
        ).as_video_stream_profile()
        self.color_profile = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        self.ir1_profile = self.profile.get_stream(
            rs.stream.infrared, 1
        ).as_video_stream_profile()
        self.ir2_profile = self.profile.get_stream(
            rs.stream.infrared, 2
        ).as_video_stream_profile()
        # Depth frame aligner
        self.align = rs.align(rs.stream.color)

    def _run_grab_frames(self):
        """
        Private method to continuously grab frames in a separate thread.
        """
        while not self._stop_event.is_set():
            self.grab_frames()
            if self.callback_fn is not None:
                self.callback_fn(
                    self.color_frame, self.depth_frame, self.ir1_frame, self.ir2_frame
                )

    def close(self):
        """
        Closes the RealSenseCamera object, stopping the frame grabbing thread and the pipeline.
        """
        # Stop the thread
        self._stop_event.set()
        self._thread.join()
        # Stop the pipeline
        self.pipeline.stop()

    def grab_frames(self):
        """
        Grabs frames from the RealSense camera and stores them in instance variables.
        """
        frames = self.pipeline.wait_for_frames()
        if self.align_to_color:
            frames = self.align.process(frames)
        if frames is None:
            print("Warning: failed to grab frames")
            self.close()

        if self.enable_depth:
            self.depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
        else:
            self.depth_frame = None
        if self.enable_color:
            self.color_frame = np.asanyarray(frames.get_color_frame().get_data())
        else:
            self.color_frame = None
        if self.enable_ir:
            self.ir1_frame = np.asanyarray(frames.get_infrared_frame(1).get_data())
            self.ir2_frame = np.asanyarray(frames.get_infrared_frame(2).get_data())
        else:
            self.ir1_frame = None
            self.ir2_frame = None

        if self.enable_imu:
            self.accel_frame = frames.first_or_default(rs.stream.accel)
            self.gyro_frame = frames.first_or_default(rs.stream.gyro)
            # Optionally convert the IMU frames to arrays as needed, e.g., np.asanyarray(...)
        else:
            self.accel_frame = None
            self.gyro_frame = None

    def getIntrinsics(self):
        """
        Gets the intrinsic parameters of cameras.

        Returns:
            dict: A dictionary containing intrinsics for RGB, IR1, IR2, and Depth streams.
        """
        rgb_intr = self.color_profile.get_intrinsics()
        ir1_intr = self.ir1_profile.get_intrinsics()
        ir2_intr = self.ir2_profile.get_intrinsics()
        depth_intr = self.depth_profile.get_intrinsics()

        ir1_intr = self.parseIntr(ir1_intr)
        ir2_intr = self.parseIntr(ir2_intr)
        depth_intr = self.parseIntr(depth_intr)
        rgb_intr = self.parseIntr(rgb_intr)

        return {"RGB": rgb_intr, "IR1": ir1_intr, "IR2": ir2_intr, "Depth": depth_intr}

    def getExtrinsics(self):
        """
        Gets the extrinsics between different camera streams.

        Returns:
            dict: A dictionary containing pose of all streams with respect to depth/infrared1.
        """
        ir1_T_ir2 = self.ir2_profile.get_extrinsics_to(
            self.ir1_profile
        )  # Pose of ir2 with respect to ir1
        ir1_T_rgb = self.color_profile.get_extrinsics_to(
            self.ir1_profile
        )  # Pose of rgb with respect to ir1
        ir1_T_ir2 = self.toPose(ir1_T_ir2)
        ir1_T_rgb = self.toPose(ir1_T_rgb)
        return {"ir1_T_ir2": ir1_T_ir2, "ir1_T_rgb": ir1_T_rgb}

    def toPose(self, e):
        """
        Converts a RealSense extrinsics object to a 4x4 transformation matrix.

        Args:
            e (rs.extrinsics): The extrinsics object.

        Returns:
            numpy.ndarray: A 4x4 transformation matrix representing the extrinsics.
        """
        R = np.array(e.rotation).reshape(3, 3)
        t = np.array(e.translation).reshape(3, 1)
        return np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])

    def parseIntr(self, intr):
        """
        Parses intrinsics data into a dictionary for easier access.

        Args:
            intr (rs.intrinsics): The intrinsics object.

        Returns:
            dict: A dictionary containing intrinsics data.
        """
        h, w = intr.height, intr.width
        fx, fy = intr.fx, intr.fy
        cx, cy = intr.ppx, intr.ppy
        dist = np.array(intr.coeffs)
        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        return {
            "size": (w, h),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "K": K,
            "D": dist,
        }
    
    def alignDepth2Color(self):

        depth_scale = self.depth_sensor.get_depth_scale()

        print("Depth Scale is: " , depth_scale)

        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale


        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        # if not aligned_depth_frame or not color_frame:
            

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        return {"Depth" : depth_image, "Color" : color_image}

    


def depth2PointCloud(
    depth_map, fx, fy, cx, cy, min_depth=0.2, max_depth=3.0, depth_scale=1000.0
):
    """
    Converts a depth map to a point cloud.

    Args:
        depth_map (numpy.ndarray): The depth map as a 2D numpy array.
        fx (float): The focal length in the x-axis.
        fy (float): The focal length in the y-axis.
        cx (float): The principal point in the x-axis.
        cy (float): The principal point in the y-axis.
        min_depth (float, optional): Minimum valid depth value (default: 0.2 meters).
        max_depth (float, optional): Maximum valid depth value (default: 3.0 meters).
        depth_scale (float, optional): Scaling factor to convert depth values (default: 1000.0).

    Returns:
        numpy.ndarray: A point cloud represented as a numpy array of shape (N, 3) where N is the number of points.

    Note:
        This function applies median filtering and removes outliers from the depth map
        before converting it to a point cloud.

    Example:
        depth_map = load_depth_map()
        fx = 500.0
        fy = 500.0
        cx = 320.0
        cy = 240.0
        point_cloud = depth2PointCloud(depth_map, fx, fy, cx, cy)
    """
    rows, cols = depth_map.shape

    # Apply the median filter to the depth image
    depth = cv2.medianBlur(depth_map, ksize=3)
    depth = depth.astype(np.float32) / depth_scale

    # Remove the outliers
    idx = np.where(
        ((depth.reshape(-1) <= min_depth) + (depth.reshape(-1) >= max_depth))
    )[0]

    # Create a mesh grid for the depth map
    mesh_x, mesh_y = np.meshgrid(np.arange(cols), np.arange(rows))
    f = (fx + fy) / 2

    # Calculate 3D coordinates (X, Y, Z) from depth values
    x = (mesh_x - cx) * depth / f
    y = (mesh_y - cy) * depth / f
    z = depth
    point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    point_cloud = np.delete(point_cloud, idx, axis=0)

    return point_cloud

def saveTo16bits(image):
    img_name = "{}.png".format(img_name)
    cv2.imwrite(img_name, image.astype(np.uint16), -1)







    