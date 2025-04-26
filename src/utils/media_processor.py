#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 

Version: 
Autor: dreamy-xay
Date: 2024-10-28 17:25:37
LastEditors: dreamy-xay
LastEditTime: 2024-10-29 17:22:10
"""
import os
import cv2
from PIL import ImageGrab
from tqdm import tqdm


class MediaProcessor:
    image_suffix = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    video_suffix = (".mp4", ".avi", ".mov", ".flv")

    def __init__(self, src_path, dst_path, run_func, return_input_image=True):
        self.src_path = src_path
        self.dst_path = dst_path
        self.is_save = os.path.exists(dst_path) if isinstance(dst_path, str) else False

        self.run = run_func

        self.return_input_image = return_input_image

    def process_image(self, image_path, show=False, save=False):
        save = save and self.is_save

        # Read the image and run the run function
        image = cv2.imread(image_path)

        # If the image fails to be read, an exception is thrown
        if image is None:
            raise ValueError(f"Don't read image: {image_path}")

        # Get the name of the image
        image_name = os.path.basename(image_path)

        # Run the run function
        result, result_image = self.run(image, image_name)

        # Display the result image
        if show:
            self.show_image(result_image, image_name)

        # Save the result image
        if save:
            cv2.imwrite(os.path.join(self.dst_path, image_name), result_image)

        if self.return_input_image:
            return {"type": "image", "name": image_name, "result": result, "image": image}
        else:
            return {"type": "image", "name": image_name, "result": result}

    def process_video(self, video_path, show=False, save=False):
        save = save and self.is_save

        # Read the video and process it frame by frame
        cap = cv2.VideoCapture(video_path)

        # If the video fails to be read, an exception is thrown
        if not cap.isOpened():
            raise ValueError(f"Don't open video: {video_path}")

        # Obtain the file name of the video
        video_name = os.path.basename(video_path)
        _video_name, _video_ext = os.path.splitext(video_name)

        if show:
            video_name_id = video_name.replace(".", "_").replace(" ", "_")

        if save:
            # Get the basic information of the video
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a VideoWriter object
            out = cv2.VideoWriter(os.path.join(self.dst_path, video_name), fourcc, fps, (frame_width, frame_height))

        # Create a result save array
        results = []

        # Current frame rate
        current_fps = 1

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing Video [{video_name}]", unit="frame") as bar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_video_name = f"{_video_name}_fps{current_fps}.{_video_ext}"
                result, processed_frame = self.run(frame, current_video_name)  # Process every frame

                results.append(result)  # Keep the result of each frame into the result array

                if show:
                    self.show_image(processed_frame, current_video_name, cache_id=video_name_id)  # Displays the processed frame

                if save:
                    out.write(processed_frame)  # Write the processed frame to the video

                current_fps += 1
                bar.update(1)

        cap.release()

        if save:
            out.release()

        return {"type": "video", "name": video_name, "result": results}

    def process_directory(self, directory_path, show=False, save=False):
        # Get all the file paths in the directory
        file_path_list = self.get_file_paths(directory_path)
        # Save the results in an array
        results = []
        # Iterate and run all the files in the directory
        with tqdm(total=len(file_path_list), desc="Processing Media", unit="file") as bar:
            for filename, file_path in file_path_list:
                result = self.process_file(bar, filename, file_path, show, save)
                if result is not None:
                    results.append(result)
        return results

    def process_file(self, bar, filename, file_path, show=False, save=False):
        result = None
        if file_path.lower().endswith(self.image_suffix):
            bar.set_postfix({"image_name": filename})
            result = self.process_image(file_path, show, save)
        elif file_path.lower().endswith(self.video_suffix):
            bar.set_postfix({"video_name": filename})
            result = self.process_video(file_path, show, save)
        else:
            bar.set_postfix({"invalid_file_name": filename})

        bar.update(1)

        return result

    def process(self, show=False, save=False):
        # Save the results in an array
        results = []
        # Processed according to the path type
        if os.path.isfile(self.src_path):
            if self.path.lower().endswith(self.image_suffix):
                results.append(self.process_image(self.src_path, show, save))
            elif self.path.lower().endswith(self.video_suffix):
                results.append(self.process_video(self.src_path, show, save))
        elif os.path.isdir(self.src_path):
            return self.process_directory(self.src_path, show, save)
        else:
            raise ValueError(f"Invalid path: {self.src_path}")

        return results

    @staticmethod
    def show_image(image, winname="Demo", ratio=3 / 4, cache_id=""):
        """
        Show image using OpenCV.

        Args:
            image (np.ndarray): the image to be shown.
            ratio (float): the ratio of the image height to the screen height.

        Returns:
            None
        """
        if cache_id and hasattr(MediaProcessor, f"image_size_cache{cache_id}"):
            image_size_cache = getattr(MediaProcessor, f"image_size_cache{cache_id}", None)
            image_width = image_size_cache[0]
            image_height = image_size_cache[1]
        else:
            screen = ImageGrab.grab()
            screen_width, screen_height = screen.size

            # Calculate the ratio of setting the height of the image to the height of the screen
            image_height = int(screen_height * ratio)
            image_width = int(image.shape[1] * (image_height / image.shape[0]))

            # If the width of the image exceeds the ratio of the width of the screen, adjust the width and height to maintain the image aspect ratio
            if image_width > screen_width * ratio:
                image_width = int(screen_width * ratio)
                image_height = int(image.shape[0] * (image_width / image.shape[1]))

            if cache_id:
                setattr(MediaProcessor, f"image_size_cache{cache_id}", (image_width, image_height))

        # Resize the image
        image_resized = cv2.resize(image, (image_width, image_height))

        # Show image
        cv2.imshow(winname, image_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def get_file_paths(directory_path):
        # Get all the file paths in the directory
        file_path_list = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_path_list.append((filename, file_path))

        return file_path_list
