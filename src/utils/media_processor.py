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

        # 读取图片并运行 run 函数
        image = cv2.imread(image_path)

        # 如果图片读取失败，抛出异常
        if image is None:
            raise ValueError(f"Don't read image: {image_path}")

        # 获取图像名
        image_name = os.path.basename(image_path)

        # 运行 run 函数
        result, result_image = self.run(image, image_name)

        # 显示结果图片
        if show:
            self.show_image(result_image, image_name)

        # 保存结果图片
        if save:
            cv2.imwrite(os.path.join(self.dst_path, image_name), result_image)

        if self.return_input_image:
            return {"type": "image", "name": image_name, "result": result, "image": image}
        else:
            return {"type": "image", "name": image_name, "result": result}

    def process_video(self, video_path, show=False, save=False):
        save = save and self.is_save

        # 读取视频并逐帧处理
        cap = cv2.VideoCapture(video_path)

        # 如果视频读取失败，抛出异常
        if not cap.isOpened():
            raise ValueError(f"Don't open video: {video_path}")

        # 获取视频文件名
        video_name = os.path.basename(video_path)
        _video_name, _video_ext = os.path.splitext(video_name)

        if show:
            video_name_id = video_name.replace(".", "_").replace(" ", "_")

        if save:
            # 获取视频的基本信息
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建 VideoWriter 对象
            out = cv2.VideoWriter(os.path.join(self.dst_path, video_name), fourcc, fps, (frame_width, frame_height))

        # 创建结果保存数组
        results = []

        # 当前帧数
        current_fps = 1

        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing Video [{video_name}]", unit="frame") as bar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_video_name = f"{_video_name}_fps{current_fps}.{_video_ext}"
                result, processed_frame = self.run(frame, current_video_name)  # 处理每一帧

                results.append(result)  # 将每一帧的结果保持到结果数组中

                if show:
                    self.show_image(processed_frame, current_video_name, cache_id=video_name_id)  # 显示处理后的帧

                if save:
                    out.write(processed_frame)  # 将处理后的帧写入视频

                current_fps += 1
                bar.update(1)

        cap.release()

        if save:
            out.release()

        return {"type": "video", "name": video_name, "result": results}

    def process_directory(self, directory_path, show=False, save=False):
        # 获取目录中的所有文件路径
        file_path_list = self.get_file_paths(directory_path)
        # 结果保存数组
        results = []
        # 遍历并运行目录中的所有文件
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
        # 结果保存数组
        results = []
        # 根据路径类型进行处理
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

            # 计算设置图片高度为屏幕高度的比率
            image_height = int(screen_height * ratio)
            image_width = int(image.shape[1] * (image_height / image.shape[0]))

            # 如果图片宽度超过屏幕宽度的比率，调整宽度和高度以保持图片宽高比
            if image_width > screen_width * ratio:
                image_width = int(screen_width * ratio)
                image_height = int(image.shape[0] * (image_width / image.shape[1]))

            if cache_id:
                setattr(MediaProcessor, f"image_size_cache{cache_id}", (image_width, image_height))

        # 调整图片尺寸
        image_resized = cv2.resize(image, (image_width, image_height))

        # 显示图片
        cv2.imshow(winname, image_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def get_file_paths(directory_path):
        # 获取目录中的所有文件路径
        file_path_list = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_path_list.append((filename, file_path))

        return file_path_list
