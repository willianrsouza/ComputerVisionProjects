import os
import cv2
import numpy as np
import regex as re


class ImageSearch:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=8, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    def search(self):
        base_path = os.path.dirname(__file__)
        target_path = os.path.abspath(os.path.join(base_path, 'search', 'img.jpg'))
        try:
            closest_indices = self._matching(target_path)
            images = self._get_images(closest_indices)
            return images
        except Exception as ex:
            print(ex)
            return None

    def _cut_scenes(self):
        base_path = os.path.dirname(__file__)
        movie_path = os.path.abspath(os.path.join(base_path, './movies/Pulp-Fiction-1994'))
        output_directory = os.path.abspath(os.path.join(base_path, './scenes'))

        captures = cv2.VideoCapture(movie_path)

        frame_interval = 30
        current_frame = 0

        while True:
            ret, frame = captures.read()

            if not ret:
                break

            if current_frame % frame_interval == 0:
                frame_filename = f'{output_directory}frame_{current_frame}.jpg'
                cv2.imwrite(frame_filename, frame)

                print(f"Frame {current_frame} saved as {frame_filename}")

            current_frame += 1
        captures.release()

    def _generate_scenes_descriptor(self):
        base_path = os.path.dirname(__file__)
        scenes = os.path.abspath(os.path.join(base_path, './scenes'))
        scenes_descriptors_path = os.path.abspath(os.path.join(base_path, './scenes_descriptors'))

        for i, filename in enumerate(os.listdir(scenes)):
            if filename.endswith('.jpg'):
                scene_path = os.path.join(scenes, filename)
                current_scene = cv2.imread(scene_path)
                current_scene_gray = cv2.cvtColor(current_scene, cv2.COLOR_BGR2GRAY)
                current_scene_gray = cv2.GaussianBlur(current_scene_gray, (5, 5), 0)

                _, descriptor = self.sift.detectAndCompute(current_scene_gray, None)

                if descriptor is not None and len(descriptor) > 0:
                    descriptor = descriptor[:1_000_000]

                    np.save(os.path.join(scenes_descriptors_path, f'{filename}.npy'), descriptor)

    def _load_descriptors(self):
        base_path = os.path.dirname(__file__)
        scenes_descriptors_path = os.path.abspath(os.path.join(base_path, './scenes_descriptors'))

        descriptors_list = []

        for filename in os.listdir(scenes_descriptors_path):
            if filename.endswith('.npy'):
                descriptor_path = os.path.join(scenes_descriptors_path, filename)

                descriptors = np.load(descriptor_path, allow_pickle=True)
                if descriptors is not None and descriptors.ndim == 2 and descriptors.size > 0:
                    descriptors_list.append((filename, descriptors))

        return descriptors_list

    def _matching(self, target_path):
        target_image = cv2.imread(target_path)
        target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        _, new_descriptors = self.sift.detectAndCompute(target_image_gray, None)
        descriptors_list = self._load_descriptors()

        average_distances = []
        descriptor_names = []

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for descriptor_name, descriptors in descriptors_list:
            matches = bf.match(descriptors, new_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            distances = [match.distance for match in matches]

            average_distance = np.mean(distances)
            average_distances.append(average_distance)
            descriptor_names.append(descriptor_name)

        sorted_indices = np.argsort(average_distances)
        closest_indices = sorted_indices[:4]

        closest_descriptors = [(descriptor_names[i]) for i in closest_indices]

        return closest_descriptors

    def _get_images(self, frames_to_pick: list):
        base_path = os.path.dirname(__file__)
        saved_frames = os.path.abspath(os.path.join(base_path, './scenes'))

        matched_images = []

        for frame_number in frames_to_pick:
            frame = re.search(r'\d+', frame_number)
            if frame:
                frame_number_str = frame.group()

                for filename in os.listdir(saved_frames):
                    number_match = re.search(r'\d+', filename)
                    if number_match and number_match.group() == frame_number_str:
                        file_path = os.path.join(saved_frames, filename)
                        matched_images.append(file_path)

        return matched_images

    def _process_scenes(self):
        self._generate_scenes_descriptor()
