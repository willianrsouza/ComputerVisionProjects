import time

import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import regex as re


class ImageSearch:
    def __init__(self):
        self.movie_path = '/src/movies/Pulp-Fiction-1994.mp4'
        self.max_features_per_frame = 2_000_000_000
        self.k = 1024
        self.dim = 300

    def process_search_images(self):
        self._extract_descriptor(
            frames='/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/search',
            features_directory='/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/search_features'
        )
        time.sleep(3)
        self._bag_visual_word_search()
        time.sleep(3)
        results = self.search()
        print(results)
        time.sleep(3)
        return self.get_images(results)

    def get_images(self, frames_to_pick: list[dict]):
        base_path = os.path.dirname(__file__)
        saved_frames = os.path.abspath(os.path.join(base_path, './saved_frames'))

        matched_images = []

        for frame_info in frames_to_pick:
            frame_number = str(frame_info['frame_number'])  # Ensure frame_number is a string
            distance = frame_info['distance']

            for filename in os.listdir(saved_frames):
                number_match = re.search(r'\d+', filename)

                if number_match and number_match.group() == frame_number:
                    file_path = os.path.join(saved_frames, filename)
                    matched_images.append(file_path)

        return matched_images

    def _bag_visual_word_search(self):
        features_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/search_features'
        cluster_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/cluster'
        bow_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/search_bow'

        k = 1024

        # Load cluster centroids from the file
        cluster_filepath = os.path.join(cluster_directory, 'cluster.cluster')
        centroids = np.loadtxt(cluster_filepath)

        # Create a KNN model
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(centroids)

        # Iterate over all files in the features directory
        for filename in os.listdir(features_directory):
            if filename.endswith('.sift'):
                features_filepath = os.path.join(features_directory, filename)

                # Read descriptors from the file
                with open(features_filepath, 'r') as file:
                    lines = file.readlines()
                    # Extract descriptors from the lines (assuming descriptors start after 'Descriptors:')
                    descriptors = np.loadtxt(lines[0:])

                # Compute the nearest centroid index for each descriptor
                distances, indices = knn.kneighbors(descriptors)
                # Count the occurrences of each centroid
                unique_indices, counts = np.unique(indices, return_counts=True)
                # List to store the counts for each centroid
                centroid_counts = [0] * k

                for idx, item in enumerate(unique_indices):
                    centroid_counts[item] += counts[idx]

                # Save the centroid counts to a file in a single line
                bow_filepath = os.path.join(bow_directory, f"{os.path.splitext(filename)[0]}.sift.bow")
                with open(bow_filepath, 'w') as bow_file:
                    bow_file.write(' '.join(map(str, centroid_counts)))

                print(f"Bag of Visual Words for {filename} saved as {bow_filepath}")

    def search(self):
        bow_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/bow'
        target_file = '/src/base_model/search_bow/cropped_image.sift.bow'

        target_content = np.loadtxt(target_file)

        all_content = []
        all_filenames = []

        for filename in os.listdir(bow_directory):
            if filename.endswith('.sift.bow'):
                filepath = os.path.join(bow_directory, filename)
                content = np.loadtxt(filepath)

                all_content.append(content)
                all_filenames.append(filename)

        knn_model = NearestNeighbors(n_neighbors=3)
        knn_model.fit(np.vstack(all_content))

        distances, indices = knn_model.kneighbors(target_content.reshape(1, -3), 3)

        files = []

        for i, distance in enumerate(distances.flatten()):
            filename = all_filenames[indices.flatten()[i]]
            frame_number_match = re.search(r'\d+', filename)

            if frame_number_match:
                frame_number = frame_number_match.group()
                files.append({'frame_number': frame_number, 'frame': filename, 'distance': distance})

        return files

    def process_movie_frames(self):
       # self._cut_scenes()
        #time.sleep(3)
        #self._extract_descriptor()
       # time.sleep(3)
        self._clustering()
        time.sleep(3)
        self._bag_visual_word()

        print('Done.')

    def _cut_scenes(self):
        output_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/saved_frames/'
        captures = cv2.VideoCapture(self.movie_path)

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

    def _extract_descriptor(self,
                            frames_format: str = '.jpg',
                            frames: str = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/saved_frames',
                            features_directory: str = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/features'):

        sift = cv2.SIFT_create()

        for filename in os.listdir(frames):
            if filename.endswith(frames_format):
                frame_path = os.path.join(frames, filename)
                img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

                _, descriptors = sift.detectAndCompute(img, None)

                if descriptors is not None:
                    np.random.shuffle(descriptors)

                    if len(descriptors) > self.max_features_per_frame:
                        descriptors = descriptors[:self.max_features_per_frame]

                    features_filename = os.path.splitext(filename)[0] + '.sift'
                    features_filepath = os.path.join(features_directory, features_filename)

                    with open(features_filepath, 'w') as file:
                        np.savetxt(file, descriptors, fmt='%f')

                    print(f"Features for {filename} saved as {features_filepath}")

    def _clustering(self):
        features_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/features'
        cluster_directory = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/cluster'

        # List to store all descriptors
        all_descriptors = []

        # Iterate over all files in the features directory
        for filename in os.listdir(features_directory):
            if filename.endswith('.sift'):
                features_filepath = os.path.join(features_directory, filename)

                # Read descriptors from the file
                with open(features_filepath, 'r') as file:
                    lines = file.readlines()
                    # Extract descriptors from the lines (assuming descriptors start after 'Descriptors:')
                    descriptors = [np.fromstring(line, dtype=float, sep=' ') for line in lines[2:] if line.strip()]
                    all_descriptors.extend(descriptors)

        # Convert the list of descriptors to a NumPy array
        all_descriptors = np.array(all_descriptors)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans.fit(all_descriptors)

        # Save cluster centroids to a file
        cluster_filepath = os.path.join(cluster_directory, 'cluster.cluster')
        np.savetxt(cluster_filepath, kmeans.cluster_centers_, fmt='%f')

        print(f"Cluster centroids saved to {cluster_filepath}")

    def _bag_visual_word(
            self,
            features_directory: str = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/features',
            bow_directory: str = '/Users/willian/Documents/personal-projects/ComputerVisionProjects/src/bow'
    ):
        cluster_filepath = '/src/base_model/cluster/cluster.cluster'
        centroids = np.loadtxt(cluster_filepath)

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(centroids)

        for filename in os.listdir(features_directory):
            if filename.endswith('.sift'):
                features_filepath = os.path.join(features_directory, filename)

                # Read descriptors from the file
                with open(features_filepath, 'r') as file:
                    lines = file.readlines()
                    # Extract descriptors from the lines (assuming descriptors start after 'Descriptors:')
                    descriptors = np.loadtxt(lines[0:])

                distances, indices = knn.kneighbors(descriptors)

                # Count the occurrences of each centroid
                unique_indices, counts = np.unique(indices, return_counts=True)

                # List to store the counts for each centroid
                centroid_counts = [0] * self.k

                for idx, item in enumerate(unique_indices):
                    centroid_counts[item] += counts[idx]

                # Save the centroid counts to a file in a single line
                bow_filepath = os.path.join(bow_directory, f"{os.path.splitext(filename)[0]}.sift.bow")

                with open(bow_filepath, 'w') as bow_file:
                    bow_file.write(' '.join(map(str, centroid_counts)))

                print(f"Bag of Visual Words for {filename} saved as {bow_filepath}")

    def _open_image(self, path: str):
        image = cv2.imread(path)

        if image is None:
            print('Image Not Found')
        else:
            print('Press button: 0 to close.')
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    model = ImageSearch()
    model.process_search_images()

