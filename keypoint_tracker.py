import multiprocessing as mp
import sys
import os

import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift

sys.path.append(os.path.dirname(__file__))
from utils.utils import filter_points_by_bounds

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

class KeypointTrackerProcess():
    def __init__(self, config):
        self._config = config
        self._device = torch.device(self._config['device'])
        self._dinov2 = None # Set in the new process
        self._bounds_min = np.array(self._config['bounds_min'])
        self._bounds_max = np.array(self._config['bounds_max'])
        self._mean_shift = MeanShift(bandwidth=self._config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self._patch_size = 14  # dinov2
        np.random.seed(self._config['seed'])
        torch.manual_seed(self._config['seed'])
        torch.cuda.manual_seed(self._config['seed'])

        # multiprocessing
        self._data_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._stop_event = mp.Event()
        self._process = None

    def start(self):
        self._process = mp.Process(target=self._run)
        print("Starting keypoint tracker process")
        self._process.start()

    def stop(self):
        print("Stopping keypoint tracker process")
        self._stop_event.set()
        self._process.join()
        print("Keypoint tracker process stopped")

    def send(self, data):
        '''
        data: dict
        {
                'rgb': np.ndarray, [H, W, 3]
                'points': np.ndarray, [N, 3]
                'masks': np.ndarray, [H, W]
        }
        '''
        self._data_queue.put(data)

    def get(self):
        '''
        return a dict as 
        {
            'keypoints': np.ndarray, [N, 3]
            'projected': np.ndarray, [H, W, 3]
        }
        '''
        return self._result_queue.get()
    
    def _run(self):
        self._dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self._device)
        while not self._stop_event.is_set():
            try:
                data = self._data_queue.get(timeout=1)
            except:
                # print(YELLOW + f"[KeypointTracker]: No data received in 1 seconds." + RESET)
                continue
            rgb, points, masks = data["rgb"], data["points"], data["masks"]
            keypoints, projected = self._get_keypoints(rgb, points, masks)
            result = {
                'keypoints': keypoints,
                'projected': projected,
            }
            self._result_queue.put(result)

    def _get_keypoints(self, rgb, points, masks):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)
        # for each mask, cluster in feature space to get meaningful regions, and uske their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat, masks)
        # exclude keypoints that are outside of the workspace
        within_space = filter_points_by_bounds(candidate_keypoints, self._bounds_min, self._bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]
        # merge close points by clustering in cartesian space
        merged_indices = self._merge_clusters(candidate_keypoints)
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]
        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]
        # project keypoints to image space
        projected = self._project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat)
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        # convert masks to binary masks
        masks = [masks == uid for uid in np.unique(masks)]
        # ensure input shape is compatible with dinov2
        H, W, _ = rgb.shape
        patch_h = int(H // self._patch_size)
        patch_w = int(W // self._patch_size)
        new_H = patch_h * self._patch_size
        new_W = patch_w * self._patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
        # shape info
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info
    
    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        projected = rgb.copy()
        # overlay keypoints on the image
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # draw a box
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # draw text
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self._dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim]
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # float32 [H*W, feature_dim]
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        for rigid_group_id, binary_mask in enumerate(masks):
            # ignore mask that is too large
            if np.mean(binary_mask) > self._config['max_mask_ratio']:
                continue
            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch  = (feature_points_torch - feature_points_torch.min(0)[0]) / (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # cluster features to get meaningful regions
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self._config['num_candidates_per_mask'],
                distance='euclidean',
                device=self._device,
            )
            cluster_centers = cluster_centers.to(self._device)
            for cluster_id in range(self._config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        self._mean_shift.fit(candidate_keypoints)
        cluster_centers = self._mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices