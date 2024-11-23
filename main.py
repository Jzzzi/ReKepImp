import os
import sys
import json
import time
import torch
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))
# from utils.utils import get_config, GREEN, RESET, get_callable_grasping_cost_fn, load_functions_from_txt, YELLOW, print_opt_debug_dict, get_linear_interpolation_steps, spline_interpolate_poses
from utils.utils import *
from enviroment import RealEnviroment
from constraint_generator import ConstraintGenerator
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
# from visualizer import Visualizer

class Main:
    def __init__(self, visualize=False):
        self._global_config = get_config()
        self._main_config = self._global_config['main']
        self._bounds_min = np.array(self._global_config['bounds_min'])
        self._bounds_max = np.array(self._global_config['bounds_max'])
        self._visualize = visualize
        # set random seed
        np.random.seed(self._global_config['seed'])
        torch.manual_seed(self._global_config['seed'])
        torch.cuda.manual_seed(self._global_config['seed'])
        # constraint generator
        self._constraint_generator = ConstraintGenerator(self._global_config['constraint_generator'])
        # initialize environment
        self._env = RealEnviroment(self._global_config, visualize=True)

        # initialize solvers
        self._subgoal_solver = SubgoalSolver(self._global_config['subgoal_solver'])
        # TODO set the path solver
        self._path_solver = PathSolver(self._global_config['path_solver'])
        # initialize visualizer
        # if self._visualize:
        #     self._visualizer = Visualizer()

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        self._env.reset()
        # seems to be no such key in the config
        # rgb = cam_obs[self.config['vlm_camera']]['rgb']
        # points = cam_obs[self.config['vlm_camera']]['points']
        # mask = cam_obs[self.config['vlm_camera']]['seg']
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        try:
            if rekep_program_dir is None:
                observation = self._env.observe()
                keypoints = observation['keypoints']
                projected_img = observation['projected']
                #     self.visualizer.show_img(projected_img)
                metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
                rekep_program_dir = self._constraint_generator.generate(projected_img, instruction, metadata)
                print(GREEN + f"[Main]: Constraints generated in {rekep_program_dir}" + RESET)
            # ====================================
            # = execute
            # ====================================
            self._execute(rekep_program_dir, disturbance_seq)
        except Exception as e:
            print(f"{RED}[Main]: {e}{RESET}")
            self._env.stop()
        self._env.stop()

    # def _update_disturbance_seq(self, stage, disturbance_seq):
    #     if disturbance_seq is not None:
    #         if stage in disturbance_seq and not self._applied_disturbance[stage]:
    #             # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
    #             self.env.disturbance_seq = disturbance_seq[stage](self.env)
    #             self._applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self._program_info = json.load(f)
        # self._applied_disturbance = {stage: False for stage in range(1, self._program_info['num_stages'] + 1)}
        # register keypoints to be tracked
        # load constraints
        self._constraint_fns = dict()
        for stage in range(1, self._program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self._env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self._constraint_fns[stage] = stage_dict
        
        # bookkeeping of which keypoints can be moved in the optimization
        self._keypoint_movable_mask = np.zeros(self._program_info['num_keypoints'], dtype=bool)
        self._keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self._last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            observation = self._env.observe()
            self._keypoints = observation['keypoints']
            key2obj = observation['key2obj']
            for i in range(len(self._keypoints)):
                obj = key2obj[i]
                if self._env._objects_grasped[obj]:
                    self._keypoint_movable_mask[i] = False
            self._curr_ee_pose = self._env.get_ee_pose()
            self._curr_joint_pos = self._env.get_arm_joint_postions()
            self._sdf_voxels = self._env.get_sdf_voxels(self._main_config['sdf_voxel_size'])
            self._collision_points = self._env.get_collision_points()
            print(GREEN + f"[Main]: Stage {self._stage}" + RESET)
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self._stage > 1:
                path_constraints = self._constraint_fns[self._stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self._keypoints[0], self._keypoints)
                    if violation > self._main_config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self._stage - 1, 0, -1):
                    path_constraints = self._constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self._keypoints[0], self._keypoints)
                        if violation > self._main_config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(YELLOW + f"[Main]: backtrack to stage {new_stage}" + RESET)
                self._update_stage(new_stage)
            else:
                next_subgoal = self._get_next_subgoal(from_scratch=self._first_iter)
                print(GREEN + f"[Main]: Get next subgoal." + RESET)
                next_path = self._get_next_path(next_subgoal, from_scratch=self._first_iter)
                print(GREEN + f"[Main]: Get next path." + RESET)
                self._first_iter = False
                self._action_queue = next_path.tolist()
                self._last_sim_step_counter = self._env._step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while len(self._action_queue) > 0 and count < self._main_config['action_steps_per_iter']:
                    next_action = self._action_queue.pop(0)
                    precise = len(self._action_queue) == 0
                    self._env.execute_action(next_action, wait=True)
                    observation = self._env.observe()
                    cv2.imshow('Projected', cv2.cvtColor(observation['projected'], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    count += 1
                if len(self._action_queue) == 0:
                    if self._is_grasp_stage:
                        print(GREEN + "[Main]: Try to grasp" + RESET)
                        get_goal = True
                        # if subgoal constraint is satisfied, execute grasp action
                        sub_goal_constraints = self._constraint_fns[self._stage]['subgoal']
                        for constraints in sub_goal_constraints:
                            violation = constraints(self._keypoints[0], self._keypoints)
                            if violation > self._main_config['constraint_tolerance']:
                                # get_goal = False
                                break
                        if get_goal:
                            self._execute_grasp_action()
                        else:
                            continue
                    elif self._is_release_stage:
                        print(GREEN + "[Main]: Try to release" + RESET)
                        # if subgoal constraint is satisfied, execute release action
                        sub_goal_constraints = self._constraint_fns[self._stage]['subgoal']
                        get_goal = True
                        for constraints in sub_goal_constraints:
                            violation = constraints(self._keypoints[0], self._keypoints)
                            if violation > self._main_config['constraint_tolerance']:
                                # get_goal = False
                                break
                        if get_goal:
                            self._execute_release_action()
                        else:
                            continue
                    # if completed, save video and return
                    if self._stage == self._program_info['num_stages']: 
                        print(GREEN + "[Main]: Task completed, but Video saver not implemented" + RESET)
                        return
                    # progress to next stage
                    print(GREEN + f"[Main]: progress to stage {self._stage + 1}" + RESET)
                    self._update_stage(self._stage + 1)

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self._constraint_fns[self._stage]['subgoal']
        path_constraints = self._constraint_fns[self._stage]['path']
        subgoal_pose, debug_dict = self._subgoal_solver.solve(self._curr_ee_pose,
                                                            self._keypoints,
                                                            self._keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self._sdf_voxels,
                                                            self._collision_points,
                                                            self._is_grasp_stage,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self._is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self._main_config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self._stage
        print_opt_debug_dict(debug_dict)
        # if self._visualize:
        #     self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self._constraint_fns[self._stage]['path']
        path, debug_dict = self._path_solver.solve(self._curr_ee_pose,
                                                    next_subgoal,
                                                    self._keypoints,
                                                    self._keypoint_movable_mask,
                                                    path_constraints,
                                                    self._sdf_voxels,
                                                    self._collision_points,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        # if self._visualize:
        #     self._visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self._curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self._main_config['interpolate_pos_step_size'],
                                                    self._main_config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self._env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self._stage = stage
        self._is_grasp_stage = self._program_info['grasp_keypoints'][self._stage - 1] != -1
        self._is_release_stage = self._program_info['release_keypoints'][self._stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self._is_grasp_stage + self._is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self._is_grasp_stage:  # ensure gripper is open for grasping stage
            self._env.open_gripper()
        # clear action queue
        self._action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self._first_iter = True
        print(GREEN + f"[Main]: stage {self._stage} started" + RESET)

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self._keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self._env.get_object_by_keypoint(i)
            self._keypoint_movable_mask[i] = self._env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        self._env.close_gripper()
        time.sleep(0.5)
    
    def _execute_release_action(self):
        self._env.open_gripper()

if __name__ == "__main__":

    use_cached_query = False

    task = {
            'instruction': 'put the cup into the pannel',
            'rekep_program_dir': '/home/liujk/ReKepImp/vlm_query/2024-11-19_19-41-04_over_lap_the_red_cup_on_the_green_cup',
    }
    instruction = task['instruction']
    main = Main(visualize=False)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if use_cached_query else None,)