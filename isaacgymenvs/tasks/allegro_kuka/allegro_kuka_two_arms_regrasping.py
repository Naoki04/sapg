# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List, Tuple

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_two_arms import AllegroKukaTwoArmsBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import tolerance_curriculum, tolerance_successes_objective


class AllegroKukaTwoArmsRegrasping(AllegroKukaTwoArmsBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.goal_object_indices = []
        self.goal_asset = None

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _object_keypoint_offsets(self):
        """Regrasping task uses only a single object keypoint since we do not care about object orientation."""
        return [[0, 0, 0]]

    def _load_additional_assets(self, object_asset_root, arm_y_offset: float):
        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.disable_gravity = True
        self.goal_asset = self.gym.load_asset(
            self.sim, object_asset_root, self.asset_files_dict["ball"], goal_asset_options
        )
        goal_rb_count = self.gym.get_asset_rigid_body_count(self.goal_asset)
        goal_shapes_count = self.gym.get_asset_rigid_shape_count(self.goal_asset)
        return goal_rb_count, goal_shapes_count

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        goal_start_pose = gymapi.Transform()
        goal_asset = self.goal_asset
        goal_handle = self.gym.create_actor(
            env_ptr, goal_asset, goal_start_pose, "goal_object", env_idx + self.num_envs, 0, 0
        )
        self.gym.set_actor_scale(env_ptr, goal_handle, 0.5)
        self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
        goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
        for name in self.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            self.rigid_body_name_to_idx['goal/' + name] = self.gym.find_actor_rigid_body_index(env_ptr, goal_handle, name, gymapi.DOMAIN_ENV)
        self.goal_object_indices.append(goal_object_idx)

    def _after_envs_created(self):
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def _reset_target(self, env_ids: Tensor, tensor_reset=True) -> None:
        # sample random target location in some volume
        if tensor_reset:
            target_volume_origin = self.target_volume_origin
            target_volume_extent = self.target_volume_extent

            target_volume_min_coord = target_volume_origin + target_volume_extent[:, 0]
            target_volume_max_coord = target_volume_origin + target_volume_extent[:, 1]
            target_volume_size = target_volume_max_coord - target_volume_min_coord

            rand_pos_floats = torch_rand_float(0.0, 1.0, (len(env_ids), 3), device=self.device)
            target_coords = target_volume_min_coord + rand_pos_floats * target_volume_size

            # let the target be close to 1st or 2nd arm, randomly
            left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            x_ofs = 0.75
            x_pos = torch.where(
                left_right_random > 0,
                x_ofs * torch.ones_like(left_right_random),
                -x_ofs * torch.ones_like(left_right_random),
            )

            target_coords[:, 0] += x_pos.squeeze(dim=1)

            self.goal_states[env_ids, 0:3] = target_coords

            self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]

            # we also reset the object to its initial position
        self.reset_object_pose(env_ids, tensor_reset=tensor_reset)

        # since we put the object back on the table, also reset the lifting reward
        if tensor_reset:
            self.lifted_object[env_ids] = False

        self.deferred_set_actor_root_state_tensor_indexed(
            [self.object_indices[env_ids], self.goal_object_indices[env_ids]]
        )

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return [self.goal_object_indices[env_ids]]

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        rew_buf, is_success = super().compute_kuka_reward()
        return rew_buf, is_success

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective

    def _extra_curriculum(self):
        self.success_tolerance, self.last_curriculum_update = tolerance_curriculum(
            self.last_curriculum_update,
            self.frame_since_restart,
            self.tolerance_curriculum_interval,
            self.prev_episode_successes,
            self.success_tolerance,
            self.initial_tolerance,
            self.target_tolerance,
            self.tolerance_curriculum_increment,
        )
    
    def update_rigid_body_state_dict(self, state_ddict, env_idx=-1):
        """
        state_ddict: defaultdict of the form Dict[str, List[Tensor]]
        """
        print(self.object_asset_scales[env_idx])
        isaacgym_to_blender_name = {
            'allegro/iiwa7_link_0': 'link_0',
            'allegro/iiwa7_link_1': 'link_1',
            'allegro/iiwa7_link_2': 'link_2',
            'allegro/iiwa7_link_3': 'link_3',
            'allegro/iiwa7_link_4': 'link_4',
            'allegro/iiwa7_link_5': 'link_5',
            'allegro/iiwa7_link_6': 'link_6',
            'allegro/iiwa7_link_7': 'link_7',
            'allegro1/iiwa7_link_0': 'link_0.002',
            'allegro1/iiwa7_link_1': 'link_1.002',
            'allegro1/iiwa7_link_2': 'link_2.002',
            'allegro1/iiwa7_link_3': 'link_3.002',
            'allegro1/iiwa7_link_4': 'link_4.002',
            'allegro1/iiwa7_link_5': 'link_5.002',
            'allegro1/iiwa7_link_6': 'link_6.002',
            'allegro1/iiwa7_link_7': 'link_7.002',
            #'allegro/iiwa7_link_ee': 'link_ee',
            'allegro/allegro_mount': 'allegro_mount',
            'allegro1/allegro_mount': 'allegro_mount.002',
            'allegro/palm_link': 'base_link',
            'allegro1/palm_link': 'base_link.002',
            
            'allegro/index_link_0': 'primary_base',
            'allegro/index_link_1': 'primary_proximal',
            'allegro/index_link_2': 'primary_medial',
            'allegro/index_link_3': 'touch_sensor_base',
            'allegro/middle_link_0': 'primary_base.001',
            'allegro/middle_link_1': 'primary_proximal.001',
            'allegro/middle_link_2': 'primary_medial.001',
            'allegro/middle_link_3': 'touch_sensor_base.001',
            'allegro/ring_link_0': 'primary_base.002',
            'allegro/ring_link_1': 'primary_proximal.002',
            'allegro/ring_link_2': 'primary_medial.002',
            'allegro/ring_link_3': 'touch_sensor_base.002',
            
            'allegro1/index_link_0': 'primary_base.006',
            'allegro1/index_link_1': 'primary_proximal.006',
            'allegro1/index_link_2': 'primary_medial.006',
            'allegro1/index_link_3': 'touch_sensor_base.006',
            'allegro1/middle_link_0': 'primary_base.007',
            'allegro1/middle_link_1': 'primary_proximal.007',
            'allegro1/middle_link_2': 'primary_medial.007',
            'allegro1/middle_link_3': 'touch_sensor_base.007',
            'allegro1/ring_link_0': 'primary_base.008',
            'allegro1/ring_link_1': 'primary_proximal.008',
            'allegro1/ring_link_2': 'primary_medial.008',
            'allegro1/ring_link_3': 'touch_sensor_base.008',
            
            'allegro/thumb_link_0': 'thumb_base',
            'allegro/thumb_link_1': 'thumb_proximal',
            'allegro/thumb_link_2': 'thumb_medial',
            'allegro/thumb_link_3': 'touch_sensor_thumb_base',
            'allegro1/thumb_link_0': 'thumb_base.002',
            'allegro1/thumb_link_1': 'thumb_proximal.002',
            'allegro1/thumb_link_2': 'thumb_medial.002',
            'allegro1/thumb_link_3': 'touch_sensor_thumb_base.002',
            
            'object/object': 'cube_multicolor',
            'table/box': 'cube', # 0.475 0.4 0.3
            'goal/ball': 'ball'
            
        }
        
        for key, value in isaacgym_to_blender_name.items():
            rigid_body_idx = self.rigid_body_name_to_idx[key]
            pose = self.rigid_body_states[env_idx, rigid_body_idx,0:7].cpu().numpy()
            import transforms3d
            
            pos = pose[0:3]
            quat = pose[3:7]
            rot = transforms3d.euler.quat2euler([quat[3], quat[0], quat[1], quat[2]])
            
            state_ddict[value].append((pos, rot))
