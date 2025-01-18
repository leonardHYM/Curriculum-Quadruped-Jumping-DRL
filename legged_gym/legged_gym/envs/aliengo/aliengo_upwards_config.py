# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from isaacgym.torch_utils import *
import numpy as np
import torch

class AliengoUpwardsCfg( LeggedRobotCfg ):
    additional_termination_conditions = True #False --> cannot trained successfully
    task_name = 'aliengo_upwards'
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.40] # x,y,z [m]
        # rel_foot_pos = [[0.551,0.551,-0.323,-0.323], # x
        #                 [0.138,-0.138,0.138,-0.138], # y
        #                 [-0.001,-0.001,-0.134,-0.134]]
        # rel_foot_pos = [[0.228,0.228,-0.253,-0.253], # x
        #                 [0.138,-0.138,0.138,-0.138], # y
        #                 [-0.465,-0.465,-0.465,-0.465]] # z  # relative to the COM pos        
        # new default standing position:
        rel_foot_pos = [[0.239,0.239,-0.301,-0.302], # x
                        [0.154,-0.154,0.154,-0.154], # y
                        [-0.305,-0.305,-0.299,-0.299]] # z  # relative to the COM pos [FL FR RL RR]
        # original position during training:
        # rel_foot_pos = [[0.228,0.228,-0.253,-0.253], # x
        #                 [0.138,-0.138,0.137,-0.137], # y
        #                 [-0.465,-0.465,-0.465,-0.465]] # z  # relative to the COM pos
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.06,  # [rad]
            "RL_hip_joint": 0.06,  # [rad]
            "FR_hip_joint": -0.06,  # [rad]
            "RR_hip_joint": -0.06,  # [rad]

            "FL_thigh_joint": 0.9,  # [rad]
            "RL_thigh_joint": 1.1,  # [rad]
            "FR_thigh_joint": 0.9,  # [rad]
            "RR_thigh_joint": 1.1,  # [rad]

            "FL_calf_joint": -1.8,  # [rad]
            "RL_calf_joint": -1.8,  # [rad]
            "FR_calf_joint": -1.8,  # [rad]
            "RR_calf_joint": -1.8,  # [rad]
        }
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.05,   # [rad]
        #     'RL_hip_joint': 0.05,   # [rad]
        #     'FR_hip_joint': -0.05 ,  # [rad]
        #     'RR_hip_joint': -0.05,   # [rad]

        #     'FL_thigh_joint': 0.8,     # [rad]
        #     'RL_thigh_joint': 1.,   # [rad]
        #     'FR_thigh_joint': 0.8,     # [rad]
        #     'RR_thigh_joint': 1.,   # [rad]

        #     'FL_calf_joint': -1.5,   # [rad]
        #     'RL_calf_joint': -1.5,    # [rad]
        #     'FR_calf_joint': -1.5,  # [rad]
        #     'RR_calf_joint': -1.5,    # [rad]
        # }

        K_HIP = 10.0
        K_THIGH = 16.0
        K_CALF = 20.0
        # damping
        D_HIP = 0.1
        D_THIGH = 0.1
        D_CALF = 0.1
       
        DEFAULT_HIP_ANGLE = 0.06
        DEFAULT_THIGH_ANGLE = 1.0
        DEFAULT_CALF_ANGLE = -1.8

        spring_stiffness = torch.tensor([K_HIP,K_THIGH,K_CALF]).repeat(1,4)
        spring_damping = torch.tensor([D_HIP,D_THIGH,D_CALF]).repeat(1,4)
        spring_rest_pos = torch.tensor([DEFAULT_HIP_ANGLE,DEFAULT_THIGH_ANGLE,DEFAULT_CALF_ANGLE]).repeat(1,4)

    # class init_state_slope( LeggedRobotCfg.init_state ):
    #     pos = [0.56, 0.0, 0.35] # x,y,z [m]
    #     default_joint_angles = { # = target angles [rad] when action = 0.0
    #         'FL_hip_joint': 0.03,   # [rad]
    #         'RL_hip_joint': 0.03,   # [rad]
    #         'FR_hip_joint': -0.03,  # [rad]
    #         'RR_hip_joint': -0.03,   # [rad]

    #         'FL_thigh_joint': 1.0,     # [rad]
    #         'RL_thigh_joint': 1.9,   # [rad]1.8
    #         'FR_thigh_joint': 1.0,     # [rad]
    #         'RR_thigh_joint': 1.9,   # [rad]

    #         'FL_calf_joint': -2.2,   # [rad]
    #         'RL_calf_joint': -0.9,    # [rad]
    #         'FR_calf_joint': -2.2,  # [rad]
    #         'RR_calf_joint': -0.9,    # [rad]
    #     }

    class viewer:
        camera_track_robot = False
        ref_env = 0
        pos = [5., -4, 0.32]  # [m]
        lookat = [5., 0, 0.32]  # [m]
        simulate_camera = False

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        distribution = "uniform" # "uniform" or "gaussian"
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_vel = 1.0
            contacts_noise_prob = 0.3

    class env( LeggedRobotCfg.env ):
        episode_length_s = 3 # episode length in seconds
        use_state_history = True
        state_history_length = 20
        state_stored_length = 5

        jump_type = "upward" # "upward" or "forward" or "forward_with_obstacles"

        continuous_jumping = False# # if true: the robot states are not reset after each jump.
        continuous_jumping_reset_probability = 0.9
        use_springs = False
        reset_height = 0.02#0.15 # [m]
        reset_landing_error = 0.2 # [in m]

        debug_draw = False
        throttle_to_real_time = False

        reset_orientation_error = 0.8 # [rad]

        known_contact_feet = True
        known_height = False
        jumping_target = True
        pass_remaining_time = False
        pass_has_jumped = True
        known_quaternion = True
        known_ori_error = False   
        known_error_quaternion = False     
        object_information = True

        num_observations = 0
        if not use_state_history:
            state_history_length = 1

        # Num observations is lin and ang vel (each 3*state_history_length), joint angles (12*state_history_length), 
        # and joint velocities (12*state_history_length) 
        # and action (12*state_history_length) 
        num_observations += 2*3*state_history_length  + 3*12*state_history_length
        
        # + orientation (quaternion*state_history_length)
        if known_quaternion:
            num_observations += 4*state_history_length
        if known_ori_error:
            num_observations += 1*state_history_length
        if known_error_quaternion:
            num_observations += 4*state_history_length
        # And and height if it's known (1*state_history_length)
        if known_height:
            num_observations += 1*state_history_length
        # And desired jumping position (relative) and quat if added:
        if jumping_target:
            num_observations += 7 + 6

        if pass_remaining_time: # Pass remaining time in the episode (one scalar value)
            num_observations += 1

        if pass_has_jumped: # Pass has_jumped bool (one scalar value)
            num_observations += 1
        
        if known_contact_feet: # Pass contact state at the feet (4*state_history_length)
            num_observations += 4*state_history_length
        
        # num_observations += 187 # If measuring height of terrain

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.20}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5 # Reduce the hip action scaling factor
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        use_action_filter = True
        filter_freq = 5.0 # [Hz]

        # action_filter_alpha = 0.5
        filter_type = "EMA" # or "EMA" or "butterworth"
        butterworth_order = 2

        safety_clip_actions = True # Clip ALL actions whenever ANY joint approaches its limit.

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf' #a1/urdf/a1.urdf'#
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True # aliengo and a1 must be True
        fix_base_link = False

        armature = 0.0
        use_physx_armature = False

    class domain_rand ( LeggedRobotCfg.domain_rand ): # modified Reference State Initialization(RSI)
        push_robots = True
        push_interval_s = 1.
        max_push_vel_xy = 1.

        # Randomise some of the initial robot states:
        pos_vel_random_prob = 0.7

        push_upwards = False
        push_upwards_prob = 0.5

        randomize_base_com = False
        added_com_range = [-0.2, 0.2]        
        randomize_robot_pos = False #True #
        randomize_robot_vel = False#True #
        randomize_robot_ori = True
        randomize_dof_pos = True
        randomize_spring_params = True
        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]
        randomize_PD_gains = True

        randomize_has_jumped = True#True # Randomize if the robot has jumped or not at the start of the episode
        has_jumped_random_prob = 0.8
        reset_has_jumped = True # Whether to reset the has_jumped to False at a random point of the episode
        manual_has_jumped_reset_time = 0 # Manual time at which to reset has_jumped (in steps)
        # Curriculum:
        curriculum = False # Enable initial state randomisation curriculum
        rand_vel_interval_ep = 5 # How many episodes until curriculum is changed
        # At the first flight step, set the velocity of the agent to the desired one
        push_towards_goal = False
        sim_latency = True
        base_latency = 0 # in ms
        sim_pd_latency = False

        lag_timesteps = 6
        randomize_lag_timesteps = False

        randomize_motor_offset = True

        randomize_base_mass = True
        randomize_com = False#True
        randomize_restitution = True
        randomize_link_mass = True

        randomize_joint_friction = True
        randomize_joint_damping = True

        randomize_joint_armature = False

        randomize_gravity = False
        gravity_rand_interval_s = 4.0 # in seconds
        gravity_impulse_duration = 0.99 # in seconds

        # Initial state randomisation ranges:
        class ranges():
            min_robot_vel = [-0.0, -0.0,-0.5] # min velocity [m/s] 
            max_robot_vel = [0.0,0.0,3.0] # max velocity [m/s] 

            min_robot_pos = [-0.0, 0.0, 0.0] # min robot pos [m] (relative to base height 0.32)
            max_robot_pos = [-0.0, 0.0, 0.7] # 0.7 # min robot pos [m] (relative to base height 0.32)

            # At each curriculum step the range changes by the increment step:
            pos_variation_increment = [0.0,0.005]
            vel_variation_increment = [0.01,0.01]
            min_ori_euler = [-0.0,-0.0, -0.1]
            max_ori_euler = [ 0.0, 0.0, 0.1]
            ori_variation_increment = [0.001,0.001,0.01]

            spring_stiffness_percentage = 0.3 # in percentages
            spring_damping_percentage = 0.3 # in percentages
            spring_rest_pos_range = [-0.1,0.1] # in rads

            motor_strength_ranges = [0.9,1.1] # in percentages of nominal values
            p_gains_range = [0.9,1.1] # percentages of nominal values
            d_gains_range = [0.9,1.1] # percentages of nominal values

            added_mass_range = [-1.0,3.0]

            # Latency to simulate (sampled at start of episode):
            latency_range = [0.0,40.0] # Latency to simulate at the observations -> policy level (in ms)
            pd_latency_range = [0.0,0.0] # Latency to simulate at the low-level PD controller (in ms)

            additional_latency_range = [-5.0,5.0] # Additional random latency to add to the latency at each step

            motor_offset_range = [-0.02, 0.02] # Offset to add to the motor angles

            restitution_range = [0.0, 0.4]
            friction_range = [0.01, 3.0]
            
            com_displacement_range = [-0.1, 0.1]

            joint_friction_range = [0.0, 0.04]
            joint_damping_range = [0.0, 0.01]

            added_link_mass_range = [0.7,1.3] # Factor

            joint_armature_range = [0.0,0.0] # Factor

            gravity_range = [-1.,1.]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35#0.25
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        class scales():
            #---------- Task rewards (once per episode): ----------- #

            # Rewards for reaching desired pose upon landing:
            task_pos = -0.#200.0 # Final reward is scale * (dt*decimation)
            task_ori = 100.0
            task_max_height = 4000.0 # Reward for maximum height (minus minimum height) achieved
            max_height = 400.

            termination = -20.
            jumping = 200#50.

            #---------- Continuous rewards (at every time step): ----------- #

            # Rewards for maintaining desired pose AFTER landing:
            post_landing_pos = 3. # Reward for maintaining landing position
            post_landing_ori = 3. # Reward for returning to desired orientation after landing

            base_height_flight = 10.#160.#80. # Reward for being in the air, only active the first jump
            base_height_stance = 5.#5. # Reward fo

            tracking_lin_vel = 5.0 # Reward for tracking desired linear velocity
            tracking_ang_vel = 0.5 # Reward for tracking desired angular velocity
            symmetric_joints = -3. # Reward for symmetric joint angles b/w left and right legs
            default_pose = -5#6. # Reward for staying close to default pose post landing
            feet_distance = -20.0 # Reward for keeping feet close to the body in flight

            #---------- Regularisation rewards: ----------- #

            energy_usage_actuators = -1e-2 # Additional energy usage penalty for the actuators.
            torques_actuators = -0.0 # Penalty for large torques

            base_acc = -1e-6 # Penalty for large base acceleration
            change_of_contact = 0.0 # Reward for maintaining contact state
            early_contact = 5.0 # Reward for maintaining contact with ground early in the episode
            feet_contact_forces = -0.0#-5 # Penalty for large contact forces at the feet
            action_rate = -0.2 # Penalty for large change of actions
            action_rate_second_order = -0.0 # Penalty for large change of action rate

            dof_vel = -0.0
            dof_acc = -1e-6
            dof_jerk = -0.0

            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -0.0

            f_hip_motion = 0.#-0.06
            r_hip_motion = 0.#-0.06
            f_thigh_motion = 0.#-0.04
            r_thigh_motion = 0.#-0.06
            f_calf_motion = 0.#-0.04
            r_calf_motion = 0.#-0.06
            

            
        # Exponential kernel sigma coefficients
        command_pos_tracking_sigma = 0.05
        post_landing_pos_tracking_sigma = 0.001
        command_ori_tracking_sigma = 0.05

        flight_reward_sigma = 0.1
        max_height_reward_sigma = 0.05
        squat_reward_sigma = 0.001
        stance_reward_sigma = 0.005
        dof_pos_sigma = 0.1

        vel_tracking_sigma = 0.1

        # penalty_sigma = 0.5
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 1.0

        sigma_rew_neg = 0.2
        sigma_neg_rew_curriculum = False
        sigma_neg_rew_curriculum_duration = 1000 # in update steps (= 5*episodes)
        sigma_neg_rew_initial_duration = 1000 # Keep at initial value for this many update steps

        max_contact_force = 150.0

    class normalization( LeggedRobotCfg.normalization ):
        clip_actions = 100.

    class commands():
        jump_over_box = False
        num_commands = 13 # default: relative x,y,z for jump and desired quaternion (euler angles use xyz notation)
        # and 6 for centre of object and its dimensions.
        upward_jump_probability = 0.1
        curriculum = False
        curriculum_type = "time-based"
        randomize_commands = True
        curriculum_interval = 5
        max_curriculum = 1.

        num_levels = 11
        randomize_yaw = False
        
        class ranges(): 
            # The command distances are relative to the initial agent position and are sampled from
            # the ranges below:

            # This is the min/maximum ranges in the jump's distance curriculum (x_des = dx~pos_dx + x)
            pos_dx_lim = [-0.0,0.0]
            pos_dy_lim = [-0.0,0.0]
            pos_dz_lim = [-0.0,0.0]
            # These are the starting ranges for the jump's distances (i.e. if curriculum 
            # if off, these stay the same for the whole training.)
            pos_dx_ini = [0.0,0.0]
            pos_dy_ini = [-0.0,0.0]
            pos_dz_ini = [0.0,0.0]
            # These are the steps for the jump distance changes every curriculum update.
            pos_variation_increment = [0.01,0.01,0.01]

        class distances(): # If you want to pass a fixed command distance, use these values:
            x = 0.0
            y = 0.0
            z = 0.0
            # Specify in Euler 'xyz' notation (order is important as it gets converted to quat later):
            # des_angles_euler = [0.0,0.0,0.0]
            des_yaw = None

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        selected = False
        curriculum = False
        make_terrain_uneven = False
        sloped_terrain_number = 5
        slope_range = [0.0,0.2]

        # terrain_kwargs =  { 'type': 'discrete_obstacles_terrain', 'max_height': 0.1, 'min_size': 0.2,\
        #     'max_size':0.5,'num_rects':10, 'platform_size': 3.}
        # terrain_kwargs = { 'type':'stairs_terrain', 'step_width': 1, 'step_height':0.2}
        terrain_kwargs = {'type':'box_terrain', 'box_width': 1.5, 'box_height':0.6, 'box_length':0.5,'make_uneven':make_terrain_uneven }

        num_zero_height_terrains = 10
        terrain_difficulty_height_range = [0.0, 0.3]
        terrain_difficulty_width_range = [0.6,1.2]
        
        measure_heights = False
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
        # terrain_length = 8.
        # terrain_width = 8.
        num_rows = 30
        num_cols = 30
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        # max_init_terrain_level = 4

        static_friction = 1.
        dynamic_friction = 1.
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            solver_type = 1 # 0: pgs

class AliengoUpwardsCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        max_grad_norm = 1.0
        clip_param = 0.2 # PPO clip parameter
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'aliengo_upwards' # for saving and loading training results, which is subject to personal design.
        num_steps_per_env = 24 # Try 30?
  
