from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    class noise:
        add_noise = True
        noise_level = 0.6

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1
        
    class env(LeggedRobotCfg.env):
        
        frame_stack = 15
        C_frame_stack = 3
        num_single_obs =47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        num_privileged_obs = int(C_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 114514
        episode_length_s = 24
        use_ref_actions = False

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        currculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02
      
    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        
        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"

        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2


    class rewards:
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700
        
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            vel_mismatch_exp = 0.5
            low_speed = 0.2
            track_vel_hard = 0.5

            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2

            feet_air_time = 0.0
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2

            feet_contact_forces = -0.01

            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            

            base_height = -10.0
            base_acc = 0.2
            orientation = -1.0
            default_joint_pos = 0.5

            torques = -1e-5
            action_smoothness = -0.002
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            collision = 0.0
            
            
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

        class normalization:
            class obs_scales:
                lin_vel = 2.
                ang_vel = 1.
                dof_pos = 1.
                dof_vel = 0.05
                quat = 1.
                height_measurements = 5.0
            clip_observations = 18.
            clip_actions = 18.

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 5
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = 'PPO'
        max_iterations = 3001
        num_step_per_env = 60

        save_interval = 100
        run_name = ''
        experiment_name = 'g1'

        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

  
