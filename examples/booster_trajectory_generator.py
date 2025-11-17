#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Booster T1 Humanoid Robot Trajectory Generator
# Generates self-collision-free trajectories for left or right arm

# Standard Library
import argparse
import time
from datetime import datetime
from typing import List, Tuple

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate self-collision-free trajectories for Booster T1 robot"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="booster_t1_left_arm.yml",
        choices=["booster_t1_left_arm.yml", "booster_t1_right_arm.yml"],
        help="Robot configuration file",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="booster_trajectories.npz",
        help="Output NPZ file path",
    )
    parser.add_argument(
        "--workspace_bounds",
        type=float,
        nargs=6,
        default=[-0.2, 0.5, -0.3, 0.5, 0.0, 0.6],
        metavar=("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
        help="Workspace bounds for random pose sampling (relative to robot base)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=50,
        help="Maximum attempts to find valid poses/trajectories per trajectory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_random_orientation",
        action="store_true",
        help="Use random orientations (otherwise uses fixed downward orientation)",
    )
    parser.add_argument(
        "--interpolation_dt",
        type=float,
        default=0.02,
        help="Interpolation timestep for trajectory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of trajectories to generate in parallel (batch processing)",
    )
    return parser.parse_args()


def generate_random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Generate a random unit quaternion using the algorithm from Shoemake 1992.
    
    Args:
        rng: Numpy random generator
        
    Returns:
        Random unit quaternion [qw, qx, qy, qz]
    """
    u = rng.uniform(0, 1, size=3)
    q = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
    ])
    # Return in [qw, qx, qy, qz] format
    return np.array([q[3], q[0], q[1], q[2]])


def sample_workspace_poses(
    workspace_bounds: List[float],
    num_poses: int,
    use_random_orientation: bool,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Sample random poses in workspace.
    
    Args:
        workspace_bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
        num_poses: Number of poses to sample
        use_random_orientation: If True, use random orientations, else fixed downward
        rng: Numpy random generator
        
    Returns:
        List of poses, each [x, y, z, qw, qx, qy, qz]
    """
    x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds
    
    poses = []
    for _ in range(num_poses):
        position = np.array([
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
            rng.uniform(z_min, z_max),
        ])
        
        if use_random_orientation:
            quaternion = generate_random_quaternion(rng)
        else:
            # Fixed downward orientation (gripper pointing down)
            quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # [qw, qx, qy, qz]
        
        pose = np.concatenate([position, quaternion])
        poses.append(pose)
    
    return poses


def filter_ik_solvable_poses(
    motion_gen: MotionGen,
    poses: List[np.ndarray],
    tensor_args: TensorDeviceType,
) -> List[Tuple[np.ndarray, torch.Tensor]]:
    """Filter poses that have valid IK solutions.
    
    Args:
        motion_gen: MotionGen instance
        poses: List of candidate poses [x, y, z, qw, qx, qy, qz]
        tensor_args: Tensor device configuration
        
    Returns:
        List of (pose, ik_solution) tuples for valid poses
    """
    valid_poses = []
    
    for pose in poses:
        # Convert to Pose object
        goal_pose = Pose(
            position=tensor_args.to_device([pose[:3]]),
            quaternion=tensor_args.to_device([pose[3:]]),
        )
        
        # Try to solve IK
        # Use a random seed state for IK
        seed_config = motion_gen.get_retract_config().view(1, -1)
        ik_result = motion_gen.ik_solver.solve_single(
            goal_pose,
            seed_config,
            num_seeds=30,
        )
        
        if ik_result.success.item():
            # Get the best solution
            ik_solution = ik_result.solution[0]
            valid_poses.append((pose, ik_solution))
    
    return valid_poses


def filter_ik_solvable_poses_batch(
    motion_gen: MotionGen,
    poses: List[np.ndarray],
    tensor_args: TensorDeviceType,
) -> List[Tuple[np.ndarray, torch.Tensor]]:
    """Filter poses that have valid IK solutions using batch processing.
    
    Args:
        motion_gen: MotionGen instance
        poses: List of candidate poses [x, y, z, qw, qx, qy, qz]
        tensor_args: Tensor device configuration
        
    Returns:
        List of (pose, ik_solution) tuples for valid poses
    """
    if len(poses) == 0:
        return []
    
    # Stack all poses into batch tensors
    positions = np.array([pose[:3] for pose in poses])
    quaternions = np.array([pose[3:] for pose in poses])
    
    # Create batch Pose object
    goal_pose_batch = Pose(
        position=tensor_args.to_device(positions),
        quaternion=tensor_args.to_device(quaternions),
    )
    
    # Solve IK for all poses in batch
    seed_config = motion_gen.get_retract_config().view(1, -1)
    ik_result = motion_gen.ik_solver.solve_batch(
        goal_pose_batch,
        seed_config.repeat(len(poses), 1),
        num_seeds=30,
    )
    
    # Filter successful solutions
    valid_poses = []
    for i, (pose, success) in enumerate(zip(poses, ik_result.success)):
        if success.item():
            valid_poses.append((pose, ik_result.solution[i]))
    
    return valid_poses


def generate_trajectory(
    motion_gen: MotionGen,
    start_state: JointState,
    goal_pose: Pose,
    plan_config: MotionGenPlanConfig,
) -> Tuple[bool, JointState]:
    """Generate a single trajectory.
    
    Args:
        motion_gen: MotionGen instance
        start_state: Starting joint state
        goal_pose: Goal pose
        plan_config: Planning configuration
        
    Returns:
        (success, interpolated_trajectory) tuple
    """
    result = motion_gen.plan_single(start_state, goal_pose, plan_config)
    
    if result.success.item():
        trajectory = result.get_interpolated_plan()
        return True, trajectory
    else:
        return False, None


def generate_trajectories_batch(
    motion_gen: MotionGen,
    start_states: JointState,
    goal_poses: Pose,
    plan_config: MotionGenPlanConfig,
) -> Tuple[torch.Tensor, List[JointState]]:
    """Generate multiple trajectories in parallel.
    
    Args:
        motion_gen: MotionGen instance
        start_states: Batch of starting joint states
        goal_poses: Batch of goal poses
        plan_config: Planning configuration
        
    Returns:
        (success_mask, trajectories) tuple where success_mask is boolean tensor
        and trajectories is list of trajectory JointStates (None for failed ones)
    """
    batch_size = start_states.position.shape[0]
    
    # Handle single trajectory case with plan_single instead of plan_batch
    if batch_size == 1:
        result = motion_gen.plan_single(start_states, goal_poses, plan_config)
        if result.success.item():
            traj = result.get_interpolated_plan()
            return result.success.view(-1), [traj]
        else:
            return result.success.view(-1), [None]
    
    # Batch case (batch_size > 1)
    result = motion_gen.plan_batch(start_states, goal_poses, plan_config)
    
    # Get interpolated trajectories for the batch using get_paths()
    # This returns a list of trajectories, including unsuccessful ones
    path_list = result.get_paths()
    
    # Create list with None for unsuccessful trajectories
    trajectories = []
    for i, (success, traj) in enumerate(zip(result.success, path_list)):
        if success.item():
            trajectories.append(traj)
        else:
            trajectories.append(None)
    
    return result.success, trajectories


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    # Setup logging
    setup_curobo_logger("warn")
    
    print("=" * 80)
    print("Booster T1 Trajectory Generator")
    print("=" * 80)
    print(f"Robot: {args.robot}")
    print(f"Number of trajectories: {args.num_trajectories}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output_file}")
    print(f"Workspace bounds: {args.workspace_bounds}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # Initialize tensor args
    tensor_args = TensorDeviceType()
    
    # Load robot config
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
    
    # Create world config with a far-away dummy obstacle
    # (required by collision checker, but won't affect self-collision checking)
    dummy_cuboid = Cuboid(
        name="dummy_far_obstacle",
        pose=[100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],  # Very far away
        dims=[0.01, 0.01, 0.01],  # Very small
    )
    world_config = WorldConfig(cuboid=[dummy_cuboid])
    
    print("\n[1/6] Initializing MotionGen...")
    # Initialize MotionGen
    # Note: When batch_size > 1, num_graph_seeds must be 1 for batch mode compatibility
    num_graph_seeds = 1 if args.batch_size > 1 else 8
    
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        args.robot,
        world_config,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_ik_seeds=50,
        num_trajopt_seeds=8,
        num_graph_seeds=num_graph_seeds,
        interpolation_dt=args.interpolation_dt,
        use_cuda_graph=False,  # Disable for flexibility
        interpolation_steps=10000,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    
    if args.batch_size > 1:
        print(f"  Using num_graph_seeds=1 for batch mode compatibility")
    
    print("[2/6] MotionGen initialized successfully")
    
    # Planning config
    # For batch mode, disable graph fallback to avoid warnings
    # (batch mode with graph requires num_graph_seeds=1 which triggers warnings)
    if args.batch_size > 1:
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=None,  # Disable graph fallback in batch mode
            max_attempts=3,
            enable_finetune_trajopt=True,
        )
        print("  Graph fallback disabled for batch mode (trajopt only)")
    else:
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,  # Enable graph fallback after 2 attempts
            max_attempts=3,
            enable_finetune_trajopt=True,
        )
    
    # Storage for trajectories
    trajectories_data = {}
    successful_trajectories = 0
    total_attempts = 0
    
    print(f"\n[3/6] Generating {args.num_trajectories} trajectories (batch size: {args.batch_size})...")
    start_time = time.time()
    
    while successful_trajectories < args.num_trajectories:
        # Calculate how many trajectories to attempt in this batch
        remaining = args.num_trajectories - successful_trajectories
        current_batch_size = min(args.batch_size, remaining)
        
        if total_attempts >= args.num_trajectories * args.max_attempts:
            print(f"\nWarning: Reached maximum attempts ({total_attempts})")
            break
        
        total_attempts += current_batch_size
        
        print(f"\n--- Batch starting at trajectory {successful_trajectories + 1}/{args.num_trajectories} (batch size: {current_batch_size}) ---")
        
        # Step 1: Sample random workspace poses (more than batch size for filtering)
        print("  Sampling workspace poses...")
        candidate_poses = sample_workspace_poses(
            args.workspace_bounds,
            num_poses=current_batch_size * 10,  # Sample extra for filtering
            use_random_orientation=args.use_random_orientation,
            rng=rng,
        )
        
        # Step 2: Filter for IK-solvable poses using BATCH IK
        print("  Filtering IK-solvable poses (batched)...")
        valid_poses = filter_ik_solvable_poses_batch(motion_gen, candidate_poses, tensor_args)
        
        if len(valid_poses) < current_batch_size:
            print(f"Only found {len(valid_poses)} valid poses, need {current_batch_size}, retrying...")
            continue
        
        print(f"  ✓ Found {len(valid_poses)} valid poses")
        
        # Step 3: Sample random start configurations (batch)
        print("  Sampling start configurations (batched)...")
        try:
            start_configs = motion_gen.ik_solver.sample_configs(
                current_batch_size, 
                rejection_ratio=10
            )
            if len(start_configs) < current_batch_size:
                print(f"Failed to sample {current_batch_size} collision-free configs, retrying...")
                continue
        except Exception as e:
            print(f"Error sampling start configs: {e}, retrying...")
            continue
        
        # Create batch JointState
        start_state_batch = JointState.from_position(
            start_configs[:current_batch_size],
            joint_names=motion_gen.kinematics.joint_names,
        )
        
        # Step 4: Select random goal poses and create batch
        print("  Planning trajectories (batched)...")
        selected_indices = rng.choice(len(valid_poses), size=current_batch_size, replace=False)
        goal_poses_list = [valid_poses[i][0] for i in selected_indices]
        
        # Stack into batch Pose
        goal_positions = np.array([p[:3] for p in goal_poses_list])
        goal_quaternions = np.array([p[3:] for p in goal_poses_list])
        goal_pose_batch = Pose(
            position=tensor_args.to_device(goal_positions),
            quaternion=tensor_args.to_device(goal_quaternions),
        )
        
        # Generate trajectories in BATCH
        success_mask, trajectories = generate_trajectories_batch(
            motion_gen, start_state_batch, goal_pose_batch, plan_config
        )
        
        num_successful = torch.count_nonzero(success_mask).item()
        print(f"  ✓ {num_successful}/{current_batch_size} trajectories generated successfully!")
        
        # Step 5: Store successful trajectories
        for i, (success, trajectory) in enumerate(zip(success_mask, trajectories)):
            if success.item() and trajectory is not None:
                traj_id = successful_trajectories
                trajectories_data[f"trajectory_{traj_id}_position"] = trajectory.position.cpu().numpy()
                trajectories_data[f"trajectory_{traj_id}_velocity"] = trajectory.velocity.cpu().numpy()
                trajectories_data[f"trajectory_{traj_id}_acceleration"] = (
                    trajectory.acceleration.cpu().numpy()
                )
                trajectories_data[f"trajectory_{traj_id}_dt"] = args.interpolation_dt
                trajectories_data[f"trajectory_{traj_id}_joint_names"] = trajectory.joint_names
                
                successful_trajectories += 1
                
                if successful_trajectories >= args.num_trajectories:
                    break
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("[4/6] Trajectory Generation Complete")
    print(f"  Successfully generated: {successful_trajectories}/{args.num_trajectories}")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Success rate: {successful_trajectories/total_attempts*100:.1f}%")
    print(f"  Time elapsed: {elapsed_time:.2f}s")
    print("=" * 80)
    
    if successful_trajectories == 0:
        print("\nNo trajectories were generated. Exiting without saving.")
        return
    
    # Add metadata
    print("\n[5/6] Preparing metadata...")
    metadata = {
        "robot_config": args.robot,
        "num_trajectories": successful_trajectories,
        "workspace_bounds": args.workspace_bounds,
        "interpolation_dt": args.interpolation_dt,
        "use_random_orientation": args.use_random_orientation,
        "seed": args.seed,
        "generation_timestamp": datetime.now().isoformat(),
        "total_attempts": total_attempts,
    }
    trajectories_data["metadata"] = metadata
    
    # Save to NPZ file
    print(f"[6/6] Saving trajectories to {args.output_file}...")
    np.savez(args.output_file, **trajectories_data)
    
    print(f"\n✓ Trajectories saved successfully!")
    print(f"\nTo load the trajectories:")
    print(f"  data = np.load('{args.output_file}', allow_pickle=True)")
    print(f"  trajectory_0_pos = data['trajectory_0_position']")
    print(f"  metadata = data['metadata'].item()")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

