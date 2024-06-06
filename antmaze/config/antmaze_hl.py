import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("jump", "J"),
]

plan_args_to_watch = [
    ("prefix", ""),
    ##
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ##
    ("conditional", "cond"),
    ("jump", "J"),
    ("kernel_size", "k"),
]

logbase = "logs"
base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 270,
        "jump": 30,
        "jump_action": "none",
        "condition": True,
        "n_diffusion_steps": 256,
        "action_weight": 0.1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (2, 4, 8),
        "upsample_k": (3, 3),
        "downsample_k": (3, 3),
        "attention": False,
        "kernel_size": 5,
        "dim": 32,
        "renderer": "utils.Maze2dRenderer",
        ## dataset
        "loader": "datasets.GoalDataset",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["antmaze_set_terminals"],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 1001,
        "dist_thre": 0.01,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 2.5e6,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 1000,
        "sample_freq": 10000,
        "n_saves": 50,
        "save_parallel": False,
        "n_reference": 50,
        "n_samples": 10,
        "bucket": None,
        "device": "cuda",
    },
    "plan": {
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 270,
        "jump": 30,
        "jump_action": "none",
        "attention": False,
        "condition": True,
        "kernel_size": 5,
        "dim": 32,
        "n_diffusion_steps": 256,
        "normalizer": "LimitsNormalizer",
        ## serialization
        "vis_freq": 10,
        "logbase": logbase,
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        ## loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}_J{jump}",
        "diffusion_epoch": "latest",
    },
}

# ------------------------ overrides ------------------------#


antmaze_large_diverse_v2 = antmaze_large_diverse_v0 = {
    "diffusion": {
        "horizon": 450,
        "n_diffusion_steps": 256,
        "upsample_k": (4, 3),
        "downsample_k": (3, 4),
        "dim_mults": (1, 4, 8),
        "dist_thre": 0.01,
    },
    "plan": {
        "horizon": 450,
        "n_diffusion_steps": 256,
    },
}

antmaze_ultra_diverse_v0 = {
    "diffusion": {
        "horizon": 720,
        "n_diffusion_steps": 256,
        "upsample_k": (4, 4, 4),
        "downsample_k": (4, 4, 4),
        "dim_mults": (1, 4, 8),
        "dist_thre": 0.01,
    },
    "plan": {
        "horizon": 720,
        "n_diffusion_steps": 256,
    },
}
