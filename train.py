import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.resample import create_named_schedule_sampler
from utils import dist_util, logger
from utils.image_datasets import load_data, load_data_local
from utils.train_util import TrainLoop
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log(args)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    if args.stage == 1:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.stage == 2:
        data = load_data_local(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
        )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        stage=args.stage,
        max_steps=args.max_steps,
        auto_scale_grad_clip=args.auto_scale_grad_clip,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="ffhq256_deca.lmdb",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        batch_size=4,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        log_dir="logs/stage1",
        num_workers=8,
        max_steps=50000,
        auto_scale_grad_clip=1.0,
        stage=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()

# python scripts/train.py --latent_dim 64 --encoder_type resnet18 --log_dir logs/stage1 --data_dir ffhq256_deca.lmdb --lr 1e-4 --p2_weight True --image_size 256 --batch_size 8 --max_steps 200000 --num_workers 8 --save_interval 5000 --stage 1

# python scripts/train.py --latent_dim 64 --encoder_type resnet18 --log_dir logs/stage2 --resume_checkpoint ckpt/stage1_model050000.pt --data_dir personal_deca.lmdb --lr 1e-5 --p2_weight True --image_size 256 --batch_size 4 --max_steps 5000 --num_workers 8 --save_interval 5000 --stage 2
