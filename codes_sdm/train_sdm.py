"""
Train a diffusion model on images.
"""
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion, args_to_dict, add_dict_to_argparser
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(args.image_size)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_root=args.data_root,
        data_list=args.data_list,
        batch_size=args.batch_size,
        height=args.image_size,
        width=int(args.image_size * 0.75)
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def model_and_diffusion_defaults():
    """
    Defaults for model and diffusion defaults.
    """
    return dict(
        image_size=256*4,
        num_classes=13,
        num_channels=128, # 256
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=32, # 64
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=True,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        no_instance=True,   #---------------------------------
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def create_argparser():
    defaults = dict(
        schedule_sampler='uniform',
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,      
        ema_rate='0.9999',
        drop_rate=0.0,
        fp16_scale_growth=1e-3,
        is_train=True,
        log_interval=10,
        save_interval=50000,
        resume_checkpoint='',
        data_root='../data/zalando-hd-resized/',
        data_list='train_pairs.txt',
        save_dir='../checkpoints/sdm/1024/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
