import os
import argparse
import torch as th
from conf_mgt import conf_base
from utils import yamlread
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_data
from PIL import Image
# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_args,
)

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf):
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    print('model is correct')
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress
    
    if not os.path.exists(conf.save_dir):
        os.makedirs(conf.save_dir)

    cond_fn = None
    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y)
    
    print("sampling...")
    data = load_data(data_root=conf.data_root, data_list=conf.data_list, batch_size=1, height=conf.image_size, width=int(conf.image_size * 0.75), up=conf.up)

    for i, (batch, cond) in enumerate(data):
        print(i)        
        model_kwargs = {}

        model_kwargs['gt'] = batch
        model_kwargs['gt_keep_mask'] = cond['gt_keep_mask']
        model_kwargs['y'] = cond['y']
        
        for k in model_kwargs.keys():
            if isinstance(model_kwargs[k], th.Tensor):
                model_kwargs[k] = model_kwargs[k].to(device)
        
        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result = sample_fn(
            model_fn,
            (1, 3, conf.image_size, int(conf.image_size*0.75)),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])

        Image.fromarray(srs[0]).save(os.path.join(conf.save_dir, cond['im_name'][0]))

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
