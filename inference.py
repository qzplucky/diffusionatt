import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
from glob import glob

from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from torchvision.utils import save_image

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset

import pickle


def create_inter_data(dataset, modes, meanshape_path=""):
    # Build DECA
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg)

    meanshape = None
    if os.path.exists(meanshape_path):
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        print("not use meanshape")

    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    with th.no_grad():
        code2 = deca.encode(img2)
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")

    for i in range(len(dataset) - 1):

        img1 = dataset[i]["image"].unsqueeze(0).to("cuda")

        with th.no_grad():
            code1 = deca.encode(img1)

        # To align the face when the pose is changing
        ffhq_center = None
        ffhq_center = deca.decode(code1, return_ffhq_center=True)

        tform = dataset[i]["tform"].unsqueeze(0)
        tform = th.inverse(tform).transpose(1, 2).to("cuda")
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        code1["tform"] = tform
        if meanshape is not None:
            code1["shape"] = meanshape

        for mode in modes:

            code = {}
            for k in code1:
                code[k] = code1[k].clone()

            origin_rendered = None

            if mode == "pose":
                code["pose"][:, :3] = code2["pose"][:, :3]
            elif mode == "light":
                code["light"] = code2["light"]
            elif mode == "exp":
                code["exp"] = code2["exp"]
                code["pose"][:, 3:] = code2["pose"][:, 3:]
            elif mode == "latent":
                pass

            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
                align_ffhq=True,
                ffhq_center=ffhq_center,
            )

            origin_rendered = opdict["rendered_images"].detach()

            batch = {}
            batch["image"] = original_image * 2 - 1
            batch["image2"] = image2 * 2 - 1
            batch["rendered"] = opdict["rendered_images"].detach()
            batch["normal"] = opdict["normal_images"].detach()
            batch["albedo"] = opdict["albedo_images"].detach()
            batch["mode"] = mode
            batch["origin_rendered"] = origin_rendered
            yield batch


def main():
    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ckpt = th.load(args.model_path)

    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()

    imagepath_list = []

    if not os.path.exists(args.source) or not os.path.exists(args.target):
        print("source file or target file doesn't exists.")
        return

    imagepath_list = []
    if os.path.isdir(args.source):
        imagepath_list += (
                glob(args.source + "/*.jpg")
                + glob(args.source + "/*.png")
                + glob(args.source + "/*.bmp")
        )
    else:
        imagepath_list += [args.source]
    imagepath_list += [args.target]
    dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)

    modes = args.modes.split(",")

    data = create_inter_data(dataset, modes, args.meanshape)

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    os.system("mkdir -p " + args.output_dir)

    noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

    vis_dir = args.output_dir

    idx = 0
    for batch in data:
        image = batch["image"]
        image2 = batch["image2"]
        rendered, normal, albedo = batch["rendered"], batch["normal"], batch["albedo"]

        # show_tensor(image)
        # show_tensor(image2)
        # show_tensor(normal)
        # show_tensor(albedo)
        # show_tensor(rendered)

        save_image(rendered, os.path.join(vis_dir, "rendered_{}_".format(idx) + batch["mode"]) + ".png")
        save_image(normal, os.path.join(vis_dir, "normal_{}_".format(idx) + batch["mode"]) + ".png")
        save_image(albedo, os.path.join(vis_dir, "albedo_{}_".format(idx) + batch["mode"]) + ".png")

        physic_cond = th.cat([rendered, normal, albedo], dim=1)

        image = image
        physic_cond = physic_cond

        with th.no_grad():
            if batch["mode"] == "latent":
                detail_cond = model.encode_cond(image2)
            else:
                detail_cond = model.encode_cond(image)

        sample = sample_fn(
            model,
            (1, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs={"physic_cond": physic_cond, "detail_cond": detail_cond},
        )
        sample = (sample + 1) / 2.0
        sample = sample.contiguous()

        save_image(
            sample, os.path.join(vis_dir, "{}_".format(idx) + batch["mode"]) + ".png"
        )
        idx += 1


def show_tensor(tensor_data):
    import matplotlib.pyplot as plt

    numpy_data = tensor_data.squeeze(0).cpu().numpy()
    numpy_data = numpy_data.transpose(1, 2, 0)

    plt.imshow(numpy_data)
    plt.axis('off')
    plt.show()


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        # model_path="logs/stage2/model005000.pt",
        # model_path="logs/stage2_wo_cross/model005000.pt",
        # model_path="logs/stage2_biden/model005000.pt",
        # model_path="logs/stage2_taylor/model005000.pt",
        # model_path="logs/stage2_taylor_10/model005000.pt",
        # model_path="logs/stage1/model200000.pt",
        model_path="logs/stage2_234/model005000.pt",

        # source="personal_images_aligned/",
        # source="personal_images_aligned/gettyimages-1352102251-05f099a39b830451ca79f235ffbe1a40f830862d.png",
        # source="personal_images_biden_aligned/",
        # source="personal_images_biden_aligned/biden-GettyImages-1392610778.png",
        # source="personal_images_taylor_aligned/",
        source="234/",
        # source="personal_images_taylor_aligned/7.png",

        # target="personal_images_aligned/14vid-Obama-mediumSquareAt3X-v3.png",
        # target="personal_images_biden_aligned/Joe-Biden-CTO-Business-1239469788.png",
        target="234/3.png",

        # output_dir="logs/output_adds/",
        # output_dir="logs/output_biden/",
        # output_dir="logs/output_taylor/",
        # output_dir="logs/output_taylor_stage1/",
        output_dir="logs/output_234/",

        modes="light",  # pose,exp,light
        meanshape="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

# python scripts/inference.py --source SOURCE_IMAGE_FILE --target TARGET_IMAGE_FILE --output_dir log/ouput/ --modes light --model_path log/stage2/model005000.pt --meanshape PATH_TO_MEANSHAPE --timestep_respacing ddim20

# python scripts/inference.py --meanshape PATH_TO_MEANSHAPE --timestep_respacing ddim20

# python scripts/inference.py --timestep_respacing ddim20
