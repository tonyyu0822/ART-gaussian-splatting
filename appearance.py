import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
from argparse import Namespace
from scene.cameras import Camera
import numpy as np
from tglcourse.generation_utils import calc_vgg_features, ContentLossToTarget, CLIPLossToTargets
from mayavi import mlab
import cv2
import os
import sys

def get_first_image_dimensions(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            image = cv2.imread(full_path)
            if image is not None:
                height, width, _ = image.shape
                return width, height
    return None

def preview(view):
    with torch.no_grad():
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, "pic.png")
        return Image.open("pic.png")
    
def visualize_gaussians_mayavi(gaussians):
    xyz = gaussians._xyz.cpu().detach().numpy()
    mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], scale_factor=0.1)
    mlab.show()


def main():
    sys.argv = ["render.py", "--model_path", "/scratch/by12/Multi-elevation_NeRF/gaussian_splatting/output/sfm_front_subset/",
           "--source_path", "/scratch/by12/Dataset/front_door_1/",
           "--iteration", "30000", "--skip_train", "--skip_test", "--quiet"]

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset, iteration, pipeline = model.extract(args), args.iteration, pipeline.extract(args)
    w, h = get_first_image_dimensions('/scratch/by12/Dataset/Wriva_extracted/siteS01-carla-01/colmap/images')

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras()
        view = views[30]
        im = preview(view)

    im.resize((w, h))

    gaussians_features = [gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._scaling, gaussians._rotation, gaussians._opacity]
    names = ["xyz", "features_dc", "features_rest", "scaling", "rotation", "opacity"]
    print(visualize_gaussians_mayavi(gaussians))

    # for feature, name in zip(gaussians_features, names):
    #     print(f"Feature: {name}, Shape: {feature.shape}, Min: {feature.min()}, Max: {feature.max()}")

if __name__ == "__main__":
    main()
