import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
from tqdm import tqdm
        
def load_ply(path, sh_degree=3):
    max_sh_degree = sh_degree  

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    weights = np.asarray(plydata.elements[0]["weight"])[..., np.newaxis]
    weights_cnt = np.asarray(plydata.elements[0]["weights_cnt"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    _features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    _opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    _weight = torch.tensor(weights, dtype=torch.float, device="cuda").requires_grad_(False)
    _weights_cnt = torch.tensor(weights_cnt, dtype=torch.int, device="cuda").requires_grad_(False)        
    _scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    _rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    active_sh_degree = max_sh_degree

    gaussians = {
        "xyz": _xyz,
        "features_dc": _features_dc,
        "features_rest": _features_rest,
        "opacity": _opacity,
        "weight": _weight,
        "weights_cnt": _weights_cnt,
        "scaling": _scaling,
        "rotation": _rotation
    }
    return gaussians

def save_ply_from_dict(dict, save_path):
    
    def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('weight')
        l.append('weights_cnt')
        for i in range(_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    # mkdir_p(os.path.dirname(save_path))

    xyz = dict['xyz'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = dict['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = dict['features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = dict['opacity'].detach().cpu().numpy()
    weights = dict['weight'].detach().cpu().numpy()
    weights_cnt = dict['weights_cnt'].detach().cpu().numpy()
    scale = dict['scaling'].detach().cpu().numpy()
    rotation = dict['rotation'].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(dict['features_dc'], dict['features_rest'],  dict['scaling'], dict['rotation'])]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    # print(np.shape(xyz), np.shape(opacities), np.shape(weights), len(elements))
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, weights, weights_cnt, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

def calculate_dist_vectorized(center, points):
    dist = torch.sqrt(torch.sum((points - center) ** 2, dim=1))
    return dist
def calculate_center(gaussians_xyz):
    center = torch.mean(gaussians_xyz, dim=0)
    return center

# load bg and object gaussians
background_path = '/home/neiljin/Documents/PycharmProjects/gaussian_splatting_mask/output/room/point_cloud/iteration_30000/point_cloud.ply'
object_path = '/home/neiljin/Documents/PycharmProjects/gaussian_splatting_mask/output/bicycle_cnt/mask_selected_gaussians_bike.ply'
background_gaussians = load_ply(background_path)
object_gaussians = load_ply(object_path)

# use xyz for nearest selection
xyzs_object = object_gaussians['xyz']
xyzs_background = background_gaussians['xyz'] # (5474545, 3)
dist_list = []

for i, xyz_object in enumerate(tqdm(xyzs_object)):
    distances = calculate_dist_vectorized(xyz_object, xyzs_background)
    sorted_indices = torch.argsort(distances)
    nearest_indice = sorted_indices[0]
    nearest_bg_gaussians_xyzs = xyzs_background[nearest_indice]
    nearest_distances = distances[nearest_indice]

    selected_gaussians_dc = background_gaussians['features_dc'][nearest_indice]
    selected_gaussians_rest = background_gaussians['features_rest'][nearest_indice]

    # switch SH
    # object_gaussians_dc = object_gaussians['features_dc'].clone()
    # object_gaussians_dc[i] = selected_gaussians_dc
    # object_gaussians['features_dc'] = object_gaussians_dc

    object_gaussians_rest = object_gaussians['features_rest'].clone()
    # object_gaussians_rest[i] = selected_gaussians_rest*(2*torch.exp(-nearest_distances.detach())*torch.mean(selected_gaussians_dc))
    object_gaussians_rest[i] *= 20

    object_gaussians['features_rest'] = object_gaussians_rest

# merge new object and bg
merged_gaussians = {}
for key in background_gaussians:
    if key in object_gaussians:
        if key=="weights_cnt"or"weight":
            merged_gaussians[key] = torch.cat([background_gaussians[key].data, object_gaussians[key].data], dim=0)
        else:
            merged_gaussians[key] = nn.Parameter(torch.cat([background_gaussians[key].data, object_gaussians[key].data], dim=0))
    else:
        print("gaussians format won't match")
save_ply_from_dict(merged_gaussians, 'edited_ply/merged_scene.ply')
save_ply_from_dict(object_gaussians,'edited_ply/object_sh_switch.ply')

