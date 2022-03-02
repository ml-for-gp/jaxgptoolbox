import sys
sys.path.append('../../../')
import jaxgptoolbox as jgp
import numpy as onp
import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax.experimental import optimizers
import pickle

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2
import tqdm

def load_tiny_nerf(path):
    data = onp.load(path)
    images = data["images"] # n_img x H x W x 3
    poses = data["poses"] # n_img x 4 x 4
    focal = float(data["focal"])
    return np.array(images), np.array(poses), focal

def jax_imshow(jax_img):
    img = onp.array(jax_img)
    img = img[:,:,[2,1,0]]
    scale = int(600. / img.shape[0])
    dim = (img.shape[0]*scale, img.shape[1]*scale)
    img_resize = cv2.resize(img, dim)
    cv2.imshow('image', img_resize)
    cv2.waitKey(0)

class NeRF:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params

    def initialize_weights(self):
        """
        initialize network weights
        """
        sizes = self.hyper_params["h_mlp"]

        # add input dimension
        n_raw_in = self.hyper_params["n_in"] # point dim
        n_encode = self.hyper_params["n_pos_encode"] # positional encoding with sin/cos
        n_in = n_raw_in + n_raw_in * 2 * n_encode # "2" is due to sin/cos
        sizes.insert(0, n_in)

        sizes.append( self.hyper_params["n_out"]) # add output dimension

        # initialization network parameters
        params = []    
        for ii in range(len(sizes) - 1):
            if ii == (len(sizes) - 2):
                # last layer only outputs RGB, n_out - 1
                # last layer inputs additional view dir, n_in + 3
                W, b = jgp.he_initialization(sizes[ii+1]-1, sizes[ii]+3)
            elif ii == (len(sizes) - 3):
                # second last layer outputs additional volume density
                W, b = jgp.he_initialization(sizes[ii+1]+1, sizes[ii])
            else:
                W, b = jgp.he_initialization(sizes[ii+1], sizes[ii])
            print(W.shape)
            params.append([W, b])
        # last layer add view direction
        return params
    
    def positional_encoding(self, x_in):
        """
        positional encoding: x -> [x, sin(w*x), cos(w*x)]
        where w = 2^[0, 1, ..., n_encode-1]
        """
        n_encode = self.hyper_params["n_pos_encode"]
        w = np.power(2., np.arange(0., n_encode))
        x = w[:,None] * x_in[None,:]
        x = x.flatten()
        return np.concatenate((x_in, np.sin(2*np.pi*x), np.cos(2*np.pi*x)))

    def activation(self, x): 
        return jax.nn.leaky_relu(x)

    def forward_single(self, params, dir, x):
        x = self.positional_encoding(x)
        for ii in range(len(params) - 1):
            W, b = params[ii]
            x = self.activation(np.dot(W, x) + b)
        # second last layer outputs volume density (sigma)
        sigma = jax.nn.relu(x[0])
        x = x[1:]
        # last layer append view direction
        x = np.concatenate((x, dir))
        W, b = params[-1]
        rgb = jax.nn.sigmoid(np.dot(W, x) + b)
        return np.append(rgb, sigma)
    forward = jax.vmap(forward_single, in_axes=(None, None, None, 0), out_axes=0)

    def path_integral_single(self, params, orig, dir):
        # generate query locations
        n_samples = self.hyper_params["n_samples_per_path"]
        near = self.hyper_params["near_plane"]
        far = self.hyper_params["far_plane"]
        dists = np.linspace(near, far, n_samples)
        x = orig[None,:] + dir[None,:] * dists[:,None]

        # forward pass to get (r,g,b,density)
        rgbd = self.forward(params, dir, x)

        # volume integral to get colors (see NeRF paper Eq.3)
        d = dists[1] - dists[0] # distance between samples
        cumsum_density = np.cumsum(rgbd[:,3])
        T = np.exp(-cumsum_density*d)
        w = T * (1.-np.exp(-rgbd[:,3]*d)) # weights of each evaluation
        rgb_out = w.dot(rgbd[:,:3]) # integrate along the path
        return rgb_out
    path_integral = jax.vmap(path_integral_single, in_axes=(None, None, None, 0), out_axes=0)

def generate_rays_from_camera(image_height, image_width, focal, pose):
    """
    generate camera rays in the world space for each pixel

    Inputs
    image_height: number of pixels in H direction
    image_width: number of pixels in W direction
    focal: focal length of a camera
    pose: 4x4 array of camera pose matrix

    Outputs
    ray_origin (3,) array of ray origin
    ray_directions HxWx3 array of ray directions
    """
    i, j = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing="xy")
    k = -np.ones_like(i)
    i = (i - image_width * 0.5) / focal
    j = -(j - image_height * 0.5) / focal
    directions = np.stack([i, j, k], axis=-1)
    camera_matrix = pose[:3, :3]
    ray_directions = np.einsum("ijl,kl", directions, camera_matrix)
    # ray_origins = np.broadcast_to(pose[:3, -1], ray_directions.shape)
    ray_origins = pose[:3, -1]
    return ray_origins, ray_directions

def angles_to_pose(theta, phi, radius):
    translation = lambda t: np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ]
        )
    rotation_phi = lambda phi: np.asarray(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )
    rotation_theta = lambda th: np.asarray(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ]
        )

    pose = translation(radius)
    pose = rotation_phi(phi / 180.0 * np.pi) @ pose
    pose = rotation_theta(theta / 180.0 * np.pi) @ pose
    return (
        np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]) @ pose
    )

def generate_test_poses():
    video_angle = onp.linspace(0.0, 360.0, 120, endpoint=False)
    poses = onp.zeros((len(video_angle), 4, 4))
    for ii in range(len(video_angle)):
        poses[ii] = angles_to_pose(video_angle[ii], -30, 4.0)
    return poses