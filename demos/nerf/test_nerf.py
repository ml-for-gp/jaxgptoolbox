from utils import *

if __name__ == "__main__":
    poses = generate_test_poses()
    imgs, _, focal = load_tiny_nerf('./tiny_nerf_data.npz')

    # hyper parameters
    hyper_params = {
        "n_in": 3,
        "n_out": 4,
        "n_pos_encode": 6,
        "n_samples_per_path": 128,
        "near_plane": 2.0,
        "far_plane": 6.0,
        "h_mlp": [128,128,128],
        "step_size": 1e-4,
        "num_epochs": 1000,
        "batch_size": 200
    }

    # initialize a nerf network
    model = NeRF(hyper_params)
    with open('nerf_params.pkl', 'rb') as handle:
        params = pickle.load(handle)

    # create video
    n_imgs = poses.shape[0]
    img_H = imgs.shape[1]
    img_W = imgs.shape[2]
    fig = plt.figure()
    def animate(ii):
        plt.cla()
        orig, dirs = generate_rays_from_camera(img_H, img_W, focal, poses[ii])
        dirs = np.reshape(dirs, (-1, 3))

        out = model.path_integral(params, orig, dirs)
        out = onp.reshape(onp.array(out), (img_H, img_W, 3))
        im = plt.imshow(out)
        return im
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(n_imgs), interval=50)
    anim.save("nerf_interpolation.mp4")

