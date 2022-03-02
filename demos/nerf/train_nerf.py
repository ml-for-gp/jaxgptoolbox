from utils import *

# Re-implementation of "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Mildenhall et al 2020
if __name__ == "__main__":
    imgs, poses, focal = load_tiny_nerf('./tiny_nerf_data.npz')
    # jax_imshow(imgs[0])

    # hyper parameters
    hyper_params = {
        "n_in": 3,
        "n_out": 4,
        "n_pos_encode": 6,
        "n_samples_per_path": 256,
        "near_plane": 2.0,
        "far_plane": 6.0,
        "h_mlp": [128,128,128,128,128,128],
        "step_size": 1e-4,
        "num_epochs": 300,
        "batch_size": 200
    }

    # initialize a nerf network
    model = NeRF(hyper_params)
    params = model.initialize_weights()

    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=hyper_params["step_size"])
    opt_state = opt_init(params)

    def loss(params_, origin_, directions_, img_):
        out = model.path_integral(params_, origin_, directions_)
        out = np.clip(out, 0.0, 1.0)
        loss_val = np.mean((out - img_)**2)
        return loss_val
    
    @jit
    def update(epoch, opt_state, origin_, directions_, img_):
        params_ = get_params(opt_state)
        value, grads = value_and_grad(loss, argnums = 0)(params_, origin_, directions_, img_)
        opt_state = opt_update(epoch, grads, opt_state)
        return value, opt_state

    # training
    loss_history = onp.zeros(hyper_params["num_epochs"])
    pbar = tqdm.tqdm(range(hyper_params["num_epochs"]))
    # data for training
    n_imgs = imgs.shape[0]
    img_H = imgs.shape[1]
    img_W = imgs.shape[2]
    batch_size = hyper_params["batch_size"]
    for epoch in pbar:
        for ii in range(n_imgs):
            # gradient step
            orig, dirs = generate_rays_from_camera(img_H, img_W, focal, poses[ii])
            dirs = np.reshape(dirs, (-1, 3))
            img = np.reshape(imgs[ii], (-1, 3))

            # split rays into batches
            img_batches = np.array_split(img, batch_size)
            dirs_batches = np.array_split(dirs, batch_size)
            for b in range(batch_size):
                img_b = img_batches[b]
                dirs_b = dirs_batches[b]
                loss_value, opt_state = update(epoch, opt_state, orig, dirs_b, img_b)
            
                # save loss
                loss_history[epoch] += loss_value / n_imgs / batch_size
        pbar.set_postfix({"loss": loss_history[epoch]})

        if epoch % 1 == 0: # plot loss history every 1000 iter
            plt.close(1)
            plt.figure(1)
            plt.semilogy(loss_history[:epoch])
            plt.title('reconstruction loss')
            plt.grid()
            plt.savefig("nerf_loss_history.jpg")

            params = get_params(opt_state)
            with open("nerf_params.pkl", 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save final parameters
    params = get_params(opt_state)
    with open("nerf_params.pkl", 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)