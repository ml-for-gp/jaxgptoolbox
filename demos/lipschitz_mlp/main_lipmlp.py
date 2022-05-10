from model import *

# implementation of "Learning Smooth Neural Functions via Lipschitz Regularization" by Liu et al. 2022
if __name__ == '__main__':
  random.seed(1)

  # hyper parameters
  hyper_params = {
    "dim_in": 2,
    "dim_t": 1,
    "dim_out": 1,
    "h_mlp": [64,64,64,64,64],
    "step_size": 1e-4,
    "grid_size": 32,
    "num_epochs": 200000,
    "samples_per_epoch": 512
  }
  alpha = 1e-6

  # initialize a mlp
  model = lipmlp(hyper_params)
  params = model.initialize_weights()

  # optimizer
  opt_init, opt_update, get_params = optimizers.adam(step_size=hyper_params["step_size"])
  opt_state = opt_init(params)

  # define loss function and update function
  def loss(params_, alpha, x_, y0_, y1_):
    out0 = model.forward(params_, np.array([0.0]), x_) # star when t = 0.0
    out1 = model.forward(params_, np.array([1.0]), x_) # circle when t = 1.0
    loss_sdf = np.mean((out0 - y0_)**2) + np.mean((out1 - y1_)**2)
    loss_lipschitz = model.get_lipschitz_loss(params_)
    return loss_sdf + alpha * loss_lipschitz

  @jit
  def update(epoch, opt_state, alpha, x_, y0_, y1_):
    params_ = get_params(opt_state)
    value, grads = value_and_grad(loss, argnums = 0)(params_, alpha, x_, y0_, y1_)
    opt_state = opt_update(epoch, grads, opt_state)
    return value, opt_state

  # training
  loss_history = onp.zeros(hyper_params["num_epochs"])
  pbar = tqdm.tqdm(range(hyper_params["num_epochs"]))
  for epoch in pbar:
    # sample a bunch of random points
    x = np.array(random.rand(hyper_params["samples_per_epoch"], hyper_params["dim_in"]))
    y0 = jgp.sdf_star(x)
    y1 = jgp.sdf_circle(x)

    # update
    loss_value, opt_state = update(epoch, opt_state, alpha, x, y0, y1)
    loss_history[epoch] = loss_value
    pbar.set_postfix({"loss": loss_value})

    if epoch % 1000 == 0: # plot loss history every 1000 iter
      plt.close(1)
      plt.figure(1)
      plt.semilogy(loss_history[:epoch])
      plt.title('Reconstruction loss + Lipschitz loss')
      plt.grid()
      plt.savefig("lipschitz_mlp_loss_history.jpg")

  # save final parameters
  params = get_params(opt_state)
  with open("lipschitz_mlp_params.pkl", 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # normalize weights during test time
  params_final = model.normalize_params(params)

  # save result as a video
  sdf_cm = mpl.colors.LinearSegmentedColormap.from_list('SDF', [(0,'#eff3ff'),(0.5,'#3182bd'),(0.5,'#31a354'),(1,'#e5f5e0')], N=256)

  # create video
  fig = plt.figure()
  x = jgp.sample_2D_grid(hyper_params["grid_size"]) # sample on unit grid for visualization
  def animate(t):
      plt.cla()
      out = model.forward_eval(params_final, np.array([t]), x)
      levels = onp.linspace(-0.5, 0.5, 21)
      im = plt.contourf(out.reshape(hyper_params['grid_size'],hyper_params['grid_size']), levels = levels, cmap=sdf_cm)
      plt.axis('equal')
      plt.axis("off")
      return im
  anim = animation.FuncAnimation(fig, animate, frames=np.linspace(0, 1, 50), interval=50)
  anim.save("lipschitz_mlp_interpolation.mp4")