from model import *

if __name__ == '__main__':
  random.seed(1)

  # hyper parameters
  hyper_params = {
    "dim_in": 2,
    "dim_t": 1,
    "dim_out": 1,
    "h_mlp": [64,64,64],
    "step_size": 1e-4,
    "grid_size": 32,
    "num_epochs": 50000,
    "samples_per_epoch": 512
  }

  # initialize a mlp
  model = mlp(hyper_params)
  params = model.initialize_weights()

  # optimizer
  opt_init, opt_update, get_params = optimizers.adam(step_size=hyper_params["step_size"])
  opt_state = opt_init(params)

  # define loss function and update function
  def loss(params_, x_, y0_, y1_):
    out0 = model.forward(params_, np.array([0.0]), x_) # star when t = 0.0
    out1 = model.forward(params_, np.array([1.0]), x_) # circle when t = 1.0
    loss_sdf = np.mean((out0 - y0_)**2) + np.mean((out1 - y1_)**2)
    return loss_sdf

  @jit
  def update(epoch, opt_state, x_, y0_, y1_):
    params_ = get_params(opt_state)
    value, grads = value_and_grad(loss, argnums = 0)(params_, x_, y0_, y1_)
    opt_state = opt_update(epoch, grads, opt_state)
    return value, opt_state

  # training
  loss_history = onp.zeros(hyper_params["num_epochs"])
  pbar = tqdm.tqdm(range(hyper_params["num_epochs"])) # progress bar
  for epoch in pbar:
    # sample a bunch of random points
    x = np.array(random.rand(hyper_params["samples_per_epoch"], hyper_params["dim_in"]))
    y0 = jgp.sdf_star(x) # target SDF values at x
    y1 = jgp.sdf_circle(x) # target SDF values at x

    # update network parameters
    loss_value, opt_state = update(epoch, opt_state, x, y0, y1)
    loss_history[epoch] = loss_value
    pbar.set_postfix({"loss": loss_value})

    if epoch % 1000 == 0: # plot loss history every 1000 iter
      plt.close(1)
      plt.figure(1)
      plt.semilogy(loss_history[:epoch])
      plt.title('Reconstruction loss')
      plt.grid()
      plt.savefig("loss_history.jpg")

  # save final parameters
  params = get_params(opt_state)
  with open("mlp_params.pkl", 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # save results
  sdf_cm = mpl.colors.LinearSegmentedColormap.from_list('SDF', [(0,'#eff3ff'),(0.5,'#3182bd'),(0.5,'#31a354'),(1,'#e5f5e0')], N=256) # color map
  levels = onp.linspace(-0.5, 0.5, 21) # isoline
  x = jgp.sample_2D_grid(hyper_params["grid_size"]) # sample on unit grid for visualization

  fig = plt.figure()
  y0 = jgp.sdf_star(x)
  im = plt.contourf(y0.reshape(hyper_params['grid_size'],hyper_params['grid_size']), levels = levels, cmap=sdf_cm)
  plt.axis('equal')
  plt.axis("off")
  plt.savefig('ground truth (t=0)')

  plt.clf()
  y0_pred = model.forward(params, np.array([0.0]), x)
  im = plt.contourf(y0_pred.reshape(hyper_params['grid_size'],hyper_params['grid_size']), levels = levels, cmap=sdf_cm)
  plt.axis('equal')
  plt.axis("off")
  plt.savefig('network output (t=0)')

  plt.clf()
  y1 = jgp.sdf_circle(x)
  im = plt.contourf(y1.reshape(hyper_params['grid_size'],hyper_params['grid_size']), levels = levels, cmap=sdf_cm)
  plt.axis('equal')
  plt.axis("off")
  plt.savefig('ground truth (t=1)')

  plt.clf()
  y1_pred = model.forward(params, np.array([1.0]), x)
  im = plt.contourf(y1_pred.reshape(hyper_params['grid_size'],hyper_params['grid_size']), levels = levels, cmap=sdf_cm)
  plt.axis('equal')
  plt.axis("off")
  plt.savefig('network output (t=1)')
