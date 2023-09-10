hyper_params = {
	'dataset': 'm100k_small2', 
	'float64': False,

	'depth': 3,
	'grid_search': True,

  #KRR
  'grid_search_lamda': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
	'lamda': 1.0, # Only used if grid_search_lamda == False
  'grid_search_gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
  'gamma': 0.001,

  #MLP
  'grid_search_adam_lr': [0.00001, 0.001, 0.03, 1.0, 3.0],
  'grid_search_epochs': [100, 200],
  'grid_search_batch_size': [32, 64, 128],
  'grid_search_reg': [0.01, 0.1, 1.0, 10.0],
  'width': 512,
  'adam_lr': 0.0005,
  'epochs': 3000,

  #MLP
  'grid_search_adam_lr': [0.00001, 0.001, 0.03, 1.0, 3.0],
  'grid_search_epochs': [100, 200],
  'grid_search_batch_size': [32, 64, 128],
  'grid_search_reg': [0.01, 0.1, 1.0, 10.0],
  'width': 512,
  'adam_lr': 0.0005,
  'epochs': 3000,
  'batch_size': 128,
  'reg': 1.0,

  #GP
  'grid_search_max_iter': [5, 10, 15, 20, 30, 50, 70],
  'max_iter': 10,

	# Number of users to keep (randomly)
	'user_support': -1, # -1 implies use all users
	'seed': 42,
}
