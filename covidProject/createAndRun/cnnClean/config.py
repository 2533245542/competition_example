tune_frequency = 1



space = [
     hp.choice('feature_indices', indice_list[-2:-1]),
     hp.choice('in_batch_size', [3,4,5,6,7,8,9,10,11,12,13]),
     hp.choice('steps_ahead', [1]),
     hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
     hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
     hp.choice('num_cells', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
     hp.lognormal('learning_rate', np.log(.01), 3.),
     hp.choice('cnn_filters', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]),
     hp.choice('cnn_kernel_size', [1,3,5,7,9]),
     hp.choice('cnn_pool_size', [1,2,3,4,5,6]),
     hp.choice('cnn_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
]