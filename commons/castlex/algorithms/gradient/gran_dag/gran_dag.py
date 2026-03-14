import castle.algorithms

class GraNDAG(castle.algorithms.GraNDAG):
    # The original GraNDAG requires in input 'input_dim', corresponding
    # to the number of columns in the dataset used for the inference.
    # However, this value is ONLY saved for future usage.
    # This class delegates the extraction of this information directly
    # in the 'learn()' method

    def __init__(self,
                 #input_dim,
                 hidden_num=2,
                 hidden_dim=10,
                 batch_size=64,
                 lr=0.001,
                 iterations=10000,
                 model_name='NonLinGaussANM',
                 nonlinear='leaky-relu',
                 optimizer='rmsprop',
                 h_threshold=1e-8,
                 device_type='cpu',
                 device_ids='0',
                 use_pns=False,
                 pns_thresh=0.75,
                 num_neighbors=None,
                 normalize=False,
                 precision=False,
                 random_seed=42,
                 jac_thresh=True,
                 lambda_init=0.0,
                 mu_init=0.001,
                 omega_lambda=0.0001,
                 omega_mu=0.9,
                 stop_crit_win=100,
                 edge_clamp_range=0.0001,
                 norm_prod='paths',
                 square_prod=False):
        super().__init__(
            input_dim=1,
            hidden_num=hidden_num,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            lr=lr,
            iterations=iterations,
            model_name=model_name,
            nonlinear=nonlinear,
            optimizer=optimizer,
            h_threshold=h_threshold,
            device_type=device_type,
            device_ids=device_ids,
            use_pns=use_pns,
            pns_thresh=pns_thresh,
            num_neighbors=num_neighbors,
            normalize=normalize,
            precision=precision,
            random_seed=random_seed,
            jac_thresh=jac_thresh,
            lambda_init=lambda_init,
            mu_init=mu_init,
            omega_lambda=omega_lambda,
            omega_mu=omega_mu,
            stop_crit_win=stop_crit_win,
            edge_clamp_range=edge_clamp_range,
            norm_prod=norm_prod,
            square_prod=square_prod
        )

    def learn(self, data, columns=None, **kwargs):
        self.input_dim = data.shape[1]
        return super().learn(data, columns=columns, **kwargs)
# end
