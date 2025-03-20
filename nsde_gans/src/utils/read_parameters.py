import json

def get_parameters(params_path = './Params/model1.json' ):
        with open(params_path) as f:
            params = json.load(f)
        initial_noise_size = params['initial_noise_size'] # How many noise dimensions to sample at the start of the SDE.
        noise_size = params['noise_size'] # How many dimensions the Brownian motion has.
        hidden_size = params['hidden_size'] # How big the hidden size of the generator SDE and the discriminator CDE are.
        mlp_size = params['mlp_size'] # How big the layers in the various MLPs are.
        num_layers = params['num_layers'] # How many hidden layers to have in the various MLPs.
        
        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr = params['generator_lr'] # Learning rate often needs careful tuning to the problem.
        discriminator_lr = params['discriminator_lr'] # Learning rate often needs careful tuning to the problem.
        batch_size = params['batch_size'] # Batch size.
        steps = params['steps'] # How many steps to train both generator and discriminator for.
        init_mult1 = params['init_mult1'] # Changing the initial parameter size can help.
        init_mult2 = params['init_mult2']
        weight_decay = params['weight_decay'] # Weight decay.
        swa_step_start = params['swa_step_start'] # When to start using stochastic weight averaging.
        steps_per_print = params['steps_per_print'] # How often to print the loss.
        t_size = params['t_size'] 

        return ( initial_noise_size, noise_size, hidden_size, mlp_size, num_layers,
                generator_lr, discriminator_lr, batch_size, steps, init_mult1, init_mult2,
                weight_decay, swa_step_start, steps_per_print, t_size )