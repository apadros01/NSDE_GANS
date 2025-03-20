import json
import os
import torch
import torch.optim.swa_utils as swa_utils
from nsde_gans.src.utils.read_parameters import get_parameters
from nsde_gans.src.utils.preprocess_cvd import preprocess_data
from nsde_gans.src.models.model1 import Generator, Discriminator
from nsde_gans.src.utils.metrics import evaluate_loss, fidelity
from nsde_gans.src.utils.synthetic_data_generation import synthtetic_dataset, sample_generator


if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    params_path = os.path.join(current_path, '..', '..', 'config', 'model1.json')
    data_path = os.path.join(current_path, '..', '..', '..', 'data')

    initial_noise_size, noise_size, hidden_size, mlp_size, num_layers,\
                generator_lr, discriminator_lr, batch_size, steps, init_mult1, init_mult2,\
                weight_decay, swa_step_start, steps_per_print, t_size = get_parameters(params_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

    X, df_cvd, df_no_cvd = preprocess_data(data_path)
    ts = torch.linspace(0, t_size - 1, t_size, device=device)
    data_size = X.shape[1]
    
    # dataloader
    ys_coeffs = torch.from_numpy(X)
    ys_coeffs = ys_coeffs.to(torch.float32)
    ys_coeffs = torch.reshape(ys_coeffs,(X.shape[0],1,X.shape[1]))
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    # Models
    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers+1).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers-1).to(device)

    # Weight averaging really helps with GAN training.
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # Picking a good initialisation is important!
    # In this case these were picked by making the parameters for the t=0 part of the generator be roughly the right
    # size that the untrained t=0 distribution has a similar variance to the t=0 data distribution.
    # Then the func parameters were adjusted so that the t>0 distribution looked like it had about the right variance.
    # What we're doing here is very crude -- one can definitely imagine smarter ways of doing things.
    # (e.g. pretraining the t=0 distribution)
    # with torch.no_grad():
    #     for param in generator._initial.parameters():
    #         param *= init_mult1
    #     for param in generator._func.parameters():
    #         param *= init_mult2



    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
    discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=discriminator_lr,
                                                weight_decay=weight_decay)
    

    # Train both generator and discriminator.
    for step in range(steps):
        real_samples, = next(infinite_train_dataloader)

        generated_samples = generator(ts, batch_size)
        generated_score = discriminator(generated_samples)
        real_score = discriminator(real_samples)
        loss = generated_score - real_score
        loss.backward()
        for param in generator.parameters():
            param.grad *= -1
        generator_optimiser.step()
        discriminator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        ###################
        # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
        # LipSwish activation functions).
        ###################
        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        # Stochastic weight averaging typically improves performance.
        if step > swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_unaveraged_loss = evaluate_loss(ts, batch_size, train_dataloader, generator, discriminator)
            if step > swa_step_start:
                total_averaged_loss = evaluate_loss(ts, batch_size, train_dataloader, averaged_generator.module,
                                                    averaged_discriminator.module)
                print(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                            f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                print(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
        if step % 500 == 0 or step == steps - 1:
            df_synt = synthtetic_dataset(averaged_generator,ts,size=500,df=df_cvd.iloc[:,1:27])
            print(fidelity(df_cvd.iloc[:,1:27], df_synt, k=2, c=10).agg('mean'))
            
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    df_synt = synthtetic_dataset(generator = generator, ts = ts, size=500, df=df_no_cvd.iloc[:,1:27])
    results = fidelity(df_no_cvd.iloc[:,1:27], df_synt, k=2, c=10).agg('mean')
    print(results)


