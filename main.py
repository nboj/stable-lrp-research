from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
import sys
import src.sd as sd
import src.utils as utils

from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from IPython.display import display
def lrp_denoise_step(x_t, eps_theta, alpha_t, beta_t, R_out, eps=1e-6):
    w_xt   = alpha_t.sqrt()          # scalar
    w_eps  = beta_t.sqrt()           # scalar

    Z      = w_xt * x_t + w_eps * eps_theta + eps * torch.sign(x_t)
    R_xt   = (w_xt  * x_t       / Z) * R_out      # goes to next x_t
    R_eps  = (w_eps * eps_theta / Z) * R_out      # goes to UNet
    return R_xt, R_eps

if __name__ == "__main__":
    # Load the pre-trained model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to('cuda')  # Use "cpu" if you don't have a compatible GPU
    #pipe.enable_sequential_cpu_offload()

    # Get the underlying UNet model
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler
    torch_device = "cuda"
    #prompt = ["""
    #An ancient, overgrown temple in a dense jungle, illuminated by the soft light of early morning.
    #"""]
    prompt = ["""
    A fluffy cat sitting on a wooden window sill.
    """]
    #prompt = ["""
    #A woman riding her bike
    #"""]
    #prompt = ["""
    #A bald man drinking a soda
    #"""]
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 30 # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    #generator = torch.Generator(device=torch_device)  # Seed generator to create the initial latent noise
    batch_size = len(prompt)
    #seed = torch.seed()
    seed = 13961931025228480435
    generator = torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]

    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    #print(text_embeddings.shape)
    print(seed)

    # GOOD CAT = 6918687713540928788
    #A photo of a cat sitting on a window sill.

    with torch.no_grad():
        activations = []
    layers = []
    handles = []

    def save_activation(name, layer):
        def hook(model, input, output):
            layers.append(layer)
            activations.append((name, output))
        return hook




    # FOR FULL RELEVANCE PROPAGATION

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        device=torch_device,
        generator=generator
    )
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps)



    # latent_res = [latents]
    # noise_res = [latents]
    lays = None
    with torch.no_grad():
        for idx, t in enumerate(tqdm(scheduler.timesteps-1)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            noise_pred, samps, time, weights = sd.forward_unet(unet, latent_model_input, t, text_embeddings)

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            for handle in handles:
                handle.remove()

            if lays is None:
                lays = layers


            # save_to_h5py(idx, latents, noise_pred, activations, samps, time, latent_model_input, weights);
            # data_to_save = {
            #    "lat": latents.detach(),
            #    "pred": noise_pred,
            #    "activations": activations,
            #    "samps": samps,
            #    "time": time,
            #    "input": latent_model_input,
            #    "weights": weights
            # }
            # torch.save(data_to_save, f'./activations/data-{idx}.pt')

            del activations, layers, handles, samps, time, latent_model_input, weights
            torch.cuda.empty_cache()
            activations, layers, handles = [], [], []

        for name, layer in unet.named_modules():
            handles.append(layer.register_forward_hook(save_activation(name,layer)))

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        noise_pred, samps, time, weights = sd.forward_unet(unet, latent_model_input, t, text_embeddings)

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        for handle in handles:
            handle.remove()

        if lays is None:
            lays = layers

    # scale and decode the image latents with vae
    X = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(X).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image








    ## YIKEs
    prev = None
    R = []
    values = []
    keys = []
    weights = []


    for handle in handles:
        handle.remove()

    timestep = scheduler.timesteps[len(scheduler.timesteps)-1]

    prev = activations[562][1]
    activations[562] = (activations[562][0], prev)

    prev, q, k, v, w = utils.apply_lrp(unet, vae, lays, activations, samps, time, text_embeddings, latents, weights)
    prev = utils.norm_rel(prev)
    R.append(prev.detach().cpu())
    values.append(k)
    keys.append(k)
    weights.append(w)
    del activations, samples, time, initial_latents, weights, q, k, v









    tokens = tokenizer.convert_ids_to_tokens(text_input.input_ids[0])
    image2 = None
    total_values = None
    total_keys = None

    L = len(R)-1
    vae = vae.cpu()
    # for idx, (r, latent, noise) in enumerate(zip(reversed(R), latent_res[:-1], noise_res[:-1])):
    for idx, (r, vs, ks) in enumerate(zip(reversed(R), values, keys)):
        latent = latents
        pred = noise_pred
        # print(r.shape)

        uncond, text = (utils.norm_rel(r)*1e5).chunk(2)
        comb = 1 / 0.18215 * (uncond + guidance_scale * (text - uncond))
        uncond = 1 / 0.18215 *uncond
        text = 1 / 0.18215 *text
        X = 1 / 0.18215 * latent
        pred = 1 / 0.18215 * pred

        with torch.no_grad():
            lrp = vae.decode(comb).sample
            image = vae.decode(X.detach().cpu()).sample
            noise = vae.decode(pred.detach().cpu()).sample

        total_value = None;
        for v in vs:
            if total_value is None:
                total_value = v[0][0]
            else:
                total_value += v[0][0]
        total_key = None;
        for k in ks:
            if total_key is None:
                total_key = k[0][0]
            else:
                total_key += k[0][0]

        uncond_text, cond_text = total_value
        comb = uncond_text + guidance_scale * (cond_text - uncond_text)
        utils.visualize_text_relevance(tokens, comb.cpu().sum(dim=-1), save_path=f'./results/lrp2a-{idx}.png')
        uncond_text, cond_text = total_key
        comb = uncond_text + guidance_scale * (cond_text - uncond_text)
        utils.visualize_text_relevance(tokens, comb.cpu().sum(dim=-1), save_path=f'./results/lrp2b-{idx}.png')


        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        image.save(f'./results/noise-{idx}.png')
        display(image)

        noise = (noise / 2 + 0.5).clamp(0, 1).squeeze()
        noise = (noise.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        noise = Image.fromarray(noise)
        noise.save(f'./results/noise_pred-{idx}.png')
        display(noise)

        utils.heatmap(lrp[0].cpu().sum(axis=0), 5, 5, save_path=f'./results/lrp2-{idx}.png', log=True)
        utils.heatmap(lrp[0].cpu().sum(axis=0), 5, 5, save_path=f'./results/lrp1-{idx}.png')
        print('\n\n\n\n\n')

