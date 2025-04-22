import torch
from tqdm import trange

from comfy.samplers import KSAMPLER, CFGGuider, sampling_function


class FlowEditGuider(CFGGuider):
    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self.cfgs = {}

    def set_conds(self, **kwargs):
        self.inner_set_conds(kwargs)

    def set_cfgs(self, **kwargs):
        self.cfgs = {**kwargs}

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        latent_type = model_options['transformer_options']['latent_type']
        positive = self.conds.get(f'{latent_type}_positive', None)
        negative = self.conds.get(f'{latent_type}_negative', None)
        cfg = self.cfgs.get(latent_type, self.cfg)
        return sampling_function(self.inner_model, x, timestep, negative, positive, cfg, model_options=model_options, seed=seed)


class FlowEditGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "source_cond": ("CONDITIONING", ),
                        "target_cond": ("CONDITIONING", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "fluxtapoz"

    def get_guider(self, model, source_cond, target_cond):
        guider = FlowEditGuider(model)
        guider.set_conds(source_positive=source_cond, target_positive=target_cond)
        guider.set_cfg(1.0)
        return (guider,)
    



class FlowEditCFGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "source_pos": ("CONDITIONING", ),
                        "source_neg": ("CONDITIONING", ),
                        "target_pos": ("CONDITIONING", ),
                        "target_neg": ("CONDITIONING", ),
                        "source_cfg": ("FLOAT", {"default": 3.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.01 }),
                        "target_cfg": ("FLOAT", {"default": 13.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.01 }),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "fluxtapoz"

    def get_guider(self, model, source_pos, source_neg, target_pos, target_neg, source_cfg, target_cfg):
        guider = FlowEditGuider(model)
        guider.set_conds(source_positive=source_pos, source_negative=source_neg, target_positive=target_pos, target_negative=target_neg)
        guider.set_cfgs(source=source_cfg, target=target_cfg)
        return (guider,)

# New hybrid FlowEdit guider allowing separate source/target models
class FlowEditGuiderHybrid(CFGGuider):
    def __init__(self, src_model_patcher, tgt_model_patcher):
        # Initialize with source model patcher; we will handle both patchers in outer_sample
        super().__init__(src_model_patcher)
        self.src_model = src_model_patcher
        self.tgt_model = tgt_model_patcher
        self.cfgs = {}

    def set_conds(self, **kwargs):
        self.inner_set_conds(kwargs)

    def set_cfgs(self, **kwargs):
        self.cfgs = {**kwargs}

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # Select appropriate inner model based on latent_type
        latent_type = model_options['transformer_options']['latent_type']
        positive = self.conds.get(f'{latent_type}_positive', None)
        negative = self.conds.get(f'{latent_type}_negative', None)
        cfg = self.cfgs.get(latent_type, self.cfg)
        # Use inner models prepared in outer_sample
        model = self.src_inner_model if latent_type.startswith('source') else self.tgt_inner_model
        return sampling_function(model, x, timestep, negative, positive, cfg,
                                 model_options=model_options, seed=seed)

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        import comfy.sampler_helpers as sh
        from comfy.samplers import cast_to_load_options

        # Prepare both source and target models for sampling
        self.src_inner_model, conds, src_loaded = sh.prepare_sampling(self.src_model, noise.shape, self.conds, self.model_options)
        self.tgt_inner_model, _, tgt_loaded = sh.prepare_sampling(self.tgt_model, noise.shape, self.conds, self.model_options)
        device = self.src_model.load_device
        # Prepare denoise mask if provided
        if denoise_mask is not None:
            denoise_mask = sh.prepare_mask(denoise_mask, noise.shape, device)
        # Move tensors to correct device
        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        # Cast model options for device and dtype
        cast_to_load_options(self.model_options, device=device, dtype=self.src_model.model_dtype())
        try:
            # Inject both models
            self.src_model.pre_run()
            self.tgt_model.pre_run()
            # For CFGGuider hooks, set inner_model to source by default
            self.inner_model = self.src_inner_model
            # Run inner sampling, will call predict_noise which delegates correctly
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            # Cleanup both patchers and loaded auxiliary models
            self.src_model.cleanup()
            self.tgt_model.cleanup()
            sh.cleanup_models(conds, src_loaded + tgt_loaded)
            # Remove references
            del self.inner_model
            del self.src_inner_model
            del self.tgt_inner_model
            del src_loaded
            del tgt_loaded
        return output

class FlowEditGuiderHybridNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "source_model": ("MODEL",),
                        "target_model": ("MODEL",),
                        "source_cond": ("CONDITIONING",),
                        "target_cond": ("CONDITIONING",),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider_hybrid"
    CATEGORY = "fluxtapoz"

    def get_guider_hybrid(self, source_model, target_model, source_cond, target_cond):
        guider = FlowEditGuiderHybrid(source_model, target_model)
        guider.set_conds(source_positive=source_cond, target_positive=target_cond)
        guider.set_cfg(1.0)
        return (guider,)

class FlowEditCFGGuiderHybridNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "source_model": ("MODEL",),
                        "target_model": ("MODEL",),
                        "source_pos": ("CONDITIONING",),
                        "source_neg": ("CONDITIONING",),
                        "target_pos": ("CONDITIONING",),
                        "target_neg": ("CONDITIONING",),
                        "source_cfg": ("FLOAT", {"default": 3.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
                        "target_cfg": ("FLOAT", {"default": 13.5, "min": 0, "max": 0xffffffffffffffff, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider_hybrid"
    CATEGORY = "fluxtapoz"

    def get_guider_hybrid(self, source_model, target_model,
                          source_pos, source_neg, target_pos, target_neg,
                          source_cfg, target_cfg):
        guider = FlowEditGuiderHybrid(source_model, target_model)
        guider.set_conds(source_positive=source_pos, source_negative=source_neg,
                         target_positive=target_pos, target_negative=target_neg)
        guider.set_cfgs(source=source_cfg, target=target_cfg)
        return (guider,)


def get_flowedit_sample(skip_steps, refine_steps, seed, n_avg=1):
    @torch.no_grad()
    def flowedit_sample(model, x_init, sigmas, extra_args=None, callback=None, disable=None):
        generator = torch.Generator().manual_seed(seed)
        extra_args = {} if extra_args is None else extra_args

        model_options = extra_args.get('model_options', {})
        transformer_options = model_options.get('transformer_options', {})
        transformer_options = {**transformer_options}
        model_options['transformer_options'] = transformer_options
        extra_args['model_options'] = model_options

        source_extra_args = {**extra_args, 'model_options': { 'transformer_options': { **transformer_options,'latent_type': 'source '} }}

        sigmas = sigmas[skip_steps:]

        x_tgt = x_init.clone()
        N = len(sigmas)-1
        s_in = x_init.new_ones([x_init.shape[0]])

        for i in trange(N, disable=disable):
            sigma = sigmas[i]
            delta_sigma = sigmas[i+1] - sigma
            vs = []
            # average over n_avg noise samples per step for smoother guidance
            if i < N - refine_steps:
                for _ in range(n_avg):
                    noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)
                    zt_src = (1 - sigma) * x_init + sigma * noise
                    zt_tgt = x_tgt + zt_src - x_init
                    transformer_options['latent_type'] = 'source'
                    source_extra_args['model_options']['transformer_options']['latent_type'] = 'source'
                    vt_src = model(zt_src, sigma * s_in, **source_extra_args)
                    transformer_options['latent_type'] = 'target'
                    vt_tgt = model(zt_tgt, sigma * s_in, **extra_args)
                    vs.append(vt_tgt - vt_src)
                v_delta = torch.stack(vs, dim=0).mean(0)
            else:
                noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)
                zt_src = (1 - sigma) * x_init + sigma * noise
                if i == N - refine_steps:
                    x_tgt = x_tgt + zt_src - x_init
                zt_tgt = x_tgt
                transformer_options['latent_type'] = 'target'
                vt_tgt = model(zt_tgt, sigma * s_in, **extra_args)
                v_delta = vt_tgt
            # mean delta across samples
            x_tgt += delta_sigma * v_delta
            if callback is not None:
                callback({'x': x_tgt, 'denoised': x_tgt,
                          'i': i + skip_steps,
                          'sigma': sigma,
                          'sigma_hat': sigma})

        return x_tgt
    
    return flowedit_sample


class FlowEditSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "skip_steps": ("INT", {"default": 4, "min": 0, "max": 0xffffffffffffffff }),
            "refine_steps": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
            "n_avg": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff }),
        }, "optional": {
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "fluxtapoz"

    def build(self, skip_steps, refine_steps, seed, n_avg):
        sampler = KSAMPLER(get_flowedit_sample(skip_steps, refine_steps, seed, n_avg))
        return (sampler, )
