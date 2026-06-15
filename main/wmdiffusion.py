from typing import Callable, List, Optional, Union, Any, Dict
from functools import partial
import numpy as np
import copy
from dataclasses import dataclass
import PIL

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from torch.utils.checkpoint import checkpoint
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, UniDiffuserPipeline
from diffusers.utils import BaseOutput

from rf_inversion import generate_eta_values


@dataclass
class ModifiedStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    # nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]


class WatermarkStableDiffusionPipeline(StableDiffusion3Pipeline):
    def __init__(self,
                 transformer,
                 scheduler,
                 vae,
                 text_encoder,
                 tokenizer,
                 text_encoder_2,
                 tokenizer_2,
                 text_encoder_3,
                 tokenizer_3,
                 ):
        super(WatermarkStableDiffusionPipeline, self).__init__(transformer,
                                                               scheduler,
                                                               vae,
                                                               text_encoder,
                                                               tokenizer,
                                                               text_encoder_2,
                                                               tokenizer_2,
                                                               text_encoder_3,
                                                               tokenizer_3,
                                                               )

    # Generate image in tensor format
    def decode_latents_tensor(self, latents):
        # latents = 1 / self.vae.config.scaling_factor * latents
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        # image = self.image_processor.postprocess(image, output_type="pt")
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Convert image tensor into numpy, so that it can be converted into PIL image later
    def img_tensor_to_numpy(self, tensor):
        return tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()

    # Accept keywords args in torch.checkpoint
    # def transformer_custom_forward(self,
    #                                sample: torch.FloatTensor,
    #                                timestep: Union[torch.Tensor, float, int],
    #                                encoder_hidden_states: torch.Tensor,
    #                                pooled_projections: Optional[torch.FloatTensor] = None,
    #                                joint_attention_kwargs: Optional[Dict[str, Any]] = None, ):
    #     return self.transformer(hidden_states=sample,
    #                             timestep=timestep,
    #                             encoder_hidden_states=encoder_hidden_states,
    #                             pooled_projections=pooled_projections,
    #                             joint_attention_kwargs=joint_attention_kwargs)[0]

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            prompt_3: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: int = 28,
            timesteps: List[int] = None,
            guidance_scale: float = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            ### added parameters
            use_trainable_latents: bool = False,
            init_latents: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array` or `tensor`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        # )

        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]
        #
        # device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # # 3. Encode input prompt
        # with torch.no_grad():
        #     # prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        #     #     prompt,
        #     #     device,
        #     #     num_images_per_prompt,
        #     #     do_classifier_free_guidance,
        #     #     negative_prompt,
        #     #     prompt_embeds=prompt_embeds,
        #     #     negative_prompt_embeds=negative_prompt_embeds,
        #     # )
        #     (
        #         prompt_embeds,
        #         negative_prompt_embeds,
        #         pooled_prompt_embeds,
        #         negative_pooled_prompt_embeds,
        #     ) = self.encode_prompt(
        #         prompt=prompt,
        #         prompt_2=prompt_2,
        #         prompt_3=prompt_3,
        #         negative_prompt=negative_prompt,
        #         negative_prompt_2=negative_prompt_2,
        #         negative_prompt_3=negative_prompt_3,
        #         do_classifier_free_guidance=do_classifier_free_guidance,
        #         prompt_embeds=prompt_embeds,
        #         negative_prompt_embeds=negative_prompt_embeds,
        #         pooled_prompt_embeds=pooled_prompt_embeds,
        #         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #         device=device,
        #         num_images_per_prompt=num_images_per_prompt
        #     )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # # 4. Prepare timesteps
        # timesteps, num_inference_steps = self.retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        # num_channels_latents = self.unet.in_channels
        if not use_trainable_latents:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            init_latents = copy.deepcopy(latents)
        else:
            if init_latents is None:
                raise ValueError(f"We must have a initial trainable latents.")
            else:
                latents = init_latents

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict the noise residual
                if not use_trainable_latents:
                    # noise_pred = self.unet(
                    #     latent_model_input,
                    #     t,
                    #     encoder_hidden_states=prompt_embeds,
                    #     cross_attention_kwargs=cross_attention_kwargs,
                    # ).sample
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=joint_attention_kwargs,
                    ).sample
                else:
                    # noise_pred = checkpoint(self.unet_custom_forward, latent_model_input, t, prompt_embeds,
                    #                         cross_attention_kwargs).sample
                    noise_pred = checkpoint(self.transformer_custom_forward,
                                            latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds,
                                            joint_attention_kwargs)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents_dtype = latents.dtype
                # latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                # # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()
                #     if callback is not None and i % callback_steps == 0:
                #         callback(i, t, latents)

        if output_type == "latent":
            image = latents
            # has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        elif output_type == 'tensor':
            # 8. Post-processing
            image = self.decode_latents_tensor(latents)
            # has_nsfw_concept = None
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        if not return_dict:
            return (image, init_latents)

        return ModifiedStableDiffusionPipelineOutput(images=image,
                                                     # nsfw_content_detected=has_nsfw_concept,
                                                     init_latents=init_latents)


class WMDetectStableDiffusionPipeline(WatermarkStableDiffusionPipeline):
    def __init__(self,
                 transformer,
                 scheduler,
                 vae,
                 text_encoder,
                 tokenizer,
                 text_encoder_2,
                 tokenizer_2,
                 text_encoder_3,
                 tokenizer_3,
                 ):
        super(WMDetectStableDiffusionPipeline, self).__init__(
            transformer,
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            text_encoder_3,
            tokenizer_3, )
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    ######### From Tree-Rings repo, for inverse diffusion model ########
    # @torch.inference_mode()
    # # @torch.no_grad()
    # def get_text_embedding(self, prompt):
    #     # text_input_ids = self.tokenizer(
    #     #     prompt,
    #     #     padding="max_length",
    #     #     truncation=True,
    #     #     max_length=self.tokenizer.model_max_length,
    #     #     return_tensors="pt",
    #     # ).input_ids
    #     text_input_ids = self.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=self.tokenizer_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
    #     return text_embeddings

    # The reverse of decode_latents_tensor()
    @torch.inference_mode()
    # @torch.no_grad()
    def get_image_latents(self, image: torch.Tensor, sample=True, rng_generator=None):
        image = 2.0 * image - 1.0
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * self.vae.config.scaling_factor
        return latents

    def backward_ddim(self, x_t, alpha_t, alpha_tm1, eps_xt):
        """ from noise to image"""
        return (
                alpha_tm1 ** 0.5
                * (
                        (alpha_t ** -0.5 - alpha_tm1 ** -0.5) * x_t
                        + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
                )
                + x_t
        )

    @torch.inference_mode()
    # @torch.no_grad()
    def backward_diffusion(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            pooled_prompt_embeds=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # if old_text_embeddings is not None and new_text_embeddings is not None:
        #     prompt_to_prompt = True
        # else:
        #     prompt_to_prompt = False

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            # if prompt_to_prompt:
            #     if i < use_old_emb_i:
            #         text_embeddings = old_text_embeddings
            #     else:
            #         text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            # noise_pred = self.unet(
            #     latent_model_input, t, encoder_hidden_states=text_embeddings
            # ).sample
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False
            )[0]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # ddim 
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = self.backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    @torch.inference_mode()
    def interpolated_inversion(
            self,
            latents,
            gamma,
            DTYPE,
            prompt="",
            num_steps=28,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            seed=42
    ):

        # using FlowMatchEulerDiscreteScheduler
        self.scheduler.set_timesteps(num_steps, device=self.device)

        # check if it has sigma avaible
        if not hasattr(self.scheduler, "sigmas"):
            raise Exception(
                "Cannot find sigmas variable in scheduler. Please use FlowMatchEulerDiscreteScheduler to doing RF Inversion")

        # we get timestep directy from sigmas
        timesteps = self.scheduler.sigmas
        timesteps = torch.flip(timesteps, dims=[0])

        # # Getting null-text embedning
        # (
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds
        # ) = self.encode_prompt(  # null text
        #     prompt=prompt,
        #     prompt_2=prompt,
        #     prompt_3=prompt,
        # )

        # generate gaussain noise with seed
        # set_seed(seed)
        target_noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)

        # # Image inversion with interpolated velocity field.  t goes from 0.0 to 1.0
        with self.progress_bar(total=len(timesteps) - 1) as progress_bar:
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((latents.shape[0],), t_curr * 1000, dtype=latents.dtype, device=latents.device)

                # Null-text velocity
                pred_velocity = self.transformer(
                    hidden_states=latents,
                    timestep=t_vec,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # Prevents precision issues
                latents = latents.to(torch.float32)
                pred_velocity = pred_velocity.to(torch.float32)

                # Target noise velocity
                target_noise_velocity = (target_noise - latents) / (1.0 - t_curr)

                # interpolated velocity
                interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * pred_velocity

                # one step Euler, similar to pipeline.scheduler.step but in the forward to noise instead of denosing
                latents = latents + (t_prev - t_curr) * interpolated_velocity

                latents = latents.to(DTYPE)
                progress_bar.update()

        return latents

    def transformer_custom_forward(self,
                                   sample: torch.FloatTensor,
                                   timestep: Union[torch.Tensor, float, int],
                                   encoder_hidden_states: torch.Tensor,
                                   pooled_projections: Optional[torch.FloatTensor] = None,
                                  ):
        return self.transformer(hidden_states=sample,
                                timestep=timestep,
                                encoder_hidden_states=encoder_hidden_states,
                                pooled_projections=pooled_projections,
                               )[0]

    @torch.inference_mode()
    def interpolated_denoise(
            self,
            img_latents,
            eta_base,  # base eta value
            eta_trend,  # constant, linear_increase, linear_decrease
            start_step,  # 0-based indexing, closed interval
            end_step,  # 0-based indexing, open interval
            inversed_latents,  # can be none if not using inversed latents
            use_inversed_latents=True,
            guidance_scale=3.5,
            prompt='photo of a tiger',
            DTYPE=torch.float32,
            num_steps=28,
            seed=42,
            negative_prompt_embeds=None,
            prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            pooled_prompt_embeds=None):

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_steps, self.device)

        # # Getting text embedning
        # (
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds
        # ) = pipeline.encode_prompt(
        #     prompt=prompt,
        #     prompt_2=prompt,
        #     prompt_3=prompt
        # )

        if use_inversed_latents:
            latents = inversed_latents
        else:
            # set_seed(seed)
            latents = torch.randn_like(img_latents)

        target_img = img_latents.clone().to(torch.float32)

        # get the eta values for each steps in
        eta_values = generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)

        # handle guidance scale if need
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        with self.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                # Editing text velocity
                pred_velocity = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                # pred_velocity = checkpoint(self.transformer_custom_forward,
                #                            latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)

                # perform guidance scale
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = pred_velocity.chunk(2)
                    pred_velocity = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Prevents precision issues
                latents = latents.to(torch.float32)
                pred_velocity = pred_velocity.to(torch.float32)

                # Target image velocity
                t_curr = t / self.scheduler.config.num_train_timesteps
                target_velocity = -(target_img - latents) / t_curr

                # interpolated velocity
                eta = eta_values[i]
                interpolate_velocity = pred_velocity + eta * (target_velocity - pred_velocity)

                # denosing
                latents = self.scheduler.step(interpolate_velocity, t, latents, return_dict=False)[0]

                latents = latents.to(DTYPE)
                progress_bar.update()

        return latents
