"""
Block-swapping for LTX transformer using ModelOffloader pattern.

Keeps only a subset of transformer blocks in GPU memory at a time,
swapping them in/out from CPU as needed using ThreadPoolExecutor
and CUDA streams for efficient async transfers.

Based on the working block swapping implementation from h1111/modules/custom_offloading_utils.py.
"""

import types

import torch
from torch import nn

from ltx_core.model.transformer.model import LTXModel, X0Model

from .custom_offloading_utils import ModelOffloader, clean_memory_on_device, weighs_to_device


def log_memory(label: str) -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEM] {label}: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")


def enable_block_swap(
    model: X0Model | LTXModel,
    blocks_in_memory: int = 6,
    device: torch.device | str = "cuda",
) -> ModelOffloader:
    """
    Enable block swapping on an existing X0Model or LTXModel using ModelOffloader.

    This function:
    1. Creates a ModelOffloader for async block transfers
    2. Prepares initial block positions (first N on GPU, rest on CPU)
    3. Monkey-patches _process_transformer_blocks to use wait/submit pattern

    Args:
        model: X0Model (wraps LTXModel) or LTXModel directly.
        blocks_in_memory: Number of transformer blocks to keep in GPU (default: 6).
        device: Target GPU device.

    Returns:
        ModelOffloader instance for controlling the swapping behavior.

    Example:
        transformer = model_ledger.transformer()
        offloader = enable_block_swap(transformer, blocks_in_memory=6)
        # ... run inference ...
        # Cleanup handled automatically
    """
    # Get the underlying LTXModel
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    device = torch.device(device) if isinstance(device, str) else device
    num_blocks = len(ltx_model.transformer_blocks)
    blocks_to_swap = num_blocks - blocks_in_memory

    if blocks_to_swap <= 0:
        print(f"[BlockSwap] blocks_in_memory ({blocks_in_memory}) >= num_blocks ({num_blocks}), no swapping needed")
        return None

    # Get reference to the actual blocks (not a copy!)
    blocks = ltx_model.transformer_blocks

    # Create offloader with ThreadPoolExecutor for async transfers
    offloader = ModelOffloader(
        block_type="ltx_transformer_block",
        blocks=blocks,
        num_blocks=num_blocks,
        blocks_to_swap=blocks_to_swap,
        supports_backward=False,
        device=device,
    )

    # Store on model for access in forward pass
    ltx_model._block_swap_offloader = offloader
    ltx_model._blocks_to_swap = blocks_to_swap
    ltx_model._blocks_ref = blocks  # Store reference for forward pass
    if isinstance(model, X0Model):
        model._block_swap_offloader = offloader
        model._blocks_to_swap = blocks_to_swap
        model._blocks_ref = blocks

    # Prepare block positions: first (num_blocks - blocks_to_swap) on GPU, rest on CPU
    offloader.prepare_block_devices_before_forward(blocks)

    # Store original method for potential restoration
    ltx_model._original_process_transformer_blocks = ltx_model._process_transformer_blocks

    # Create replacement method using wait/submit pattern
    def block_swap_process_transformer_blocks(self, video, audio, perturbations):
        """Process transformer blocks with block swapping using wait/submit pattern."""
        offloader = self._block_swap_offloader
        blocks = self._blocks_ref  # Use stored reference, not a copy

        for block_idx, block in enumerate(self.transformer_blocks):
            # Wait for this block to be ready BEFORE using it
            offloader.wait_for_block(block_idx)

            # Process the block
            video, audio = block(
                video=video,
                audio=audio,
                perturbations=perturbations,
            )

            # Submit swap for next iteration AFTER using block
            offloader.submit_move_blocks_forward(blocks, block_idx)

        return video, audio

    # Monkey-patch the method
    ltx_model._process_transformer_blocks = types.MethodType(block_swap_process_transformer_blocks, ltx_model)

    print(f"[BlockSwap] Enabled: {blocks_in_memory}/{num_blocks} blocks in GPU, {blocks_to_swap} swapping")
    return offloader


def enable_block_swap_with_activation_offload(
    model: X0Model | LTXModel,
    blocks_in_memory: int = 1,
    device: torch.device | str = "cuda",
    verbose: bool = False,
    temporal_chunk_size: int = 0,
) -> ModelOffloader:
    """
    Enable block swapping WITH activation offloading for extreme memory savings.

    Unlike regular block swapping which only moves weights to CPU, this also
    moves activations (vx, ax) to CPU between blocks. This allows processing
    arbitrarily long sequences that wouldn't fit in VRAM otherwise.

    Trade-off: ~10-20x slower due to CPU-GPU transfers per block, but uses
    minimal GPU memory regardless of sequence length.

    Args:
        model: X0Model (wraps LTXModel) or LTXModel directly.
        blocks_in_memory: Number of transformer blocks to keep in GPU (default: 1).
        device: Target GPU device.
        verbose: If True, log memory at each block.
        temporal_chunk_size: If > 0, process video in chunks of this many tokens.
            This allows processing very long videos by streaming K/V from CPU.

    Returns:
        ModelOffloader instance for controlling the swapping behavior.
    """
    from dataclasses import replace

    # Get the underlying LTXModel
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    device = torch.device(device) if isinstance(device, str) else device
    num_blocks = len(ltx_model.transformer_blocks)
    blocks_to_swap = num_blocks - blocks_in_memory

    if blocks_to_swap <= 0:
        print(f"[BlockSwap+ActOffload] blocks_in_memory ({blocks_in_memory}) >= num_blocks ({num_blocks}), no swapping needed")
        return None

    # Get reference to the actual blocks
    blocks = ltx_model.transformer_blocks

    # Create offloader with ThreadPoolExecutor for async transfers
    offloader = ModelOffloader(
        block_type="ltx_transformer_block",
        blocks=blocks,
        num_blocks=num_blocks,
        blocks_to_swap=blocks_to_swap,
        supports_backward=False,
        device=device,
    )

    # Store on model for access in forward pass
    ltx_model._block_swap_offloader = offloader
    ltx_model._blocks_to_swap = blocks_to_swap
    ltx_model._blocks_ref = blocks
    ltx_model._activation_offload_verbose = verbose
    ltx_model._temporal_chunk_size = temporal_chunk_size
    if isinstance(model, X0Model):
        model._block_swap_offloader = offloader
        model._blocks_to_swap = blocks_to_swap
        model._blocks_ref = blocks
        model._temporal_chunk_size = temporal_chunk_size

    # Prepare block positions: first (num_blocks - blocks_to_swap) on GPU, rest on CPU
    offloader.prepare_block_devices_before_forward(blocks)

    # Store original method for potential restoration
    ltx_model._original_process_transformer_blocks = ltx_model._process_transformer_blocks

    # Create replacement method with activation offloading
    def block_swap_process_transformer_blocks_with_activation_offload(self, video, audio, perturbations):
        """Process transformer blocks with block AND activation offloading."""
        offloader = self._block_swap_offloader
        blocks = self._blocks_ref
        verbose = getattr(self, '_activation_offload_verbose', False)
        temporal_chunk_size = getattr(self, '_temporal_chunk_size', 0)
        device = offloader.device

        # Extract activations - start on CPU
        vx = video.x.cpu() if video is not None and video.x is not None else None
        ax = audio.x.cpu() if audio is not None and audio.x is not None else None

        if verbose:
            log_memory("Before block loop")

        use_temporal_chunking = temporal_chunk_size > 0 and vx is not None

        for block_idx, block in enumerate(self.transformer_blocks):
            # Wait for block weights to be ready
            offloader.wait_for_block(block_idx)

            if use_temporal_chunking:
                # Use temporal chunking - keeps vx/ax on CPU, processes in chunks
                # Create video/audio args with CPU activations
                video_cpu = replace(video, x=vx) if video is not None else None
                audio_cpu = replace(audio, x=ax) if audio is not None else None

                if verbose and block_idx % 10 == 0:
                    log_memory(f"Block {block_idx} (before, chunked)")

                # Process using chunked forward (handles its own GPU transfers)
                video_out, audio_out = block.forward_chunked(
                    video=video_cpu,
                    audio=audio_cpu,
                    perturbations=perturbations,
                    chunk_size=temporal_chunk_size,
                    device=device,
                )

                # Results are already on CPU from forward_chunked
                vx = video_out.x if video_out is not None else None
                ax = audio_out.x if audio_out is not None else None

                del video_cpu, audio_cpu, video_out, audio_out
            else:
                # Original path - move full activations to GPU
                if vx is not None:
                    vx_gpu = vx.to(device, non_blocking=True)
                else:
                    vx_gpu = None

                if ax is not None:
                    ax_gpu = ax.to(device, non_blocking=True)
                else:
                    ax_gpu = None

                # Sync to ensure activations are on GPU
                torch.cuda.synchronize()

                # Create video/audio args with GPU activations
                video_gpu = replace(video, x=vx_gpu) if video is not None else None
                audio_gpu = replace(audio, x=ax_gpu) if audio is not None else None

                if verbose and block_idx % 10 == 0:
                    log_memory(f"Block {block_idx} (before)")

                # Process the block
                video_out, audio_out = block(
                    video=video_gpu,
                    audio=audio_gpu,
                    perturbations=perturbations,
                )

                # Extract results
                vx_result = video_out.x if video_out is not None else None
                ax_result = audio_out.x if audio_out is not None else None

                # Move activations back to CPU immediately
                if vx_result is not None:
                    vx = vx_result.cpu()
                    del vx_result
                if ax_result is not None:
                    ax = ax_result.cpu()
                    del ax_result

                # Clean up GPU tensors
                del vx_gpu, ax_gpu, video_gpu, audio_gpu, video_out, audio_out

            # Submit swap for next block's weights
            offloader.submit_move_blocks_forward(blocks, block_idx)

            # Clear GPU cache to actually free memory
            torch.cuda.empty_cache()

            if verbose and block_idx % 10 == 0:
                log_memory(f"Block {block_idx} (after)")

        # Final move back to GPU for output
        if vx is not None:
            vx = vx.to(device)
        if ax is not None:
            ax = ax.to(device)

        if verbose:
            log_memory("After block loop")

        # Return with final activations
        video_final = replace(video, x=vx) if video is not None else None
        audio_final = replace(audio, x=ax) if audio is not None else None

        return video_final, audio_final

    # Monkey-patch the method
    ltx_model._process_transformer_blocks = types.MethodType(
        block_swap_process_transformer_blocks_with_activation_offload, ltx_model
    )

    print(f"[BlockSwap+ActOffload] Enabled: {blocks_in_memory}/{num_blocks} blocks in GPU, {blocks_to_swap} swapping, activations offloaded")
    return offloader


def disable_block_swap(model: X0Model | LTXModel) -> None:
    """
    Disable block swapping and restore original behavior.

    Args:
        model: Model that had block swapping enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    if hasattr(ltx_model, "_block_swap_offloader"):
        offloader = ltx_model._block_swap_offloader
        device = offloader.device

        # Move all blocks back to GPU
        for block in ltx_model.transformer_blocks:
            block.to(device)
            weighs_to_device(block, device)

        # Restore original method if saved
        if hasattr(ltx_model, "_original_process_transformer_blocks"):
            ltx_model._process_transformer_blocks = ltx_model._original_process_transformer_blocks
            del ltx_model._original_process_transformer_blocks

        del ltx_model._block_swap_offloader
        del ltx_model._blocks_to_swap

        if isinstance(model, X0Model):
            if hasattr(model, "_block_swap_offloader"):
                del model._block_swap_offloader
            if hasattr(model, "_blocks_to_swap"):
                del model._blocks_to_swap

        clean_memory_on_device(device)
        print(f"[BlockSwap] Disabled: all blocks moved to GPU")


def get_block_swap_offloader(model: X0Model | LTXModel) -> ModelOffloader | None:
    """
    Get the ModelOffloader for a model if block swapping is enabled.

    Args:
        model: Model to check.

    Returns:
        ModelOffloader instance or None if not enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    return getattr(ltx_model, "_block_swap_offloader", None)


def offload_all_blocks(model: X0Model | LTXModel) -> None:
    """
    Offload all transformer blocks to CPU and cleanup the offloader.

    Used to free GPU memory after inference is complete.

    Args:
        model: Model with block swapping enabled.
    """
    if isinstance(model, X0Model):
        ltx_model = model.velocity_model
    else:
        ltx_model = model

    offloader = getattr(ltx_model, "_block_swap_offloader", None)
    if offloader is None:
        return

    device = offloader.device

    # Wait for any pending async operations
    for idx in range(len(ltx_model.transformer_blocks)):
        if idx in offloader.futures:
            offloader._wait_blocks_move(idx)

    # Shutdown the ThreadPoolExecutor
    offloader.thread_pool.shutdown(wait=True)
    offloader.futures.clear()

    # Synchronize CUDA before moving blocks
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Move all blocks to CPU
    for block in ltx_model.transformer_blocks:
        weighs_to_device(block, "cpu")

    # Synchronize again after moves complete
    if device.type == "cuda":
        torch.cuda.synchronize()

    clean_memory_on_device(device)
    print("[BlockSwap] All blocks offloaded to CPU")


# =============================================================================
# Text Encoder (Gemma3) Block Swapping
# =============================================================================

def enable_text_encoder_block_swap(
    text_encoder,
    blocks_in_memory: int = 6,
    device: torch.device | str = "cuda",
) -> ModelOffloader:
    """
    Enable block swapping for the Gemma3 text encoder.

    This function:
    1. Creates a ModelOffloader for async block transfers
    2. Prepares initial block positions (first N on GPU, rest on CPU)
    3. Monkey-patches Gemma3TextModel.forward() to use wait/submit pattern

    Args:
        text_encoder: AVGemmaTextEncoderModel or similar with .model attribute.
        blocks_in_memory: Number of decoder layers to keep in GPU (default: 6).
        device: Target GPU device.

    Returns:
        ModelOffloader instance for controlling the swapping behavior.
    """
    # Get the Gemma3TextModel that has the layers
    # Structure: text_encoder.model (Gemma3ForConditionalGeneration).language_model (Gemma3TextModel)
    gemma_text_model = text_encoder.model.language_model
    layers = gemma_text_model.layers

    device = torch.device(device) if isinstance(device, str) else device
    num_layers = len(layers)
    blocks_to_swap = num_layers - blocks_in_memory

    if blocks_to_swap <= 0:
        print(f"[TextEncoderBlockSwap] blocks_in_memory ({blocks_in_memory}) >= num_layers ({num_layers}), no swapping needed")
        return None

    # Create offloader with ThreadPoolExecutor for async transfers
    offloader = ModelOffloader(
        block_type="gemma_decoder_layer",
        blocks=list(layers),
        num_blocks=num_layers,
        blocks_to_swap=blocks_to_swap,
        supports_backward=False,
        device=device,
    )

    # Store on model for access in forward pass
    gemma_text_model._block_swap_offloader = offloader
    gemma_text_model._blocks_to_swap = blocks_to_swap
    gemma_text_model._blocks_ref = list(layers)
    gemma_text_model._block_swap_device = device

    # Move non-layer components to GPU
    gemma_text_model.embed_tokens.to(device)
    gemma_text_model.rotary_emb.to(device)
    gemma_text_model.rotary_emb_local.to(device)
    gemma_text_model.norm.to(device)

    # Move multimodal components to GPU (needed for prompt enhancement with images)
    # Gemma3ForConditionalGeneration has vision_tower and multi_modal_projector
    gemma_model = text_encoder.model  # Gemma3ForConditionalGeneration
    if hasattr(gemma_model, "vision_tower") and gemma_model.vision_tower is not None:
        gemma_model.vision_tower.to(device)
    if hasattr(gemma_model, "multi_modal_projector") and gemma_model.multi_modal_projector is not None:
        gemma_model.multi_modal_projector.to(device)

    # Move text encoder's non-Gemma components to GPU
    # These are used after hidden states extraction
    if hasattr(text_encoder, "feature_extractor_linear"):
        text_encoder.feature_extractor_linear.to(device)
    if hasattr(text_encoder, "embeddings_connector"):
        text_encoder.embeddings_connector.to(device)
    if hasattr(text_encoder, "audio_embeddings_connector"):
        text_encoder.audio_embeddings_connector.to(device)

    # Prepare block positions: first (num_layers - blocks_to_swap) on GPU, rest on CPU
    offloader.prepare_block_devices_before_forward(list(layers))

    # Patch the text encoder's _preprocess_text to use the correct device for tensors
    # The issue is that _preprocess_text creates tensors on self.model.device (CPU),
    # but with block swapping, we need them on GPU for the model and feature extraction.
    text_encoder._original_preprocess_text = text_encoder._preprocess_text

    def device_aware_preprocess_text(text: str, padding_side: str = "left"):
        """Patched _preprocess_text that creates tensors on the correct device."""
        token_pairs = text_encoder.tokenizer.tokenize_with_weights(text)["gemma"]
        # Create tensors on the block swap device (GPU) instead of self.model.device (CPU)
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=device)
        outputs = text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        projected = text_encoder._run_feature_extractor(
            hidden_states=outputs.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        return projected, attention_mask

    text_encoder._preprocess_text = device_aware_preprocess_text

    # Also patch _enhance for prompt enhancement with Gemma
    # The issue is that _enhance uses self.model.device which returns CPU when block swapping is enabled
    text_encoder._original_enhance = text_encoder._enhance

    def device_aware_enhance(
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> str:
        """Patched _enhance that creates tensors on the correct device."""
        if text_encoder.processor is None:
            text_encoder._init_image_processor()
        text = text_encoder.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = text_encoder.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(device)  # Use block swap device instead of self.model.device
        pad_token_id = text_encoder.processor.tokenizer.pad_token_id if text_encoder.processor.tokenizer.pad_token_id is not None else 0

        # Import the padding helper from base_encoder
        from ltx_core.text_encoders.gemma.encoders.base_encoder import _pad_inputs_for_attention_alignment
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            outputs = text_encoder.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = text_encoder.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    text_encoder._enhance = device_aware_enhance

    # Store original forward method
    gemma_text_model._original_forward = gemma_text_model.forward

    # Import required types for the replacement forward
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma3.modeling_gemma3 import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    def block_swap_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        """Forward pass with block swapping for Gemma3TextModel."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare mask arguments
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers with block swapping
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        offloader = self._block_swap_offloader
        blocks = self._blocks_ref

        for block_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Wait for this block to be ready BEFORE using it
            offloader.wait_for_block(block_idx)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            # Submit swap for next iteration AFTER using block
            offloader.submit_move_blocks_forward(blocks, block_idx)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Monkey-patch the forward method
    gemma_text_model.forward = types.MethodType(block_swap_forward, gemma_text_model)

    print(f"[TextEncoderBlockSwap] Enabled: {blocks_in_memory}/{num_layers} layers in GPU, {blocks_to_swap} swapping")
    return offloader


def offload_all_text_encoder_blocks(text_encoder) -> None:
    """
    Offload all text encoder decoder layers to CPU and cleanup the offloader.

    Used to free GPU memory after text encoding is complete.

    Args:
        text_encoder: Text encoder model with block swapping enabled.
    """
    gemma_text_model = text_encoder.model.language_model

    offloader = getattr(gemma_text_model, "_block_swap_offloader", None)
    if offloader is None:
        return

    device = offloader.device
    layers = gemma_text_model.layers

    # Wait for any pending async operations
    for idx in range(len(layers)):
        if idx in offloader.futures:
            offloader._wait_blocks_move(idx)

    # Shutdown the ThreadPoolExecutor
    offloader.thread_pool.shutdown(wait=True)
    offloader.futures.clear()

    # Synchronize CUDA before moving blocks
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Move all layers to CPU
    for layer in layers:
        weighs_to_device(layer, "cpu")

    # Move non-layer components to CPU too
    gemma_text_model.embed_tokens.to("cpu")
    gemma_text_model.rotary_emb.to("cpu")
    gemma_text_model.rotary_emb_local.to("cpu")
    gemma_text_model.norm.to("cpu")

    # Move multimodal components to CPU too
    gemma_model = text_encoder.model  # Gemma3ForConditionalGeneration
    if hasattr(gemma_model, "vision_tower") and gemma_model.vision_tower is not None:
        gemma_model.vision_tower.to("cpu")
    if hasattr(gemma_model, "multi_modal_projector") and gemma_model.multi_modal_projector is not None:
        gemma_model.multi_modal_projector.to("cpu")

    # Move text encoder's non-Gemma components back to CPU
    if hasattr(text_encoder, "feature_extractor_linear"):
        text_encoder.feature_extractor_linear.to("cpu")
    if hasattr(text_encoder, "embeddings_connector"):
        text_encoder.embeddings_connector.to("cpu")
    if hasattr(text_encoder, "audio_embeddings_connector"):
        text_encoder.audio_embeddings_connector.to("cpu")

    # Restore original forward if saved
    if hasattr(gemma_text_model, "_original_forward"):
        gemma_text_model.forward = gemma_text_model._original_forward
        del gemma_text_model._original_forward

    # Restore original _preprocess_text
    if hasattr(text_encoder, "_original_preprocess_text"):
        text_encoder._preprocess_text = text_encoder._original_preprocess_text
        del text_encoder._original_preprocess_text

    # Restore original _enhance
    if hasattr(text_encoder, "_original_enhance"):
        text_encoder._enhance = text_encoder._original_enhance
        del text_encoder._original_enhance

    # Cleanup attributes
    if hasattr(gemma_text_model, "_block_swap_offloader"):
        del gemma_text_model._block_swap_offloader
    if hasattr(gemma_text_model, "_blocks_to_swap"):
        del gemma_text_model._blocks_to_swap
    if hasattr(gemma_text_model, "_blocks_ref"):
        del gemma_text_model._blocks_ref
    if hasattr(gemma_text_model, "_block_swap_device"):
        del gemma_text_model._block_swap_device

    # Synchronize again after moves complete
    if device.type == "cuda":
        torch.cuda.synchronize()

    clean_memory_on_device(device)
    print("[TextEncoderBlockSwap] All layers offloaded to CPU")
