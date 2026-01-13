from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional
import torch
import torch.nn as nn


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_block_param_shapes(block: nn.Module) -> list[tuple[str, str, torch.Size, torch.dtype]]:
    """Extract parameter shapes from a block for buffer pre-allocation."""
    shapes = []
    for module_name, module in block.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None:
                shapes.append((module_name, param_name, param.shape, param.dtype))
    return shapes


def create_pinned_buffer_for_block(shapes: list[tuple[str, str, torch.Size, torch.dtype]]) -> dict[tuple[str, str], torch.Tensor]:
    """Pre-allocate pinned CPU buffers matching a block's parameter structure."""
    buffers = {}
    for module_name, param_name, shape, dtype in shapes:
        key = (module_name, param_name)
        buffers[key] = torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
    return buffers


def swap_weight_devices_cuda_fast(
    device: torch.device,
    layer_to_cpu: nn.Module,
    layer_to_cuda: nn.Module,
    spare_buffer: dict[tuple[str, str], torch.Tensor],
    stream_to_gpu: torch.cuda.Stream,
    stream_to_cpu: torch.cuda.Stream,
    event_gpu_done: torch.cuda.Event,
    event_cpu_done: torch.cuda.Event,
):
    """
    Fast weight swap using pre-allocated buffers and dual streams.

    Args:
        layer_to_cpu: Layer currently on GPU, should end up on CPU
        layer_to_cuda: Layer currently on CPU (pinned), should end up on GPU

    Uses separate streams for GPU→CPU and CPU→GPU transfers to achieve
    full PCIe duplex utilization. Event-based sync ensures correctness
    without blocking the entire stream.
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    # Collect all parameter swap jobs
    # layer_to_cpu is currently ON GPU (will move to CPU)
    # layer_to_cuda is currently ON CPU/pinned (will move to GPU)
    swap_jobs = []  # (mod_going_to_cpu, mod_going_to_gpu, param_name, data_on_gpu, data_on_cpu_pinned, spare_buf, module_name)

    modules_going_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_name, mod_going_to_gpu in layer_to_cuda.named_modules():
        mod_going_to_cpu = modules_going_to_cpu.get(module_name, None)
        if mod_going_to_cpu is None:
            continue

        for param_name, param_on_cpu in mod_going_to_gpu.named_parameters(recurse=False):
            if param_on_cpu is None:
                continue
            param_on_gpu = getattr(mod_going_to_cpu, param_name, None)
            if param_on_gpu is None:
                continue

            key = (module_name, param_name)
            spare_buf = spare_buffer.get(key)
            if spare_buf is None or spare_buf.shape != param_on_cpu.shape:
                # Fallback: allocate if shape mismatch (shouldn't happen)
                spare_buf = torch.empty_like(param_on_cpu, device='cpu', pin_memory=True)
                spare_buffer[key] = spare_buf

            swap_jobs.append((
                mod_going_to_cpu, mod_going_to_gpu, param_name,
                param_on_gpu.data,  # Currently on GPU
                param_on_cpu.data,  # Currently on CPU (pinned)
                spare_buf,
                module_name,
            ))

    # Wait for any previous operations on default stream
    torch.cuda.current_stream(device).synchronize()

    # Temporarily exit inference mode to allow inplace tensor operations
    with torch.inference_mode(False):
        # Phase 1: Start GPU→CPU transfers (into spare buffers)
        # Copy from GPU tensors to spare pinned buffers
        with torch.cuda.stream(stream_to_cpu):
            for mod_going_to_cpu, mod_going_to_gpu, param_name, data_on_gpu, data_on_cpu_pinned, spare_buf, module_name in swap_jobs:
                data_on_gpu.record_stream(stream_to_cpu)
                spare_buf.copy_(data_on_gpu, non_blocking=True)

        # Record when GPU→CPU copies are queued
        event_cpu_done.record(stream_to_cpu)

        # Phase 2: Start CPU→GPU transfers (reusing GPU memory buffers)
        # Copy from pinned CPU tensors to GPU, reusing the existing GPU buffer
        with torch.cuda.stream(stream_to_gpu):
            stream_to_gpu.wait_event(event_gpu_done)  # Wait for previous CPU→GPU from last swap
            for mod_going_to_cpu, mod_going_to_gpu, param_name, data_on_gpu, data_on_cpu_pinned, spare_buf, module_name in swap_jobs:
                # Reuse the GPU buffer, copy pinned CPU data into it
                data_on_gpu.copy_(data_on_cpu_pinned, non_blocking=True)
                # Point the module going to GPU to this buffer
                getattr(mod_going_to_gpu, param_name).data = data_on_gpu

        # Record when CPU→GPU copies are queued
        event_gpu_done.record(stream_to_gpu)

        # Phase 3: Wait for GPU→CPU to complete, then swap buffer ownership
        stream_to_cpu.synchronize()

        # Assign spare buffers (now containing old GPU data) to the modules going to CPU
        for mod_going_to_cpu, mod_going_to_gpu, param_name, data_on_gpu, data_on_cpu_pinned, spare_buf, module_name in swap_jobs:
            getattr(mod_going_to_cpu, param_name).data = spare_buf

        # Update spare_buffer dict: old pinned CPU buffers become new spare buffers
        for mod_going_to_cpu, mod_going_to_gpu, param_name, data_on_gpu, data_on_cpu_pinned, spare_buf, module_name in swap_jobs:
            key = (module_name, param_name)
            spare_buffer[key] = data_on_cpu_pinned  # Old pinned buffer is now spare

        # Ensure CPU→GPU is done before returning (needed for correct execution)
        stream_to_gpu.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    Legacy swap function - allocates buffers each time.
    Use swap_weight_devices_cuda_fast with pre-allocated buffers for better performance.
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    other_param_jobs = []  # For biases and other non-weight parameters

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
        if module_to_cpu is None:
            continue

        # Handle weight parameter with buffer reuse
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            if module_to_cpu.weight is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            elif module_to_cuda.weight.data.device.type != device.type:
                module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        # Handle all other parameters (bias, etc.) - collect them for simple transfer
        for param_name, param in module_to_cuda.named_parameters(recurse=False):
            if param_name == "weight":  # Already handled above
                continue
            if param is not None:
                cpu_param = getattr(module_to_cpu, param_name, None)
                if cpu_param is not None:
                    other_param_jobs.append((module_to_cpu, module_to_cuda, param_name, cpu_param.data, param.data))

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu - weights (allocate new pinned buffer to preserve fast transfer capability)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            # Allocate a new pinned buffer for the outgoing weights (don't overwrite cpu_data_view
            # which contains the weights we need to load to GPU next)
            if cpu_data_view.is_pinned():
                pinned_buf = torch.empty_like(cuda_data_view, device='cpu', pin_memory=True)
                pinned_buf.copy_(cuda_data_view, non_blocking=True)
                module_to_cpu.weight.data = pinned_buf
            else:
                module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        # cuda to cpu - other params (biases etc. are small, simple transfer is fine)
        for module_to_cpu, module_to_cuda, param_name, cpu_param_data, cuda_param_data in other_param_jobs:
            # If the GPU module's param is on GPU, move to CPU
            if cpu_param_data.device.type == device.type:
                setattr(module_to_cpu, param_name + "_data_backup", cpu_param_data)  # temporary backup
                getattr(module_to_cpu, param_name).data = cpu_param_data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda - weights (reuse GPU buffer, transfer from pinned is faster)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

        # cpu to cuda - other params (simple transfer, reuse buffer if available)
        for module_to_cpu, module_to_cuda, param_name, cpu_param_data, cuda_param_data in other_param_jobs:
            backup_key = param_name + "_data_backup"
            if hasattr(module_to_cpu, backup_key):
                # Reuse the GPU buffer from the module that moved to CPU
                gpu_buffer = getattr(module_to_cpu, backup_key)
                gpu_buffer.copy_(cuda_param_data, non_blocking=True)
                getattr(module_to_cuda, param_name).data = gpu_buffer
                delattr(module_to_cpu, backup_key)
            else:
                # Fallback: simple transfer to GPU
                getattr(module_to_cuda, param_name).data = cuda_param_data.to(device, non_blocking=True)

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device()

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device()


def weighs_to_device(layer: nn.Module, device: torch.device):
    """Move all parameters (weights, biases, and any other parameters) to the specified device."""
    for module in layer.modules():
        # Move all named parameters, not just weights
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param is not None:
                param.data = param.data.to(device, non_blocking=True)


def weights_to_pinned_cpu(layer: nn.Module):
    """Move all parameters to pinned CPU memory for faster GPU transfers.

    Pinned (page-locked) memory enables DMA transfers which are 2-3x faster
    than regular paged memory transfers.
    """
    for module in layer.modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param is not None and param.data.device.type != 'cuda':
                # Allocate pinned memory and copy
                pinned = torch.empty_like(param.data, device='cpu', pin_memory=True)
                pinned.copy_(param.data)
                param.data = pinned


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False, use_pinned_weights: bool = True):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug
        self.use_pinned_weights = use_pinned_weights

        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

        # Fast path: pre-allocated resources (initialized in prepare_block_devices_before_forward)
        self.use_fast_swap = False
        self.spare_buffers = {}  # worker_id -> buffer dict
        self.streams_to_gpu = {}  # worker_id -> stream
        self.streams_to_cpu = {}  # worker_id -> stream
        self.events_gpu_done = {}  # worker_id -> event
        self.events_cpu_done = {}  # worker_id -> event
        self._worker_id_counter = 0
        self._worker_id_lock = None

    def _get_worker_id(self) -> int:
        """Get a unique worker ID for the current thread."""
        import threading
        if self._worker_id_lock is None:
            self._worker_id_lock = threading.Lock()

        thread_id = threading.current_thread().ident
        # Simple round-robin assignment
        with self._worker_id_lock:
            worker_id = self._worker_id_counter % 4
            self._worker_id_counter += 1
        return worker_id

    def _init_fast_swap_resources(self, reference_block: nn.Module):
        """Initialize pre-allocated buffers and streams for fast swapping."""
        if not self.cuda_available:
            return

        shapes = get_block_param_shapes(reference_block)

        # Create resources for each worker
        for worker_id in range(4):
            self.spare_buffers[worker_id] = create_pinned_buffer_for_block(shapes)
            self.streams_to_gpu[worker_id] = torch.cuda.Stream(device=self.device)
            self.streams_to_cpu[worker_id] = torch.cuda.Stream(device=self.device)
            self.events_gpu_done[worker_id] = torch.cuda.Event()
            self.events_cpu_done[worker_id] = torch.cuda.Event()
            # Record initial event so first wait doesn't block forever
            self.events_gpu_done[worker_id].record()

        self.use_fast_swap = True

        # Calculate memory usage
        total_spare_mem = sum(buf.numel() * buf.element_size() for buf in self.spare_buffers[0].values())
        print(f"[{self.block_type}] Fast swap initialized: {total_spare_mem / 1e6:.1f}MB spare buffers x 4 workers = {total_spare_mem * 4 / 1e6:.1f}MB total")

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module, worker_id: int = 0):
        if self.cuda_available:
            if self.use_fast_swap and worker_id in self.spare_buffers:
                swap_weight_devices_cuda_fast(
                    self.device,
                    block_to_cpu,
                    block_to_cuda,
                    self.spare_buffers[worker_id],
                    self.streams_to_gpu[worker_id],
                    self.streams_to_cpu[worker_id],
                    self.events_gpu_done[worker_id],
                    self.events_cpu_done[worker_id],
                )
            else:
                swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        worker_id = self._get_worker_id()

        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda, wid):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'} (worker {wid})"
                )

            self.swap_weight_devices(block_to_cpu, block_to_cuda, wid)

            if self.debug:
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda, worker_id
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        # For fast swap, sync is already done in the swap function
        # For legacy swap, we still need a sync
        if self.cuda_available and not self.use_fast_swap:
            torch.cuda.synchronize()
            if self.debug:
                print(f"[{self.block_type}] Swap complete for block {block_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        debug: bool = False,
        use_pinned_weights: bool = True,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug, use_pinned_weights)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        num_resident = self.num_blocks - self.blocks_to_swap
        if self.debug:
            print(f"[{self.block_type}] Prepare block devices: {num_resident} blocks on GPU, {self.blocks_to_swap} blocks on CPU (pinned={self.use_pinned_weights})")

        # Move only the first (num_blocks - blocks_to_swap) blocks to GPU
        # These are the blocks that will be on GPU initially
        for i, b in enumerate(blocks[0 : num_resident]):
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure all params are on device
            if self.debug and self.device.type == "cuda":
                print(f"  Block {i} moved to GPU. GPU memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

        # Keep the remaining blocks on CPU - they will be swapped in during forward pass
        # Use pinned memory for faster transfers if enabled
        for i, b in enumerate(blocks[num_resident:]):
            # Ensure all parameters are on CPU first
            weighs_to_device(b, "cpu")
            # Then convert to pinned memory if enabled
            if self.use_pinned_weights:
                weights_to_pinned_cpu(b)

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

        # Initialize fast swap resources using first block as reference
        if len(blocks) > 0 and self.cuda_available:
            self._init_fast_swap_resources(blocks[0])

        if self.debug and self.device.type == "cuda":
            print(f"[{self.block_type}] After prepare: GPU memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if supports_backward and backward is enabled, we swap blocks more than blocks_to_swap in backward pass
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
