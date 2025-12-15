from typing import Any

import torch
import torch.nn as nn


def register_shape_hooks(model: nn.Module) -> tuple[list[tuple[str, torch.Size]], dict[str, Any]]:
    """
    Register hooks on all convolutional and linear layers of a PyTorch model to extract output shapes during a forward pass.
    """
    shapes_list = []
    hooks = {}

    def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor, name: str) -> None:
        # Store the module name and the output shape
        shapes_list.append((name, output.shape))

    # Recursively register hooks for all Conv and Linear layers
    def register_hooks_recursive(module: nn.Module, name_prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # Register hook for Conv2d, Conv1d, Conv3d, and Linear layers
            if isinstance(child, nn.Conv2d | nn.Linear):
                hooks[full_name] = child.register_forward_hook(
                    lambda module, input, output, full_name=full_name: hook_fn(module, input, output, full_name)
                )

            # Recursively register hooks for children
            register_hooks_recursive(child, full_name)

    # Start registering hooks from the top-level module
    register_hooks_recursive(model)

    return shapes_list, hooks


def remove_hooks(hooks: dict[str, Any]) -> None:
    """Remove all registered hooks."""
    for hook in hooks.values():
        hook.remove()


def get_out_shapes(model: nn.Module, dummy_input: torch.Tensor) -> list[torch.Size]:
    """Get"""
    # Register hooks
    shapes_dict, hooks = register_shape_hooks(model)

    # Run a forward pass
    _ = model(dummy_input)

    # Clean up hooks when done
    remove_hooks(hooks)

    shapes_list = [shape[1:] for _, shape in shapes_dict]

    return shapes_list
