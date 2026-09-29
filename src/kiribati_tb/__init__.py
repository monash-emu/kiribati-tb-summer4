"""Kiribati TB screening model on summer4.

summer2gen runs in float64 (computegraph enables ``jax_enable_x64``); summer4 does not turn it
on, so this package does, before anything builds a JAX array.
"""

import jax

jax.config.update("jax_enable_x64", True)
