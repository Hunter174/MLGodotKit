NeuralNetworkNode snapshots (experimental; issues #35, #50)
=========================================================

A snapshot is a **data dictionary**, not a file or executable. ``get_snapshot()``
returns ``{version: 1, optimizer: "adam", learning_rate: float, layers: [...]}``.
Each ordered layer has ``input_size``, ``output_size``, ``activation`` (one of
``linear``, ``relu``, ``leaky_relu``, ``sigmoid``), ``weights`` as input_size
rows of output_size finite numbers, and ``biases`` as one output_size row.
``load_snapshot(dictionary)`` validates the *entire* shape and numeric content
before replacing the destination model. It returns false without changing an
existing network for malformed input. Maximum: 128 layers, 2048 dimensions
per axis and 1,000,000 parameters. Repeated loads replace, not append layers.
The exporter returns an empty dictionary if its current model exceeds these
limits or has unsupported activations/nonfinite weights. Godot JSON parses
integral metadata as floats; the loader accepts only exactly integral values
for ``version`` and layer dimensions. A failed load never replaces the old
model.
Optimizer moments are deliberately **not** included: reload is for inference,
not bit-for-bit continuation of training. Changes to this schema require a
new version. A future generalized model serialization contract can supersede
this local-only first format.

For an external JSON artifact, the **game** may call ``JSON.stringify`` on the
snapshot and ``FileAccess`` to save or load it, then use ``load_snapshot``.
Game exports must include any required artifact as an ordinary external file
or game asset. Godot JSON numbers round-trip through double precision; compare
predictions with floating-point tolerance, not byte identity. A SHA-256 digest
of the JSON file detects accidental corruption only; a checksum without a
trusted signature does not authenticate adversarial content. Wrap the JSON
text and its ``String.sha256_text()`` digest in a game-owned envelope and verify
*before* calling ``load_snapshot`` if detecting changed bytes matters. Limit
file size before JSON parsing; this native API cannot bound the Godot parser's
input allocation. Never trust a player-supplied snapshot. Inference weight loading is separate from the
per-agent personality/epiphany ledger; no SaLMon code belongs here.
