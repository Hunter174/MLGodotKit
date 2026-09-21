# MLGodotKit examples

These scenes are small, runnable examples for learning the public API. Open a
scene in Godot 4.5.1 after building the local addon, then press **Play
Current Scene**.

The examples intentionally keep the computation in the scene script so the
connection between Godot and the native classes is visible.

## Examples

| Scene | Demonstrates |
| --- | --- |
| `matrix_basics.tscn` | Constructing matrices, multiplication, and conversion back to arrays |
| `linear_regression.tscn` | Fitting a continuous model and making predictions |
| `neural_network_xor.tscn` | Building a network, computing loss, and applying gradients |
| `pid_control.tscn` | Driving a measured value toward a setpoint with feedback control |

These examples are educational fixtures, not benchmarks. RL scenes remain
experimental and are documented separately until the RL API is stabilized.
