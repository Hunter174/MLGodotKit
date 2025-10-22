extends Node2D
class_name RLEnvironment

## ----------------------------------------------------------
## Signals
## ----------------------------------------------------------

# Emitted whenever a new episode starts
signal episode_reset(initial_state: Array)

# Emitted after each environment step
signal step_completed(state: Array, reward: float, done: bool)

# Emitted when the episode ends
signal episode_done(total_reward: float)

## ----------------------------------------------------------
## Variables
## ----------------------------------------------------------

var state: Array = []   # current state representation
var step_count: int = 0 # step counter for the episode
var total_reward: float = 0.0
var max_steps: int = 100  # override in subclasses if needed

## ----------------------------------------------------------
## Public Methods (to be overridden)
## ----------------------------------------------------------

func reset() -> Array:
	"""
	Resets the environment to an initial state and returns that state.
	This should be overridden by child environments (e.g. DotEnvironment).
	"""
	state = []
	step_count = 0
	total_reward = 0.0

	push_warning("RLEnvironment.reset() not implemented — override in subclass.")
	return state


func step(action: Array) -> Dictionary:
	"""
	Advances the environment one step given the agent's action.
	Should return a dictionary with:
		{
			"state": Array,  # next observation
			"reward": float,  # scalar reward
			"done": bool      # whether episode is over
		}
	"""
	push_warning("RLEnvironment.step() not implemented — override in subclass.")
	return {"state": [], "reward": 0.0, "done": true}


func render() -> void:
	"""
	Optional visualization or debug function.
	Can be used to draw the environment or agent position.
	"""
	pass
	
