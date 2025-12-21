extends Node
class_name RolloutBuffer

var states := []
var actions := []
var rewards := []
var dones := []
var log_probs := []
var values := []

func store(s, a, r, done, log_p := 0.0, v := 0.0):
	states.append(s)
	actions.append(a)
	rewards.append(r)
	dones.append(done)
	log_probs.append(log_p)
	values.append(v)

func clear():
	states.clear()
	actions.clear()
	rewards.clear()
	dones.clear()
	log_probs.clear()
	values.clear()

func size():
	return states.size()
