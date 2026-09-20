# TICKET-37: Consolidate RL addon module and fix environment naming
**Roadmap Phase:** v0.3 - RL Experimental Preview
**GitHub Issue:** #37
**Status:** BLOCKED

## Objective
Provide one coherent experimental RL module with consistent class names, paths, and lifecycle contracts.

## Requirements
- [ ] Choose canonical names for environment, runner, policy, replay buffer, and DQN classes.
- [ ] Rename `enviornment.gd` to `environment.gd`.
- [ ] Remove or migrate obsolete RL scripts and tests.
- [ ] Update the cart-pole integration test to the canonical API.
- [ ] Add a headless RL smoke test.

## Verification Plan
1. Run the full GUT suite with no parse errors.
2. Run the RL integration test explicitly.
3. Start the test project headlessly and verify the extension loads.

## Agent Notes
The current full GUT run has 51 passing unit tests, but the RL integration test is blocked by missing legacy types: `RLRunner`, `DQNPolicy`, `DQNTrainer`, and `ReplayBuffer`. The existing addon currently contains newer `core` scripts instead. Do not paper over this mismatch with aliases until the canonical API is designed.
