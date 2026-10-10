# Project execution policy

Use `gpt-6-astra` with `medium` reasoning for planning and routing. Use
`gpt-6.1-sol` with `high` reasoning for both small tasks and main implementation.
Use `gpt-5.6-sol` with `high` reasoning for independent review when needed.
Pass the selected model and reasoning effort explicitly when launching an agent.
If a required model or effort is unavailable, report the limitation without
silently substituting another setting.

Reuse a coordinator that already has the required settings. Do not force
delegation for trivial tasks. When delegation is useful, use the host's generic
agent launch interface; do not assume it supports an `agent_type` parameter.

Maintain a temporary per-task launch ledger. Serialize admission and reserve each
call before dispatch, including planning and review, under the applicable usage
policy and user budget. Record requested settings, host-accepted launch settings,
and runtime metadata separately. Leave unknown usage counters as `null`; neither
launch acceptance nor a policy file proves runtime identity or token usage.

`.laneorchestrator/config.json` is a project policy snapshot for the host to read.
It is not an automatically loaded global LaneOrchestrator configuration. Native
Windows plugin management with no-follow filesystem guarantees is unsupported;
do not modify the plugin cache, global configuration, or security settings to
apply this project policy.
