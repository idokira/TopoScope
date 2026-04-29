# Phase 0 Base Design Knowledge (Macro Planning Only)

Use this knowledge only for module/interface planning. Do NOT perform schematic-level wiring in Phase 0.

## K1 · Planning abstraction
Phase 0 defines module responsibilities, exposed interfaces, and inter-module routing intent.
Do not instantiate passives, local support circuits, or pin-level schematic wiring in this phase.

## K2 · Graph validity principle
The final design is a bipartite pin-net graph. Phase 0 must plan interfaces and routing intent consistent with that structure.

## K3 · Reference data is authoritative
Do not guess IC pins, pin functions, or part identities. Use only the provided component data and brief.

## K4 · Stable naming
Use stable semantic names.
Module IDs use `lower_snake_case`.
Net and signal names use `UPPER_SNAKE_CASE`.
RefDes, when referenced, must follow `{PREFIX}{NUMBER}`.