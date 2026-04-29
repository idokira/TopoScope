# Phase 1 Base Design Knowledge (Single-Component Wiring)

Use this knowledge for one-component wiring rounds only.

## P1 · Bipartite graph law
Pins connect to nets. Pins never connect directly to other pins.
A physical pin may appear in at most one net in the same output.

## P2 · Reference data is authoritative
Use only the provided component data, pin definitions, reference circuits, ERC rules, and environment.
Do not invent pin names, pin numbers, part identities, or generic configuration pins.

## P3 · Reference circuits are contextual
Reference circuits are recommended topologies, not blind templates.
Apply only the parts that match the current module description, virtual nets, and ERC rules.

If a reference circuit contains placeholder input/output nodes, bind them to the current environment intent.
Do not leave functional inputs floating just because the reference circuit shows them as placeholder nodes.

If a reference circuit conflicts with:
- the current module description
- virtual interface contracts
- ERC rules
- one-pin-one-net validity

then follow the explicit environment contract and electrical validity first.

## P4 · Passives and local support circuits on demand
Instantiate required local support components as needed:
- bypass capacitors
- feedback dividers
- compensation networks
- pull-ups / pull-downs
- oscillator load capacitors
- current sense / threshold / bias networks
- open-drain pull-up or wired-AND support

Do not instantiate unrelated system-level circuits that are not required by the current IC context.

## P5 · Critical pins must not float
PowerIn, Ground, enable/control-critical pins, reset/boot pins, WP/HOLD pins, and required analog inputs must be explicitly connected according to the provided component data and module intent.

Mentioned configuration behavior is valid only when the pin exists in the provided pin data.

## P6 · Interface contracts are binding
Virtual nets are external interface net names visible to this component.
When `virtual_nets_to_connect_now` lists a net, connect at least one appropriate current-IC pin or local passive pin to that net.

Never satisfy two different virtual nets by placing the same physical pin in two different net entries.
If two requested interfaces appear to be electrical aliases, use the canonical net name already provided by the environment or retry context.

## P7 · Naming rules
{naming_rules}

Use provided virtual net names exactly when connecting interface nets.
Ignore module-level and virtual-port naming rules that do not apply to this single-component JSON output.