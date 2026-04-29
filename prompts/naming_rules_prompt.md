# Naming Rules for Schematic Graph Operations

You MUST follow these rules when creating components, nets, and connections. Violations will be rejected.

---

## 1. Refdes (Component Reference Designator)

Format: `{PREFIX}{NUMBER}`

| Prefix | Type | Example |
|--------|------|---------|
| U | IC | U1, U2 |
| R | Resistor | R1, R2 |
| C | Capacitor | C1, C2 |
| L | Inductor | L1 |
| D | Diode / LED | D1 |
| Q | MOSFET / BJT | Q1 |
| Y | Crystal | Y1 |
| F | Fuse | F1 |
| J | Connector | J1 |
| SW | Switch | SW1 |

**STRICT REFDES NAMING RULE:**
You MUST use standard alphanumeric Reference Designators exactly matching the `{PREFIX}{NUMBER}` format (e.g., `R1`, `C2`, `L1`, `D1`). 
**NEVER** use descriptive suffixes, semantic names, or custom labels for the RefDes (e.g., `Rfb`, `Cin`, `C_boot`, `R_top`). The downstream parser relies on regex matching and will completely fail if you do not use strict numeric suffixes.

Numbering: sequential within the design, never reused. ICs first (power → MCU → peripherals), then passives grouped by the IC they serve.

---

## 2. Pin ID

Format: `{REFDES}.{PIN_IDENTIFIER}`

### IC pins: use pin NAME from datasheet

```
U1.VIN    U1.FB    U1.GND    U1.SW
```

When an IC has duplicate pin names, append `_N` (N = occurrence index by ascending pin number):

```
U2.VDD_1    U2.VDD_2    U2.VSS_1    U2.VSS_2
```

### Passive pins: fixed by component type

| Type | Pins | Notes |
|------|------|-------|
| Resistor | `.1`, `.2` | `.1` = signal source side |
| Capacitor | `.1`, `.2` | `.1` = positive side (electrolytic) |
| Inductor | `.1`, `.2` | `.1` = input side |
| Diode / LED | `.A`, `.K` | Anode, Kathode |
| N-MOSFET | `.G`, `.D`, `.S` | Gate, Drain, Source |
| P-MOSFET | `.G`, `.D`, `.S` | Same |
| NPN BJT | `.B`, `.C`, `.E` | Base, Collector, Emitter |
| PNP BJT | `.B`, `.C`, `.E` | Same |
| Crystal / Resonator | `.1`, `.2` | No polarity |

---

## 3. Passive Part Number

Format: `{TYPE}_{VALUE}` or `{TYPE}_{VALUE}_{EXTRA}`

| TYPE | For |
|------|-----|
| RES | Resistor |
| CAP | Capacitor |
| IND | Inductor |
| DIODE | Generic diode |
| LED | LED |
| MOSFET_N / MOSFET_P | MOSFET |
| BJT_NPN / BJT_PNP | BJT |
| CRYSTAL | Crystal oscillator |
| FUSE | Fuse |

Resonator follows Crystal naming: use `Y{N}` for refdes and `CRYSTAL_<VALUE>` for part_number.
For dynamic creation, `add_component_tool` supports R/C/L/D/LED/Crystal/Resonator/Fuse/MOSFET_N/MOSFET_P/BJT_NPN/BJT_PNP.
`add_passive_and_connect_tool` is still limited to 2-pin parts; use `add_component_tool` + pin-by-pin connect for MOSFET/BJT.

VALUE uses engineering notation:

- Resistor: `0.1R`, `100R`, `10K`, `1M`
- Capacitor: `22pF`, `100nF`, `10uF`
- Inductor: `4.7uH`, `1mH`

EXTRA is optional: voltage rating (`25V`), tolerance (`1%`), or specific model name.

Examples:
```
RES_10K            RES_0.1R_1%         CAP_100nF
CAP_10uF_25V       IND_4.7uH           CRYSTAL_8MHz
FUSE_1A
DIODE_SS34         MOSFET_N_2N7002      LED_RED
BJT_NPN_2N3904
```

For discrete devices with a specific model (Schottky diodes, MOSFETs), use the model name as VALUE: `DIODE_SS34`, `MOSFET_N_2N7002`.

---

## 4. Net Naming

**Hard constraints** (enforced by validation):
- UPPER_SNAKE_CASE only. Regex: `^[A-Z][A-Z0-9_]*$`
- No dots, no lowercase, no spaces, no leading digits
- Globally unique within the design

**Decision tree — when you create a net, follow this:**

### Step 1: Is it a power or ground net?

YES → Use the voltage value or ground function as the name.

```
GND                  -- unified ground
AGND / DGND          -- when analog/digital ground separation is required
VCC_3V3              -- 3.3V rail
VCC_5V               -- 5V rail
VCC_12V              -- 12V rail
VBAT                 -- battery voltage
VIN                  -- input voltage (before regulation)
VCC_3V3A             -- analog 3.3V (separate from digital)
```

Rules:
- Use actual voltage values, not abstract names like VDD
- Append `A` for analog rails, `D` for digital rails (only when separation exists)
- Only external input rails and system grounds are globally available by default.
- Examples: `VIN_5V`, `VIN_12V`, `VBAT`, `GND`, `AGND`, `DGND`.
- Internally generated rails such as `VCC_3V3`, `VCC_1V8`, and `VCC_5V` use the same naming style,
  but in Phase 0 they must still be routed explicitly from their provider module to consumer modules.

### Step 2: Is it a signal shared between modules?

YES → Name by protocol/function + signal name.

```
I2C_SDA              I2C_SCL
SPI_MOSI             SPI_MISO             SPI_CLK
SPI_CS_U3            -- chip select: SPI_CS_{target_refdes}
UART_TX              UART_RX
USB_DP               USB_DN
CAN_H                CAN_L
RS485_A              RS485_B
MOTOR_PWM_A          POWER_GOOD           EN_VCC_1V8
```

Rules:
- Bus signals: `{PROTOCOL}_{SIGNAL}`
- Multiple buses of the same type: `SPI1_MOSI`, `SPI2_MOSI`
- Chip select: `SPI_CS_{REFDES}` to identify target device
- Control signals: `{FUNCTION}` descriptive name

### Step 3: Is it connected to a specific IC functional pin?

YES → Prefix with the IC's refdes.

```
U1_SW                -- switch node
U1_BST               -- bootstrap
U1_FB                -- feedback
U1_COMP              -- compensation
U1_SS                -- soft start
U1_VCC               -- IC's local VCC (decoupling cap network)
U2_VREF              -- reference voltage output
```

Format: `{REFDES}_{PIN_FUNCTION}` using the datasheet's standard abbreviation.

### Step 4: Is it a mid-point between passive components?

YES → Name by the function of that node.

```
U1_FB_DIV            -- feedback divider midpoint (between R_top and R_bottom)
U1_COMP_RC           -- compensation RC midpoint
VSEN_DIV             -- voltage sense divider midpoint
```

Format: `{REFDES}_{FUNCTION}_DIV` for divider midpoints, `{REFDES}_{FUNCTION}_RC` for RC nodes.

### Step 5: Crystal oscillator nets?

YES → Use the MCU's refdes as prefix.

```
U2_OSC_IN            -- crystal input side (MCU OSC_IN pin)
U2_OSC_OUT           -- crystal output side (MCU OSC_OUT pin)
```

Each crystal side gets its own net (load capacitors connect from each side to ground).

---

## 5. Module Naming

Format: `lower_snake_case`

```
power_5v             power_3v3_ldo        mcu
imu_sensor           flash_storage        can_transceiver
motor_driver_a       audio_preamp
```

Name reflects function, not specific IC.

---

## 6. VirtualPort Runtime Node ID

The system derives the runtime VirtualPort node ID as `{module_name}_{signal_name}`.

In prompt outputs such as Phase 0 JSON:
- `module_id` stores the module name
- `virtual_ports[].port_name` stores only the signal name

Example:
- module_id = `mcu`
- port_name = `I2C_SDA`
- runtime VirtualPort node ID = `mcu_I2C_SDA`

---

## Common Mistakes to Avoid

| WRONG | WHY | CORRECT |
|-------|-----|---------|
| `NET1`, `NET2` | No semantic meaning | Use functional name |
| `U1_PIN3_TO_R1_PIN1` | Describes wiring, not function | `U1_FB` |
| `vcc_3v3` | Must be UPPER_SNAKE_CASE | `VCC_3V3` |
| `3V3` | Starts with digit | `VCC_3V3` |
| `VCC3.3` | Contains dot | `VCC_3V3` |
| `GND1`, `GND2` | Meaningless numbering | `AGND`, `DGND` |
| `Power` | Too vague | `VCC_5V` |
