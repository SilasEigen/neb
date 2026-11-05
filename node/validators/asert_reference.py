#!/usr/bin/env python3
"""ASERT difficulty reference for NEB (spec/node.md section 6.2).

The integer algorithm is aserti3-2d (absolutely scheduled exponentially rising
targets, designed by Mark Lundeberg). This file is an independent
implementation written from the specification of the algorithm. It generates
the vector files that the Rust crate neb-difficulty replays.

Unlike the reference form, NEB anchors at the genesis block itself. The genesis block has no
parent, so the time base is the genesis timestamp and the drift for the block
after a parent at height ph is (t_parent - T_genesis) - spacing * ph. This is
the reference formula with (height_diff + 1) replaced by height_diff, because
the genesis block consumed no schedule slot. With blocks exactly one spacing
apart the target is constant.

All arithmetic is integer; divisions truncate toward zero (C semantics) and
right shifts of negative values are arithmetic (floor), as the reference form
requires. The core precondition is that of the reference form:
abs(time_diff - spacing * height_diff) < 2^47.

Usage:
  python3 asert_reference.py reference > ../vectors/asert_reference_runs.txt
  python3 asert_reference.py neb  > ../vectors/neb_asert_s0.txt
"""

import sys

RBITS = 16
RADIX = 1 << RBITS
MASK256 = (1 << 256) - 1

# Compact target of the reference runs, used only to feed and read them.

REFERENCE_MAX_BITS = 0x1D00FFFF


def reference_bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1D
    word = bits & 0x00FFFFFF
    assert 0x8000 <= word <= 0x7FFFFF
    if size <= 3:
        return word >> (8 * (3 - size))
    return word << (8 * (size - 3))


REFERENCE_MAX_TARGET = reference_bits_to_target(REFERENCE_MAX_BITS)


def reference_target_to_bits(target):
    assert target > 0
    if target > REFERENCE_MAX_TARGET:
        target = REFERENCE_MAX_TARGET
    size = (target.bit_length() + 7) // 8
    if size <= 3:
        compact = target << (8 * (3 - size))
    else:
        compact = target >> (8 * (size - 3))
    if compact & 0x00800000:
        compact >>= 8
        size += 1
    assert compact == (compact & 0x007FFFFF)
    assert size < 256
    return compact | (size << 24)


# NEB compact target (spec/node.md section 4.1): target = mantissa << shift

NEB_MANTISSA_MIN = 0x800000
NEB_MANTISSA_MAX = 0xFFFFFF
NEB_SHIFT_MAX = 232  # 24-bit mantissa << 232 < 2^256: every target fits in 256 bits


def neb_bits_to_target(bits):
    shift = bits >> 24
    mantissa = bits & NEB_MANTISSA_MAX
    assert shift <= NEB_SHIFT_MAX
    assert NEB_MANTISSA_MIN <= mantissa <= NEB_MANTISSA_MAX
    return mantissa << shift


def neb_target_to_bits(target):
    assert target >= NEB_MANTISSA_MIN
    topbits = target.bit_length() - 1
    shift = topbits - 23
    assert 0 <= shift <= NEB_SHIFT_MAX
    mantissa = target >> shift
    assert NEB_MANTISSA_MIN <= mantissa <= NEB_MANTISSA_MAX
    return (shift << 24) | mantissa


# The integer algorithm (transcription of CalculateASERT).

def trunc_div(a, b):
    """Integer division truncating toward zero, like C '/'."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def core_target(ref_target, spacing, half_life, time_diff, height_diff, pow_limit):
    """The reference form of the algorithm, in Python integers."""
    assert abs(time_diff - spacing * height_diff) < (1 << 47)
    drift = time_diff - spacing * (height_diff + 1)
    return core_from_drift(ref_target, half_life, drift, pow_limit)


def core_from_drift(ref_target, half_life, drift, pow_limit):
    """The exponent/shift/factor part, from an already computed drift."""
    # exponent = drift / half_life in 16-bit fixed point: target = ref_target * 2^(exponent / 2^16)
    exponent = trunc_div(drift * RADIX, half_life)
    shifts = exponent >> RBITS          # arithmetic shift (floor), like C
    frac = exponent & 0xFFFF            # low 16 bits, like (uint16_t)exponent
    # Cubic approximation of 2^(frac / 2^16) in 16-bit fixed point (relative error below 1.3e-4). With frac < 2^16 the
    # sum below is at most 18446563080438344768 < 2^64, so uint64 arithmetic does not wrap.
    factor = RADIX + ((
        195766423245049 * frac
        + 971821376 * frac * frac
        + 5127 * frac * frac * frac
        + (1 << 47)
    ) >> 48)
    next_target = ref_target * factor
    # next_target = ref_target * factor * 2^(shifts - 16); an overflow of the left shift (past 256 bits) clamps to pow_limit
    shifts -= RBITS
    if shifts <= 0:
        next_target >>= -shifts
    else:
        shifted = (next_target << shifts) & MASK256
        if (shifted >> shifts) != next_target:
            next_target = pow_limit
        else:
            next_target = shifted
    if next_target == 0:
        next_target = 1
    elif next_target > pow_limit:
        next_target = pow_limit
    return next_target


# NEB rule: genesis anchor and NEB target format (spec/node.md section 6.2).

NEB_MIN_TARGET = NEB_MANTISSA_MIN  # the smallest target the format can express


def neb_next_bits(pow_limit_bits, gen_bits, gen_time, parent_height, parent_time):
    """bits for the block at height parent_height + 1.

    parent_height is the height of the block's parent; the genesis block is
    height 0, so the first mined block has parent_height = 0.
    """
    ref_target = neb_bits_to_target(gen_bits)
    pow_limit = neb_bits_to_target(pow_limit_bits)
    # 60 s spacing and 600 s half-life, spec/node.md section 6.2
    drift = (parent_time - gen_time) - 60 * parent_height  # genesis has no parent, no +1
    target = core_from_drift(ref_target, 600, drift, pow_limit)
    # the compact form cannot express a target below 0x800000
    if target < NEB_MIN_TARGET:
        target = NEB_MIN_TARGET
    return neb_target_to_bits(target)


# Vector generation: the deterministic reference runs.

def reference_runs():
    """The deterministic reference runs."""
    runs = []

    def add(name, anchor_bits, anchor_height, anchor_time, start_height, start_time,
            iterations, height_step, time_step):
        runs.append((name, anchor_bits, anchor_height, anchor_time, start_height,
                     start_time, iterations, height_step, time_step))

    add("run1 - steady 600s blocks at POW limit target",
        REFERENCE_MAX_BITS, 1, 0, 2, 1200, 10, lambda h: 1, lambda i: 600)
    add("run2 - steady 600s blocks at arbitrary non-limit target 0x1a2b3c4d",
        0x1A2B3C4D, 1, 0, 2, 1200, 10, lambda h: 1, lambda i: 600)
    add("run3 - steady 600s blocks at minimum limit target 0x01010000",
        0x01010000, 1, 0, 2, 1200, 10, lambda h: 1, lambda i: 600)
    add("run4 - from minimum target, a series of halflife schedule jumps, doubling target at each block",
        0x01010000, 1, 0, 2, 600 + 173400, 256 - 31, lambda h: 1, lambda i: 173400)
    add("run5 - from POW limit, a series of halflife block height jumps w/o time increment, halving target at each block",
        REFERENCE_MAX_BITS, 1, 0, 2, 0, 256 - 31, lambda h: 288, lambda i: 0)
    add("run9 - a sequence of 300s blocks across signed 32-bit max integer height",
        0x1802AEE8, 0x7FFFFFFF - 5, 1234567890 - 600, 0x7FFFFFFF - 4, 1234567890 + 300,
        10, lambda h: 1, lambda i: 300)
    add("run10 - a sequence of 900s blocks across signed 64-bit max integer height and signed 32-bit max integer time",
        0x1802AEE8, 0x7FFFFFFFFFFFFFFF - 5, 0x7FFFFFFF - 600, 0x7FFFFFFFFFFFFFFF - 4,
        0x7FFFFFFF + 900, 10, lambda h: 1, lambda i: 900)
    add("run12 - each block 1 second before the previous one",
        0x1802AEE8, 1, 10000, 2, 11200, 10000, lambda h: 1, lambda i: -1)
    return runs


def print_reference():
    for (name, anchor_bits, anchor_height, anchor_time, start_height, start_time,
         iterations, height_step, time_step) in reference_runs():
        print("## description: %s" % name)
        print("##   anchor height: %d" % anchor_height)
        print("##   anchor parent time: %d" % anchor_time)
        print("##   anchor nBits: 0x%08x" % anchor_bits)
        print("##   start height: %d" % start_height)
        print("##   start time: %d" % start_time)
        print("##   iterations: %d" % iterations)
        print("# iteration,height,time,target")
        ref_target = reference_bits_to_target(anchor_bits)
        h, t = start_height, start_time
        for i in range(1, iterations + 1):
            target = core_target(ref_target, 600, 2 * 24 * 3600, t - anchor_time,
                                 h - anchor_height, REFERENCE_MAX_TARGET)
            print("%d %d %d 0x%08x" % (i, h, t, reference_target_to_bits(target)))
            h += height_step(h)
            t += time_step(i)
        print("")


# Vector generation: NEB scenarios.

NEB_GEN_BITS = 0xBB800000    # genesis target 2^210, a realistic order of magnitude
NEB_POW_LIMIT_BITS = 0xC2800000  # pow limit 2^217 (mantissa 2^23, shift 194): 128x easier than the genesis target
NEB_GEN_TIME = 1800000000


def print_neb():
    def run(name, times):
        print("## description: %s" % name)
        print("##   genesis bits: 0x%08x" % NEB_GEN_BITS)
        print("##   genesis time: %d" % NEB_GEN_TIME)
        print("##   pow limit bits: 0x%08x" % NEB_POW_LIMIT_BITS)
        print("##   spacing: 60, half life: 600")
        print("# iteration,height,time,target")
        t = NEB_GEN_TIME
        h = 0  # parent height of the first computed block is the genesis
        for i, dt in enumerate(times, start=1):
            bits = neb_next_bits(NEB_POW_LIMIT_BITS, NEB_GEN_BITS, NEB_GEN_TIME, h, t)
            print("%d %d %d 0x%08x" % (i, h, t, bits))
            t += dt
            h += 1
        print("")

    run("neb1 - steady 60 s blocks, the target must stay at the genesis value",
        [60] * 12)
    run("neb2 - hashrate ten times lower: 600 s blocks, the target eases",
        [600] * 20)
    run("neb3 - hashrate ten times higher: 6 s blocks, the target hardens",
        [6] * 20)
    run("neb4 - one long stall of one day, then normal blocks",
        [86400] + [60] * 5)
    run("neb5 - a single block dated far in the past, the target clamps at the floor",
        [-1000000] + [60] * 3)
    run("neb6 - times going backwards fast, the target clamps at the floor again",
        [-3600] * 12)
    run("neb7 - a deterministic mixed pattern of solve times",
        [30, 90, 45, 600, 6, 300, 61, 59, 120, 15, 240, 90, 720, 45, 30])


def main(argv):
    if len(argv) != 2 or argv[1] not in ("reference", "neb"):
        print(__doc__)
        return 2
    if argv[1] == "reference":
        print_reference()
    else:
        print_neb()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
