#pragma once

// PyTorch CPU_ACC_TYPE(float,double): a double accumulator, F32 prefix stores.
// Metal has no native FP64. All admitted decay inputs are nonpositive, so an
// unsigned IEEE-754 binary64 magnitude accumulator exactly implements this
// addition and its round-to-nearest-even prefix conversion using integer math.
static ulong gdn_shift_jam(ulong x, uint shift) {
  if (shift == 0) return x;
  if (shift >= 64) return x != 0 ? 1ul : 0ul;
  return (x >> shift) | ((x & ((1ul << shift) - 1ul)) != 0 ? 1ul : 0ul);
}

static ulong gdn_f64_accumulate(ulong accumulator, float negative_input) {
  const uint bits = as_type<uint>(negative_input) & 0x7fffffffu;
  if (bits == 0) return accumulator;
  if ((accumulator >> 52) == 2047) return accumulator;
  const uint exponent = bits >> 23, fraction = bits & 0x7fffffu;
  if (exponent == 255) return fraction == 0 ? 0x7ff0000000000000ul : 0x7ff8000000000000ul;
  ulong value;
  if (exponent == 0) {
    const uint leading = 31u - clz(fraction);
    value = (ulong(int(leading) - 149 + 1023) << 52) | (ulong(fraction - (1u << leading)) << (52u - leading));
  } else {
    value = (ulong(int(exponent) - 127 + 1023) << 52) | (ulong(fraction) << 29);
  }
  if (accumulator == 0) return value;
  ulong a = max(accumulator, value), b = min(accumulator, value);
  uint ae = uint(a >> 52), be = uint(b >> 52);
  const ulong mask = (1ul << 52) - 1ul;
  ulong total = (((1ul << 52) | (a & mask)) << 3) + gdn_shift_jam(((1ul << 52) | (b & mask)) << 3, ae - be);
  if (total >= (1ul << 56)) { total = gdn_shift_jam(total, 1); ++ae; }
  ulong mantissa = total >> 3;
  const uint remainder = uint(total & 7ul);
  if (remainder > 4 || (remainder == 4 && (mantissa & 1ul) != 0)) ++mantissa;
  if (mantissa >= (1ul << 53)) { mantissa >>= 1; ++ae; }
  return (ulong(ae) << 52) | (mantissa & mask);
}

static ulong gdn_round_shift_even(ulong x, uint shift) {
  if (shift == 0) return x;
  if (shift >= 64) return 0;
  ulong result = x >> shift;
  const ulong remainder = x & ((1ul << shift) - 1ul), midpoint = 1ul << (shift - 1);
  if (remainder > midpoint || (remainder == midpoint && (result & 1ul) != 0)) ++result;
  return result;
}

static float gdn_f64_prefix(ulong accumulator) {
  if (accumulator == 0) return 0.0f;
  const uint exponent = uint(accumulator >> 52);
  if (exponent == 2047) return as_type<float>((accumulator & ((1ul << 52) - 1ul)) == 0 ? 0xff800000u : 0xffc00000u);
  int fe = int(exponent) - 1023 + 127;
  const ulong mantissa = (1ul << 52) | (accumulator & ((1ul << 52) - 1ul));
  if (fe >= 255) return as_type<float>(0xff800000u);
  if (fe <= 0) return as_type<float>(0x80000000u | uint(gdn_round_shift_even(mantissa, uint(30 - fe))));
  ulong rounded = gdn_round_shift_even(mantissa, 29);
  if (rounded >= (1ul << 24)) { rounded >>= 1; ++fe; }
  if (fe >= 255) return as_type<float>(0xff800000u);
  return as_type<float>(0x80000000u | (uint(fe) << 23) | uint(rounded & 0x7ffffful));
}
