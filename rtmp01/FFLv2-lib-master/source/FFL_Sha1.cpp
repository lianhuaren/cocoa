/*
 * SHA-1 in C
 * By Steve Reid <sreid@sea-to-sky.net>
 * 100% Public Domain
 *
 * -----------------
 * Modified 7/98
 * By James H. Brown <jbrown@burgoyne.com>
 * Still 100% Public Domain
 *
 * Corrected a problem which generated improper hash values on 16 bit machines
 * Routine SHA1Update changed from
 *   void SHA1Update(SHA1_CTX* context, unsigned char* data, unsigned int
 * len)
 * to
 *   void SHA1Update(SHA1_CTX* context, unsigned char* data, unsigned
 * long len)
 *
 * The 'len' parameter was declared an int which works fine on 32 bit machines.
 * However, on 16 bit machines an int is too small for the shifts being done
 * against
 * it.  This caused the hash function to generate incorrect values if len was
 * greater than 8191 (8K - 1) due to the 'len << 3' on line 3 of SHA1Update().
 *
 * Since the file IO in main() reads 16K at a time, any file 8K or larger would
 * be guaranteed to generate the wrong hash (e.g. Test Vector #3, a million
 * "a"s).
 *
 * I also changed the declaration of variables i & j in SHA1Update to
 * unsigned long from unsigned int for the same reason.
 *
 * These changes should make no difference to any 32 bit implementations since
 * an
 * int and a long are the same size in those environments.
 *
 * --
 * I also corrected a few compiler warnings generated by Borland C.
 * 1. Added #include <process.h> for exit() prototype
 * 2. Removed unused variable 'j' in SHA1Final
 * 3. Changed exit(0) to return(0) at end of main.
 *
 * ALL changes I made can be located by searching for comments containing 'JHB'
 * -----------------
 * Modified 8/98
 * By Steve Reid <sreid@sea-to-sky.net>
 * Still 100% public domain
 *
 * 1- Removed #include <process.h> and used return() instead of exit()
 * 2- Fixed overwriting of finalcount in SHA1Final() (discovered by Chris Hall)
 * 3- Changed email address from steve@edmweb.com to sreid@sea-to-sky.net
 *
 * -----------------
 * Modified 4/01
 * By Saul Kravitz <Saul.Kravitz@celera.com>
 * Still 100% PD
 * Modified to run on Compaq Alpha hardware.
 *
 * -----------------
 * Modified 07/2002
 * By Ralph Giles <giles@ghostscript.com>
 * Still 100% public domain
 * modified for use with stdint types, autoconf
 * code cleanup, removed attribution comments
 * switched SHA1Final() argument order for consistency
 * use SHA1_ prefix for public api
 * move public api to sha1.h
 *
 * -----------------
 * Modified 02/2012
 * By Justin Uberti <juberti@google.com>
 * Remove underscore from SHA1 prefix to avoid conflict with OpenSSL
 * Remove test code
 * Untabify
 *
 * -----------------
 * Modified 03/2012
 * By Ronghua Wu <ronghuawu@google.com>
 * Change the typedef of uint32(8)_t to uint32(8). We need this because in the
 * chromium android build, the stdio.h will include stdint.h which already
 * defined uint32(8)_t.
 *
 * -----------------
 * Modified 04/2012
 * By Frank Barchard <fbarchard@google.com>
 * Ported to C++, Google style, change len to size_t, enable SHA1HANDSOFF
 *
 * Test Vectors (from FIPS PUB 180-1)
 * "abc"
 *   A9993E36 4706816A BA3E2571 7850C26C 9CD0D89D
 * "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
 *   84983E44 1C3BD26E BAAE4AA1 F95129E5 E54670F1
 * A million repetitions of "a"
 *   34AA973C D4C4DAA4 F61EEB2B DBAD2731 6534016F
 *
 * -----------------
 * Modified 05/2015
 * By Sergey Ulanov <sergeyu@chromium.org>
 * Removed static buffer to make computation thread-safe.
 *
 * -----------------
 * Modified 10/2015
 * By Peter Boström <pbos@webrtc.org>
 * Change uint32(8) back to uint32(8)_t (undoes (03/2012) change).
 */

// Enabling SHA1HANDSOFF preserves the caller's data buffer.
// Disabling SHA1HANDSOFF the buffer will be modified (end swapped).
#define SHA1HANDSOFF

#include "FFL_Sha1.hpp"

#include <stdio.h>
#include <string.h>

namespace FFL {

namespace {

#define rol(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

// blk0() and blk() perform the initial expand.
// I got the idea of expanding during the round function from SSLeay
// FIXME: can we do this in an endian-proof way?
#ifdef RTC_ARCH_CPU_BIG_ENDIAN
#define blk0(i) block->l[i]
#else
#define blk0(i) (block->l[i] = (rol(block->l[i], 24) & 0xFF00FF00) | \
    (rol(block->l[i], 8) & 0x00FF00FF))
#endif
#define blk(i) (block->l[i & 15] = rol(block->l[(i + 13) & 15] ^ \
    block->l[(i + 8) & 15] ^ block->l[(i + 2) & 15] ^ block->l[i & 15], 1))

// (R0+R1), R2, R3, R4 are the different operations used in SHA1.
#define R0(v, w, x, y, z, i) \
    z += ((w & (x ^ y)) ^ y) + blk0(i) + 0x5A827999 + rol(v, 5); \
    w = rol(w, 30);
#define R1(v, w, x, y, z, i) \
    z += ((w & (x ^ y)) ^ y) + blk(i) + 0x5A827999 + rol(v, 5); \
    w = rol(w, 30);
#define R2(v, w, x, y, z, i) \
    z += (w ^ x ^ y) + blk(i) + 0x6ED9EBA1 + rol(v, 5);\
    w = rol(w, 30);
#define R3(v, w, x, y, z, i) \
    z += (((w | x) & y) | (w & x)) + blk(i) + 0x8F1BBCDC + rol(v, 5); \
    w = rol(w, 30);
#define R4(v, w, x, y, z, i) \
    z += (w ^ x ^ y) + blk(i) + 0xCA62C1D6 + rol(v, 5); \
    w = rol(w, 30);

#ifdef VERBOSE  // SAK
void SHAPrintContext(SHA1_CTX *context, char *msg) {
  printf("%s (%d,%d) %x %x %x %x %x\n",
         msg,
         context->count[0], context->count[1],
         context->state[0],
         context->state[1],
         context->state[2],
         context->state[3],
         context->state[4]);
}
#endif /* VERBOSE */

// Hash a single 512-bit block. This is the core of the algorithm.
void SHA1Transform(uint32_t state[5], const uint8_t buffer[64]) {
  union CHAR64LONG16 {
    uint8_t c[64];
    uint32_t l[16];
  };
#ifdef SHA1HANDSOFF
  uint8_t workspace[64];
  memcpy(workspace, buffer, 64);
  CHAR64LONG16* block = reinterpret_cast<CHAR64LONG16*>(workspace);
#else
  // Note(fbarchard): This option does modify the user's data buffer.
  CHAR64LONG16* block = const_cast<CHAR64LONG16*>(
      reinterpret_cast<const CHAR64LONG16*>(buffer));
#endif

  // Copy context->state[] to working vars.
  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];

  // 4 rounds of 20 operations each. Loop unrolled.
  // Note(fbarchard): The following has lint warnings for multiple ; on
  // a line and no space after , but is left as-is to be similar to the
  // original code.
  R0(a,b,c,d,e,0); R0(e,a,b,c,d,1); R0(d,e,a,b,c,2); R0(c,d,e,a,b,3);
  R0(b,c,d,e,a,4); R0(a,b,c,d,e,5); R0(e,a,b,c,d,6); R0(d,e,a,b,c,7);
  R0(c,d,e,a,b,8); R0(b,c,d,e,a,9); R0(a,b,c,d,e,10); R0(e,a,b,c,d,11);
  R0(d,e,a,b,c,12); R0(c,d,e,a,b,13); R0(b,c,d,e,a,14); R0(a,b,c,d,e,15);
  R1(e,a,b,c,d,16); R1(d,e,a,b,c,17); R1(c,d,e,a,b,18); R1(b,c,d,e,a,19);
  R2(a,b,c,d,e,20); R2(e,a,b,c,d,21); R2(d,e,a,b,c,22); R2(c,d,e,a,b,23);
  R2(b,c,d,e,a,24); R2(a,b,c,d,e,25); R2(e,a,b,c,d,26); R2(d,e,a,b,c,27);
  R2(c,d,e,a,b,28); R2(b,c,d,e,a,29); R2(a,b,c,d,e,30); R2(e,a,b,c,d,31);
  R2(d,e,a,b,c,32); R2(c,d,e,a,b,33); R2(b,c,d,e,a,34); R2(a,b,c,d,e,35);
  R2(e,a,b,c,d,36); R2(d,e,a,b,c,37); R2(c,d,e,a,b,38); R2(b,c,d,e,a,39);
  R3(a,b,c,d,e,40); R3(e,a,b,c,d,41); R3(d,e,a,b,c,42); R3(c,d,e,a,b,43);
  R3(b,c,d,e,a,44); R3(a,b,c,d,e,45); R3(e,a,b,c,d,46); R3(d,e,a,b,c,47);
  R3(c,d,e,a,b,48); R3(b,c,d,e,a,49); R3(a,b,c,d,e,50); R3(e,a,b,c,d,51);
  R3(d,e,a,b,c,52); R3(c,d,e,a,b,53); R3(b,c,d,e,a,54); R3(a,b,c,d,e,55);
  R3(e,a,b,c,d,56); R3(d,e,a,b,c,57); R3(c,d,e,a,b,58); R3(b,c,d,e,a,59);
  R4(a,b,c,d,e,60); R4(e,a,b,c,d,61); R4(d,e,a,b,c,62); R4(c,d,e,a,b,63);
  R4(b,c,d,e,a,64); R4(a,b,c,d,e,65); R4(e,a,b,c,d,66); R4(d,e,a,b,c,67);
  R4(c,d,e,a,b,68); R4(b,c,d,e,a,69); R4(a,b,c,d,e,70); R4(e,a,b,c,d,71);
  R4(d,e,a,b,c,72); R4(c,d,e,a,b,73); R4(b,c,d,e,a,74); R4(a,b,c,d,e,75);
  R4(e,a,b,c,d,76); R4(d,e,a,b,c,77); R4(c,d,e,a,b,78); R4(b,c,d,e,a,79);

  // Add the working vars back into context.state[].
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
}

}  // namespace

// SHA1Init - Initialize new context.
void SHA1Init(SHA1_CTX* context) {
  // SHA1 initialization constants.
  context->state[0] = 0x67452301;
  context->state[1] = 0xEFCDAB89;
  context->state[2] = 0x98BADCFE;
  context->state[3] = 0x10325476;
  context->state[4] = 0xC3D2E1F0;
  context->count[0] = context->count[1] = 0;
}

// Run your data through this.
void SHA1Update(SHA1_CTX* context, const uint8_t* data, size_t input_len) {
  size_t i = 0;

#ifdef VERBOSE
  SHAPrintContext(context, "before");
#endif

  // Compute number of bytes mod 64.
  size_t index = (context->count[0] >> 3) & 63;

  // Update number of bits.
  // TODO: Use uint64_t instead of 2 uint32_t for count.
  // count[0] has low 29 bits for byte count + 3 pad 0's making 32 bits for
  // bit count.
  // Add bit count to low uint32_t
  context->count[0] += static_cast<uint32_t>(input_len << 3);
  if (context->count[0] < static_cast<uint32_t>(input_len << 3)) {
    ++context->count[1];  // if overlow (carry), add one to high word
  }
  context->count[1] += static_cast<uint32_t>(input_len >> 29);
  if ((index + input_len) > 63) {
    i = 64 - index;
    memcpy(&context->buffer[index], data, i);
    SHA1Transform(context->state, context->buffer);
    for (; i + 63 < input_len; i += 64) {
      SHA1Transform(context->state, data + i);
    }
    index = 0;
  }
  memcpy(&context->buffer[index], &data[i], input_len - i);

#ifdef VERBOSE
  SHAPrintContext(context, "after ");
#endif
}

// Add padding and return the message digest.
void SHA1Final(SHA1_CTX* context, uint8_t digest[SHA1_DIGEST_SIZE]) {
  uint8_t finalcount[8];
  for (int i = 0; i < 8; ++i) {
    // Endian independent
    finalcount[i] = static_cast<uint8_t>(
        (context->count[(i >= 4 ? 0 : 1)] >> ((3 - (i & 3)) * 8)) & 255);
  }
  SHA1Update(context, reinterpret_cast<const uint8_t*>("\200"), 1);
  while ((context->count[0] & 504) != 448) {
    SHA1Update(context, reinterpret_cast<const uint8_t*>("\0"), 1);
  }
  SHA1Update(context, finalcount, 8);  // Should cause a SHA1Transform().
  for (int i = 0; i < SHA1_DIGEST_SIZE; ++i) {
    digest[i] = static_cast<uint8_t>(
        (context->state[i >> 2] >> ((3 - (i & 3)) * 8)) & 255);
  }

  // Wipe variables.
  memset(context->buffer, 0, 64);
  memset(context->state, 0, 20);
  memset(context->count, 0, 8);
  memset(finalcount, 0, 8);   // SWR

#ifdef SHA1HANDSOFF  // Make SHA1Transform overwrite its own static vars.
  SHA1Transform(context->state, context->buffer);
#endif
}

} 
