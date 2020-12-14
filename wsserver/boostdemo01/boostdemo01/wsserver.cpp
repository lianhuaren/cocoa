#include <cstdlib>


#include <sys/socket.h>
#include <netinet/in.h>

#include "wsserver.h"


/*###################################################################################
##  B A S E    6 4
###################################################################################*/



static char* base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";


static void base64encode(unsigned char* inbuf, int len, char* outbuf)
{
    char* out = outbuf;
    unsigned char* b = inbuf;
    while (len > 0)
    {
        if (len >= 3)
        {
            int b0 = (int)*b++;
            int b1 = (int)*b++;
            int b2 = (int)*b++;
            *out++ = base64[((b0) >> 2)];
            *out++ = base64[((b0 & 0x03) << 4) | ((b1 & 0xf0) >> 4)];
            *out++ = base64[((b1 & 0x0f) << 2) | ((b2 & 0xc0) >> 6)];
            *out++ = base64[((b2 & 0x3f))];
            len -= 3;
        }
        else if (len == 2)
        {
            int b0 = (int)*b++;
            int b1 = (int)*b++;
            *out++ = base64[((b0 >> 2))];
            *out++ = base64[((b0 & 0x03) << 4) | ((b1 & 0xf0) >> 4)];
            *out++ = base64[((b1 & 0x0f) << 2)];
            *out++ = '=';
            len -= 2;
        }
        else /* len == 1 */
        {
            int b0 = (int)*b++;
            *out++ = base64[((b0) >> 2)];
            *out++ = base64[((b0 & 0x03) << 4)];
            *out++ = '=';
            *out++ = '=';
            len -= 1;
        }
    }

    *out = '\0';
}

/*###################################################################################
##  S H A   1
###################################################################################*/

#define TR32(x) ((x) & 0xffffffffL)
#define SHA_ROTL(X,n) ((((X) << (n)) & 0xffffffffL) | (((X) >> (32-(n))) & 0xffffffffL))




static void _sha1transform(uint32_t* H, unsigned char* block)
{
    uint32_t W[80];

    int i = 0;
    for (; i < 16; i++)
    {
        uint32_t b0 = *block++;
        uint32_t b1 = *block++;
        uint32_t b2 = *block++;
        uint32_t b3 = *block++;
        W[i] = b0 << 24 | b1 << 16 | b2 << 8 | b3;
        //printf("W[%d] : %08x\n", i, W[i]);
    }


    //see 6.1.2
    for (i = 16; i < 80; i++)
        W[i] = SHA_ROTL((W[i - 3] ^ W[i - 8] ^ W[i - 14] ^ W[i - 16]), 1);

    uint32_t a = H[0];
    uint32_t b = H[1];
    uint32_t c = H[2];
    uint32_t d = H[3];
    uint32_t e = H[4];

    uint32_t T;

    for (i = 0; i < 20; i++)
    {
        //see 4.1.1 for the boolops on B,C, and D  //Ch(b,c,d))
        T = TR32(SHA_ROTL(a, 5) + ((b & c) | ((~b) & d)) + e + 0x5a827999L + W[i]);
        e = d; d = c; c = SHA_ROTL(b, 30); b = a; a = T;
        //printf("%2d %08x %08x %08x %08x %08x\n", i, a, b, c, d, e);
    }
    for (; i < 40; i++)
    {
        T = TR32(SHA_ROTL(a, 5) + (b ^ c ^ d) + e + 0x6ed9eba1L + W[i]);
        e = d; d = c; c = SHA_ROTL(b, 30); b = a; a = T;
        //printf("%2d %08x %08x %08x %08x %08x\n", i, a, b, c, d, e);
    }
    for (; i < 60; i++)
    {
        T = TR32(SHA_ROTL(a, 5) + ((b & c) ^ (b & d) ^ (c & d)) + e + 0x8f1bbcdcL + W[i]);
        e = d; d = c; c = SHA_ROTL(b, 30); b = a; a = T;
        //printf("%2d %08x %08x %08x %08x %08x\n", i, a, b, c, d, e);
    }
    for (; i < 80; i++)
    {
        T = TR32(SHA_ROTL(a, 5) + (b ^ c ^ d) + e + 0xca62c1d6L + W[i]);
        e = d; d = c; c = SHA_ROTL(b, 30); b = a; a = T;
        //printf("%2d %08x %08x %08x %08x %08x\n", i, a, b, c, d, e);
    }

    H[0] = TR32(H[0] + a);
    H[1] = TR32(H[1] + b);
    H[2] = TR32(H[2] + c);
    H[3] = TR32(H[3] + d);
    H[4] = TR32(H[4] + e);

}



/**
 * Small and simple SHA1 hash implementation for small message sizes.
 * Note that outbuf is assumed to be 20 bytes long.
 */
static void sha1hash(unsigned char* data, int len, unsigned char* outbuf)
{
    // Initialize H with the magic constants (see FIPS180 for constants)
    uint32_t H[5];
    H[0] = 0x67452301L;
    H[1] = 0xefcdab89L;
    H[2] = 0x98badcfeL;
    H[3] = 0x10325476L;
    H[4] = 0xc3d2e1f0L;

    int i;

    int bytesLeft = len;

    unsigned char* d = data;

    unsigned char block[64];

    int cont = 1;

    while (cont)
    {
        if (bytesLeft >= 64)
        {
            unsigned char* b = block;
            for (i = 0; i < 64; i++)
                *b++ = *d++;
            bytesLeft -= 64;
            _sha1transform(H, block);
        }
        else
        {
            unsigned char* b = block;
            for (i = 0; i < bytesLeft; i++)
                *b++ = *d++;
            *b++ = 0x80;
            int pad = 64 - bytesLeft - 1;
            if (pad > 0 && pad < 8) //if not enough room, finish block and start another
            {
                while (pad--)
                    *b++ = 0;
                _sha1transform(H, block);
                b = block; //reset
                pad = 64;  //reset
            }
            pad -= 8;
            while (pad--)
                *b++ = 0;
            uint64_t nrBits = 8L * len;
            *b++ = (unsigned char)((nrBits >> 56) & 0xff);
            *b++ = (unsigned char)((nrBits >> 48) & 0xff);
            *b++ = (unsigned char)((nrBits >> 40) & 0xff);
            *b++ = (unsigned char)((nrBits >> 32) & 0xff);
            *b++ = (unsigned char)((nrBits >> 24) & 0xff);
            *b++ = (unsigned char)((nrBits >> 16) & 0xff);
            *b++ = (unsigned char)((nrBits >> 8) & 0xff);
            *b++ = (unsigned char)((nrBits) & 0xff);
            _sha1transform(H, block);
            cont = 0;
        }
    }

    //copy out answer
    unsigned char* out = outbuf;
    for (i = 0; i < 5; i++)
    {
        uint32_t h = H[i];
        *out++ = (unsigned char)((h >> 24) & 0xff);
        *out++ = (unsigned char)((h >> 16) & 0xff);
        *out++ = (unsigned char)((h >> 8) & 0xff);
        *out++ = (unsigned char)((h) & 0xff);
    }

}


/**
 * Note: b64buf should be at least 26 bytes
 */
void sha1hash64(unsigned char* data, int len, char* b64buf)
{
    unsigned char hash[20];
    sha1hash(data, len, hash);
    base64encode(hash, 20, b64buf);
}
