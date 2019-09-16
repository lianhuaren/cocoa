#ifndef _flv_muxer_h_
#define _flv_muxer_h_

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct flv_muxer_t flv_muxer_t;

///Video: FLV VideoTagHeader + AVCVIDEOPACKET: AVCDecoderConfigurationRecord(ISO 14496-15) / One or more NALUs(four-bytes length + NALU)
///Audio: FLV AudioTagHeader + AACAUDIODATA: AudioSpecificConfig(14496-3) / Raw AAC frame data in UI8
///@param[in] data FLV Audio/Video Data(don't include FLV Tag Header)
///@param[in] type 8-audio, 9-video
///@return 0-ok, other-error
typedef int (*flv_muxer_handler)(void* param, int type, const void* data, size_t bytes, uint32_t timestamp);

flv_muxer_t* flv_muxer_create(flv_muxer_handler handler, void* param);
void flv_muxer_destroy(flv_muxer_t* muxer);

/// re-create AAC/AVC sequence header
int flv_muxer_reset(flv_muxer_t* muxer);

/// @param[in] data AAC ADTS stream, 0xFFF15C40011FFC...
int flv_muxer_aac(flv_muxer_t* muxer, const void* data, size_t bytes, uint32_t pts, uint32_t dts);

/// @param[in] data mp3 stream
int flv_muxer_mp3(flv_muxer_t* muxer, const void* data, size_t bytes, uint32_t pts, uint32_t dts);

// @param[in] data H.264 start code + H.264 NALU, 0x0000000168...
int flv_muxer_avc(flv_muxer_t* muxer, const void* data, size_t bytes, uint32_t pts, uint32_t dts);

// @param[in] data H.265 start code + H.265 NALU, 0x00000001...
int flv_muxer_hevc(flv_muxer_t* muxer, const void* data, size_t bytes, uint32_t pts, uint32_t dts);

/// @param[in] nalu H.264 NAL unit, don't include start code, 0x68...
int flv_muxer_h264_nalu(flv_muxer_t* muxer, const void* nalu, size_t bytes, uint32_t pts, uint32_t dts);

// @param[in] nalu H.265 NAL unit, don't include start code, ...
int flv_muxer_hevc_nalu(flv_muxer_t* muxer, const void* nalu, size_t bytes, uint32_t pts, uint32_t dts);

struct flv_metadata_t
{
	int audiocodecid;
	double audiodatarate; // kbps
	int audiosamplerate;
	int audiosamplesize;
	int stereo;

	int videocodecid;
	double videodatarate; // kbps
	double framerate; // fps
	int width;
	int height;
};

int flv_muxer_metadata(flv_muxer_t* muxer, const struct flv_metadata_t* metadata);

#if defined(__cplusplus)
}
#endif
#endif /* !_flv_muxer_h_ */
