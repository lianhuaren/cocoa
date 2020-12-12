//
//  EncoderDemoViewController.m
//  Encoder Demo
//
//  Created by Geraint Davies on 11/01/2013.
//  Copyright (c) 2013 GDCL http://www.gdcl.co.uk/license.htm
//

#import "EncoderDemoViewController.h"
#import "CameraServer.h"
#import <OpenGLES/ES2/glext.h>
#import "NALUnit.h"
#import "sys/stat.h"
#import "MP4Atom.h"
#import "AVencoder.h"

unsigned int to_host(unsigned char* p);

@interface EncoderDemoViewController()
{
    // writer output file (input to our extractor) and monitoring
    NSFileHandle* _inputFile;
//    dispatch_queue_t _readQueue;
//    dispatch_source_t _readSource;
    
    // index of current file name
    BOOL _swapping;
    
    // param set data
    NSData* _avcC;
    int _lengthSize;
    
    // POC
    POCState _pocState;
    int _prevPOC;
    
    // location of mdat
    BOOL _foundMDAT;
    uint64_t _posMDAT;
    int _bytesToNextAtom;
    BOOL _needParams;
    
    // tracking if NALU is next frame
    int _prev_nal_idc;
    int _prev_nal_type;
    // array of NSData comprising a single frame. each data is one nalu with no start code
    NSMutableArray* _pendingNALU;
    
    
    // FIFO for frame times
    NSMutableArray* _times;
    
    // FIFO for frames awaiting time assigment
    NSMutableArray* _frames;
    
    // estimate bitrate over first second
    int _bitspersecond;
    double _firstpts;
}
@end

@implementation EncoderDemoViewController

@synthesize cameraView;
@synthesize serverAddress;

- (void)viewDidLoad
{
    [super viewDidLoad];
//    GLvoid *data = glMapBufferOES(GL_ARRAY_BUFFER, GL_WRITE_ONLY_OES);
//
//    [self startPreview];
    
    [self onParamsCompletion];
}

- (void) willAnimateRotationToInterfaceOrientation:(UIInterfaceOrientation)toInterfaceOrientation duration:(NSTimeInterval)duration
{
    // this is not the most beautiful animation...
//    AVCaptureVideoPreviewLayer* preview = [[CameraServer server] getPreviewLayer];
//    preview.frame = self.cameraView.bounds;
//    [[preview connection] setVideoOrientation:toInterfaceOrientation];
}

- (void) startPreview
{
//    AVCaptureVideoPreviewLayer* preview = [[CameraServer server] getPreviewLayer];
//    [preview removeFromSuperlayer];
//    preview.frame = self.cameraView.bounds;
//    [[preview connection] setVideoOrientation:UIInterfaceOrientationPortrait];
//
//    [self.cameraView.layer addSublayer:preview];
//
//    self.serverAddress.text = [[CameraServer server] getURL];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void) onParamsCompletion
{
    // the initial one-frame-only file has been completed
    // Extract the avcC structure and then start monitoring the
    // main file to extract video from the mdat chunk.
    
    NSString* path = [[NSBundle mainBundle] pathForResource:@"params" ofType:@"mp4"];
    if ([self parseParams:path])
    {
//        if (_paramsBlock)
//        {
//            _paramsBlock(_avcC);
//        }
//        _headerWriter = nil;
        _swapping = NO;
        path = [[NSBundle mainBundle] pathForResource:@"capture" ofType:@"mp4"];
        _inputFile = [NSFileHandle fileHandleForReadingAtPath:path];
//        _readQueue = dispatch_queue_create("uk.co.gdcl.avencoder.read", DISPATCH_QUEUE_SERIAL);
//
//        _readSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, [_inputFile fileDescriptor], 0, _readQueue);
//        dispatch_source_set_event_handler(_readSource, ^{
//            [self onFileUpdate];
//        });
//        dispatch_resume(_readSource);
        
        [self onFileUpdate];
    }
}

- (BOOL) parseParams:(NSString*) path
{
    NSFileHandle* file = [NSFileHandle fileHandleForReadingAtPath:path];
    struct stat s;
    fstat([file fileDescriptor], &s);
    MP4Atom* movie = [MP4Atom atomAt:0 size:(int)s.st_size type:(OSType)('file') inFile:file];
    MP4Atom* moov = [movie childOfType:(OSType)('moov') startAt:0];
    MP4Atom* trak = nil;
    if (moov != nil)
    {
        for (;;)
        {
            trak = [moov nextChild];
            if (trak == nil)
            {
                break;
            }
            
            if (trak.type == (OSType)('trak'))
            {
                MP4Atom* tkhd = [trak childOfType:(OSType)('tkhd') startAt:0];
                NSData* verflags = [tkhd readAt:0 size:4];
                unsigned char* p = (unsigned char*)[verflags bytes];
                if (p[3] & 1)
                {
                    break;
                }
                else
                {
                    tkhd = nil;
                }
            }
        }
    }
    MP4Atom* stsd = nil;
    if (trak != nil)
    {
        MP4Atom* media = [trak childOfType:(OSType)('mdia') startAt:0];
        if (media != nil)
        {
            MP4Atom* minf = [media childOfType:(OSType)('minf') startAt:0];
            if (minf != nil)
            {
                MP4Atom* stbl = [minf childOfType:(OSType)('stbl') startAt:0];
                if (stbl != nil)
                {
                    stsd = [stbl childOfType:(OSType)('stsd') startAt:0];
                }
            }
        }
    }
    if (stsd != nil)
    {
        MP4Atom* avc1 = [stsd childOfType:(OSType)('avc1') startAt:8];
        if (avc1 != nil)
        {
            MP4Atom* esd = [avc1 childOfType:(OSType)('avcC') startAt:78];
            if (esd != nil)
            {
                // this is the avcC record that we are looking for
                _avcC = [esd readAt:0 size:(int)esd.length];
                if (_avcC != nil)
                {
                    // extract size of length field
                    unsigned char* p = (unsigned char*)[_avcC bytes];
                    _lengthSize = (p[4] & 3) + 1;
                    
                    avcCHeader avc((const BYTE*)[_avcC bytes], (int)[_avcC length]);
                    _pocState.SetHeader(&avc);
                    
                    return YES;
                }
            }
        }
    }
    return NO;
}

- (void) onFileUpdate
{
    // called whenever there is more data to read in the main encoder output file.
    
    struct stat s;
    fstat([_inputFile fileDescriptor], &s);
    int cReady = (int)(s.st_size - [_inputFile offsetInFile]);

    // locate the mdat atom if needed
    while (!_foundMDAT && (cReady > 8))
    {
        if (_bytesToNextAtom == 0)
        {
            NSData* hdr = [_inputFile readDataOfLength:8];
            cReady -= 8;
            unsigned char* p = (unsigned char*) [hdr bytes];
            int lenAtom = to_host(p);
            unsigned int nameAtom = to_host(p+4);
            if (nameAtom == (unsigned int)('mdat'))
            {
                _foundMDAT = true;
                _posMDAT = [_inputFile offsetInFile] - 8;
            }
            else
            {
                _bytesToNextAtom = lenAtom - 8;
            }
        }
        if (_bytesToNextAtom > 0)
        {
            int cThis = cReady < _bytesToNextAtom ? cReady :_bytesToNextAtom;
            _bytesToNextAtom -= cThis;
            [_inputFile seekToFileOffset:[_inputFile offsetInFile]+cThis];
            cReady -= cThis;
        }
    }
    if (!_foundMDAT)
    {
        return;
    }

    // the mdat must be just encoded video.
    [self readAndDeliver:cReady];
}

- (void) readAndDeliver:(uint32_t) cReady
{
    // Identify the individual NALUs and extract them
    while (cReady > _lengthSize)
    {
        NSData* lenField = [_inputFile readDataOfLength:_lengthSize];
        cReady -= _lengthSize;
        unsigned char* p = (unsigned char*) [lenField bytes];
        unsigned int lenNALU = to_host(p);
        
        if (lenNALU > cReady)
        {
            // whole NALU not present -- seek back to start of NALU and wait for more
            [_inputFile seekToFileOffset:[_inputFile offsetInFile] - 4];
            break;
        }
        NSData* nalu = [_inputFile readDataOfLength:lenNALU];
        cReady -= lenNALU;
        
        [self onNALU:nalu];
    }
}

// combine multiple NALUs into a single frame, and in the process, convert to BSF
// by adding 00 00 01 startcodes before each NALU.
- (void) onNALU:(NSData*) nalu
{
    unsigned char* pNal = (unsigned char*)[nalu bytes];
    int idc = pNal[0] & 0x60;
    int naltype = pNal[0] & 0x1f;

    if (_pendingNALU)
    {
        NALUnit nal(pNal, (int)[nalu length]);
        
        // we have existing data —is this the same frame?
        // typically there are a couple of NALUs per frame in iOS encoding.
        // This is not general-purpose: it assumes that arbitrary slice ordering is not allowed.
        BOOL bNew = NO;
        
        // sei and param sets go with following nalu
        if (_prev_nal_type < 6)
        {
            if (naltype >= 6)
            {
                bNew = YES;
            }
            else if ((idc != _prev_nal_idc) && ((idc == 0) || (_prev_nal_idc == 0)))
            {
                bNew = YES;
            }
            else if ((naltype != _prev_nal_type) && (naltype == 5))
            {
                bNew = YES;
            }
            else if ((naltype >= 1) && (naltype <= 5))
            {
                nal.Skip(8);
                int first_mb = (int)nal.GetUE();
                if (first_mb == 0)
                {
                    bNew = YES;
                }
            }
        }
        
        if (bNew)
        {
            [self onEncodedFrame];
            _pendingNALU = nil;
        }
    }
    _prev_nal_type = naltype;
    _prev_nal_idc = idc;
    if (_pendingNALU == nil)
    {
        _pendingNALU = [NSMutableArray arrayWithCapacity:2];
    }
    [_pendingNALU addObject:nalu];
}

- (NSData*) getConfigData
{
    return [_avcC copy];
}

- (void) onEncodedFrame
{
    int poc = 0;
    for (NSData* d in _pendingNALU)
    {
        NALUnit nal((const BYTE*)[d bytes], (int)[d length]);
        if (_pocState.GetPOC(&nal, &poc))
        {
            break;
        }
    }
    
    if (poc == 0)
    {
        [self processStoredFrames];
        double pts = 0;
        int index = 0;
        @synchronized(_times)
        {
            if ([_times count] > 0)
            {
                pts = [_times[index] doubleValue];
                [_times removeObjectAtIndex:index];
            }
        }
        [self deliverFrame:_pendingNALU withTime:pts];
        _prevPOC = 0;
    }
    else
    {
        EncodedFrame* f = [[EncodedFrame alloc] initWithData:_pendingNALU andPOC:poc];
        if (poc > _prevPOC)
        {
            // all pending frames come before this, so share out the
            // timestamps in order of POC
            [self processStoredFrames];
            _prevPOC = poc;
        }
        if (_frames == nil)
        {
            _frames = [NSMutableArray arrayWithCapacity:2];
        }
        [_frames addObject:f];
    }
}

- (void) processStoredFrames
{
    // first has the last timestamp and rest use up timestamps from the start
    int n = 0;
    for (EncodedFrame* f in _frames)
    {
        int index = 0;
        if (n == 0)
        {
            index = (int) [_frames count] - 1;
        }
        else
        {
            index = n-1;
        }
        double pts = 0;
        @synchronized(_times)
        {
            if ([_times count] > 0)
            {
                pts = [_times[index] doubleValue];
            }
        }
        [self deliverFrame:f.frame withTime:pts];
        n++;
    }
    @synchronized(_times)
    {
        [_times removeObjectsInRange:NSMakeRange(0, [_frames count])];
    }
    [_frames removeAllObjects];
}

- (void) deliverFrame: (NSArray*) frame withTime:(double) pts
{

    if (_firstpts < 0)
    {
        _firstpts = pts;
    }
    if ((pts - _firstpts) < 1)
    {
        int bytes = 0;
        for (NSData* data in frame)
        {
            bytes += [data length];
        }
        _bitspersecond += (bytes * 8);
    }
 
//    if (_outputBlock != nil)
//    {
//        _outputBlock(frame, pts);
//    }
    
}

@end
