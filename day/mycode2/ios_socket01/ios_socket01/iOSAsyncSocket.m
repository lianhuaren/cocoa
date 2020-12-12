//
//  iOSAsyncSocket.m
//  ios_socket01
//
//  Created by  on 2020/5/20.
//  Copyright © 2020 admin. All rights reserved.
//

#import "iOSAsyncSocket.h"

#if TARGET_OS_IPHONE
#import <CFNetwork/CFNetwork.h>
#endif

#import <TargetConditionals.h>
#import <arpa/inet.h>
#import <fcntl.h>
#import <ifaddrs.h>
#import <netdb.h>
#import <netinet/in.h>
#import <net/if.h>
#import <sys/socket.h>
#import <sys/types.h>
#import <sys/ioctl.h>
#import <sys/poll.h>
#import <sys/uio.h>
#import <sys/un.h>
#import <unistd.h>

#ifndef iOSAsyncSocketLoggingEnabled
#define iOSAsyncSocketLoggingEnabled 0
#endif

#if iOSAsyncSocketLoggingEnabled

// Logging Enabled - See log level below

// Logging uses the CocoaLumberjack framework (which is also GCD based).
// https://github.com/robbiehanson/CocoaLumberjack
//
// It allows us to do a lot of logging without significantly slowing down the code.
#import "DDLog.h"

#define LogAsync   YES
#define LogContext iOSAsyncSocketLoggingContext

#define LogObjc(flg, frmt, ...) LOG_OBJC_MAYBE(LogAsync, logLevel, flg, LogContext, frmt, ##__VA_ARGS__)
#define LogC(flg, frmt, ...)    LOG_C_MAYBE(LogAsync, logLevel, flg, LogContext, frmt, ##__VA_ARGS__)

#define LogError(frmt, ...)     LogObjc(LOG_FLAG_ERROR,   (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)
#define LogWarn(frmt, ...)      LogObjc(LOG_FLAG_WARN,    (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)
#define LogInfo(frmt, ...)      LogObjc(LOG_FLAG_INFO,    (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)
#define LogVerbose(frmt, ...)   LogObjc(LOG_FLAG_VERBOSE, (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)

#define LogCError(frmt, ...)    LogC(LOG_FLAG_ERROR,   (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)
#define LogCWarn(frmt, ...)     LogC(LOG_FLAG_WARN,    (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)
#define LogCInfo(frmt, ...)     LogC(LOG_FLAG_INFO,    (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)
#define LogCVerbose(frmt, ...)  LogC(LOG_FLAG_VERBOSE, (@"%@: " frmt), THIS_FILE, ##__VA_ARGS__)

#define LogTrace()              LogObjc(LOG_FLAG_VERBOSE, @"%@: %@", THIS_FILE, THIS_METHOD)
#define LogCTrace()             LogC(LOG_FLAG_VERBOSE, @"%@: %s", THIS_FILE, __FUNCTION__)

#ifndef iOSAsyncSocketLogLevel
#define iOSAsyncSocketLogLevel LOG_LEVEL_VERBOSE
#endif

// Log levels : off, error, warn, info, verbose
static const int logLevel = iOSAsyncSocketLogLevel;

#else

// Logging Disabled

#define LogError(frmt, ...)     {}
#define LogWarn(frmt, ...)      {}
#define LogInfo(frmt, ...)      {}
#define LogVerbose(frmt, ...)   {}

#define LogCError(frmt, ...)    {}
#define LogCWarn(frmt, ...)     {}
#define LogCInfo(frmt, ...)     {}
#define LogCVerbose(frmt, ...)  {}

#define LogTrace()              {}
#define LogCTrace(frmt, ...)    {}

#endif

/**
 * Seeing a return statements within an inner block
 * can sometimes be mistaken for a return point of the enclosing method.
 * This makes inline blocks a bit easier to read.
**/
#define return_from_block  return

/**
 * A socket file descriptor is really just an integer.
 * It represents the index of the socket within the kernel.
 * This makes invalid file descriptor comparisons easier to read.
**/
#define SOCKET_NULL -1

NSString *const iOSAsyncSocketException = @"iOSAsyncSocketException";
NSString *const iOSAsyncSocketErrorDomain = @"iOSAsyncSocketErrorDomain";

NSString *const iOSAsyncSocketQueueName = @"iOSAsyncSocket";

enum iOSAsyncSocketFlags
{
    kSocketStarted                 = 1 <<  0,  // If set, socket has been started (accepting/connecting)
    kConnected                     = 1 <<  1,  // If set, the socket is connected
    kForbidReadsWrites             = 1 <<  2,  // If set, no new reads or writes are allowed
    kReadsPaused                   = 1 <<  3,  // If set, reads are paused due to possible timeout
    kWritesPaused                  = 1 <<  4,  // If set, writes are paused due to possible timeout
    kDisconnectAfterReads          = 1 <<  5,  // If set, disconnect after no more reads are queued
    kDisconnectAfterWrites         = 1 <<  6,  // If set, disconnect after no more writes are queued
    kSocketCanAcceptBytes          = 1 <<  7,  // If set, we know socket can accept bytes. If unset, it's unknown.
    kReadSourceSuspended           = 1 <<  8,  // If set, the read source is suspended
    kWriteSourceSuspended          = 1 <<  9,  // If set, the write source is suspended
    kQueuedTLS                     = 1 << 10,  // If set, we've queued an upgrade to TLS
    kStartingReadTLS               = 1 << 11,  // If set, we're waiting for TLS negotiation to complete
    kStartingWriteTLS              = 1 << 12,  // If set, we're waiting for TLS negotiation to complete
    kSocketSecure                  = 1 << 13,  // If set, socket is using secure communication via SSL/TLS
    kSocketHasReadEOF              = 1 << 14,  // If set, we have read EOF from socket
    kReadStreamClosed              = 1 << 15,  // If set, we've read EOF plus prebuffer has been drained
    kDealloc                       = 1 << 16,  // If set, the socket is being deallocated
#if TARGET_OS_IPHONE
    kAddedStreamsToRunLoop         = 1 << 17,  // If set, CFStreams have been added to listener thread
    kUsingCFStreamForTLS           = 1 << 18,  // If set, we're forced to use CFStream instead of SecureTransport
    kSecureSocketHasBytesAvailable = 1 << 19,  // If set, CFReadStream has notified us of bytes available
#endif
};

/**
 * A PreBuffer is used when there is more data available on the socket
 * than is being requested by current read request.
 * In this case we slurp up all data from the socket (to minimize sys calls),
 * and store additional yet unread data in a "prebuffer".
 *
 * The prebuffer is entirely drained before we read from the socket again.
 * In other words, a large chunk of data is written is written to the prebuffer.
 * The prebuffer is then drained via a series of one or more reads (for subsequent read request(s)).
 *
 * A ring buffer was once used for this purpose.
 * But a ring buffer takes up twice as much memory as needed (double the size for mirroring).
 * In fact, it generally takes up more than twice the needed size as everything has to be rounded up to vm_page_size.
 * And since the prebuffer is always completely drained after being written to, a full ring buffer isn't needed.
 *
 * The current design is very simple and straight-forward, while also keeping memory requirements lower.
**/

@interface iOSAsyncSocketPreBuffer : NSObject
{
    uint8_t *preBuffer;
    size_t preBufferSize;
    
    uint8_t *readPointer;
    uint8_t *writePointer;
}

- (instancetype)initWithCapacity:(size_t)numBytes NS_DESIGNATED_INITIALIZER;

- (void)ensureCapacityForWrite:(size_t)numBytes;

- (size_t)availableBytes;
- (uint8_t *)readBuffer;

- (void)getReadBuffer:(uint8_t **)bufferPtr availableBytes:(size_t *)availableBytesPtr;

- (size_t)availableSpace;
- (uint8_t *)writeBuffer;

- (void)getWriteBuffer:(uint8_t **)bufferPtr availableSpace:(size_t *)availableSpacePtr;

- (void)didRead:(size_t)bytesRead;
- (void)didWrite:(size_t)bytesWritten;

- (void)reset;

@end

@implementation iOSAsyncSocketPreBuffer

// Cover the superclass' designated initializer
- (instancetype)init NS_UNAVAILABLE
{
    NSAssert(0, @"Use the designated initializer");
    return nil;
}

- (instancetype)initWithCapacity:(size_t)numBytes
{
    if ((self = [super init]))
    {
        preBufferSize = numBytes;
        preBuffer = malloc(preBufferSize);
        
        readPointer = preBuffer;
        writePointer = preBuffer;
    }
    return self;
}

- (void)dealloc
{
    if (preBuffer)
        free(preBuffer);
}

- (void)ensureCapacityForWrite:(size_t)numBytes
{
    size_t availableSpace = [self availableSpace];
    
    if (numBytes > availableSpace)
    {
        size_t additionalBytes = numBytes - availableSpace;
        
        size_t newPreBufferSize = preBufferSize + additionalBytes;
        uint8_t *newPreBuffer = realloc(preBuffer, newPreBufferSize);
        
        size_t readPointerOffset = readPointer - preBuffer;
        size_t writePointerOffset = writePointer - preBuffer;
        
        preBuffer = newPreBuffer;
        preBufferSize = newPreBufferSize;
        
        readPointer = preBuffer + readPointerOffset;
        writePointer = preBuffer + writePointerOffset;
    }
}

- (size_t)availableBytes
{
    return writePointer - readPointer;
}

- (uint8_t *)readBuffer
{
    return readPointer;
}

- (void)getReadBuffer:(uint8_t **)bufferPtr availableBytes:(size_t *)availableBytesPtr
{
    if (bufferPtr) *bufferPtr = readPointer;
    if (availableBytesPtr) *availableBytesPtr = [self availableBytes];
}

- (void)didRead:(size_t)bytesRead
{
    readPointer += bytesRead;
    
    if (readPointer == writePointer)
    {
        // The prebuffer has been drained. Reset pointers.
        readPointer  = preBuffer;
        writePointer = preBuffer;
    }
}

- (size_t)availableSpace
{
    return preBufferSize - (writePointer - preBuffer);
}

- (uint8_t *)writeBuffer
{
    return writePointer;
}

- (void)getWriteBuffer:(uint8_t **)bufferPtr availableSpace:(size_t *)availableSpacePtr
{
    if (bufferPtr) *bufferPtr = writePointer;
    if (availableSpacePtr) *availableSpacePtr = [self availableSpace];
}

- (void)didWrite:(size_t)bytesWritten
{
    writePointer += bytesWritten;
}

- (void)reset
{
    readPointer  = preBuffer;
    writePointer = preBuffer;
}

@end

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark -
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The GCDAsyncReadPacket encompasses the instructions for any given read.
 * The content of a read packet allows the code to determine if we're:
 *  - reading to a certain length
 *  - reading to a certain separator
 *  - or simply reading the first chunk of available data
**/
@interface GCDAsyncReadPacket : NSObject
{
  @public
    NSMutableData *buffer;
    NSUInteger startOffset;
    NSUInteger bytesDone;
    NSUInteger maxLength;
    NSTimeInterval timeout;
    NSUInteger readLength;
    NSData *term;
    BOOL bufferOwner;
    NSUInteger originalBufferLength;
    long tag;
}
- (instancetype)initWithData:(NSMutableData *)d
                 startOffset:(NSUInteger)s
                   maxLength:(NSUInteger)m
                     timeout:(NSTimeInterval)t
                  readLength:(NSUInteger)l
                  terminator:(NSData *)e
                         tag:(long)i NS_DESIGNATED_INITIALIZER;

- (void)ensureCapacityForAdditionalDataOfLength:(NSUInteger)bytesToRead;

- (NSUInteger)optimalReadLengthWithDefault:(NSUInteger)defaultValue shouldPreBuffer:(BOOL *)shouldPreBufferPtr;

- (NSUInteger)readLengthForNonTermWithHint:(NSUInteger)bytesAvailable;
- (NSUInteger)readLengthForTermWithHint:(NSUInteger)bytesAvailable shouldPreBuffer:(BOOL *)shouldPreBufferPtr;
- (NSUInteger)readLengthForTermWithPreBuffer:(iOSAsyncSocketPreBuffer *)preBuffer found:(BOOL *)foundPtr;

- (NSInteger)searchForTermAfterPreBuffering:(ssize_t)numBytes;

@end

@implementation GCDAsyncReadPacket

// Cover the superclass' designated initializer
- (instancetype)init NS_UNAVAILABLE
{
    NSAssert(0, @"Use the designated initializer");
    return nil;
}

- (instancetype)initWithData:(NSMutableData *)d
                 startOffset:(NSUInteger)s
                   maxLength:(NSUInteger)m
                     timeout:(NSTimeInterval)t
                  readLength:(NSUInteger)l
                  terminator:(NSData *)e
                         tag:(long)i
{
    if((self = [super init]))
    {
        bytesDone = 0;
        maxLength = m;
        timeout = t;
        readLength = l;
        term = [e copy];
        tag = i;
        
        if (d)
        {
            buffer = d;
            startOffset = s;
            bufferOwner = NO;
            originalBufferLength = [d length];
        }
        else
        {
            if (readLength > 0)
                buffer = [[NSMutableData alloc] initWithLength:readLength];
            else
                buffer = [[NSMutableData alloc] initWithLength:0];
            
            startOffset = 0;
            bufferOwner = YES;
            originalBufferLength = 0;
        }
    }
    return self;
}

/**
 * Increases the length of the buffer (if needed) to ensure a read of the given size will fit.
**/
- (void)ensureCapacityForAdditionalDataOfLength:(NSUInteger)bytesToRead
{
    NSUInteger buffSize = [buffer length];
    NSUInteger buffUsed = startOffset + bytesDone;
    
    NSUInteger buffSpace = buffSize - buffUsed;
    
    if (bytesToRead > buffSpace)
    {
        NSUInteger buffInc = bytesToRead - buffSpace;
        
        [buffer increaseLengthBy:buffInc];
    }
}

/**
 * This method is used when we do NOT know how much data is available to be read from the socket.
 * This method returns the default value unless it exceeds the specified readLength or maxLength.
 *
 * Furthermore, the shouldPreBuffer decision is based upon the packet type,
 * and whether the returned value would fit in the current buffer without requiring a resize of the buffer.
**/
- (NSUInteger)optimalReadLengthWithDefault:(NSUInteger)defaultValue shouldPreBuffer:(BOOL *)shouldPreBufferPtr
{
    NSUInteger result;
    
    if (readLength > 0)
    {
        // Read a specific length of data
        result = readLength - bytesDone;
        
        // There is no need to prebuffer since we know exactly how much data we need to read.
        // Even if the buffer isn't currently big enough to fit this amount of data,
        // it would have to be resized eventually anyway.
        
        if (shouldPreBufferPtr)
            *shouldPreBufferPtr = NO;
    }
    else
    {
        // Either reading until we find a specified terminator,
        // or we're simply reading all available data.
        //
        // In other words, one of:
        //
        // - readDataToData packet
        // - readDataWithTimeout packet
        
        if (maxLength > 0)
            result =  MIN(defaultValue, (maxLength - bytesDone));
        else
            result = defaultValue;
        
        // Since we don't know the size of the read in advance,
        // the shouldPreBuffer decision is based upon whether the returned value would fit
        // in the current buffer without requiring a resize of the buffer.
        //
        // This is because, in all likelyhood, the amount read from the socket will be less than the default value.
        // Thus we should avoid over-allocating the read buffer when we can simply use the pre-buffer instead.
        
        if (shouldPreBufferPtr)
        {
            NSUInteger buffSize = [buffer length];
            NSUInteger buffUsed = startOffset + bytesDone;
            
            NSUInteger buffSpace = buffSize - buffUsed;
            
            if (buffSpace >= result)
                *shouldPreBufferPtr = NO;
            else
                *shouldPreBufferPtr = YES;
        }
    }
    
    return result;
}

/**
 * For read packets without a set terminator, returns the amount of data
 * that can be read without exceeding the readLength or maxLength.
 *
 * The given parameter indicates the number of bytes estimated to be available on the socket,
 * which is taken into consideration during the calculation.
 *
 * The given hint MUST be greater than zero.
**/
- (NSUInteger)readLengthForNonTermWithHint:(NSUInteger)bytesAvailable
{
    NSAssert(term == nil, @"This method does not apply to term reads");
    NSAssert(bytesAvailable > 0, @"Invalid parameter: bytesAvailable");
    
    if (readLength > 0)
    {
        // Read a specific length of data
        
        return MIN(bytesAvailable, (readLength - bytesDone));
        
        // No need to avoid resizing the buffer.
        // If the user provided their own buffer,
        // and told us to read a certain length of data that exceeds the size of the buffer,
        // then it is clear that our code will resize the buffer during the read operation.
        //
        // This method does not actually do any resizing.
        // The resizing will happen elsewhere if needed.
    }
    else
    {
        // Read all available data
        
        NSUInteger result = bytesAvailable;
        
        if (maxLength > 0)
        {
            result = MIN(result, (maxLength - bytesDone));
        }
        
        // No need to avoid resizing the buffer.
        // If the user provided their own buffer,
        // and told us to read all available data without giving us a maxLength,
        // then it is clear that our code might resize the buffer during the read operation.
        //
        // This method does not actually do any resizing.
        // The resizing will happen elsewhere if needed.
        
        return result;
    }
}

/**
 * For read packets with a set terminator, returns the amount of data
 * that can be read without exceeding the maxLength.
 *
 * The given parameter indicates the number of bytes estimated to be available on the socket,
 * which is taken into consideration during the calculation.
 *
 * To optimize memory allocations, mem copies, and mem moves
 * the shouldPreBuffer boolean value will indicate if the data should be read into a prebuffer first,
 * or if the data can be read directly into the read packet's buffer.
**/
- (NSUInteger)readLengthForTermWithHint:(NSUInteger)bytesAvailable shouldPreBuffer:(BOOL *)shouldPreBufferPtr
{
    NSAssert(term != nil, @"This method does not apply to non-term reads");
    NSAssert(bytesAvailable > 0, @"Invalid parameter: bytesAvailable");
    
    
    NSUInteger result = bytesAvailable;
    
    if (maxLength > 0)
    {
        result = MIN(result, (maxLength - bytesDone));
    }
    
    // Should the data be read into the read packet's buffer, or into a pre-buffer first?
    //
    // One would imagine the preferred option is the faster one.
    // So which one is faster?
    //
    // Reading directly into the packet's buffer requires:
    // 1. Possibly resizing packet buffer (malloc/realloc)
    // 2. Filling buffer (read)
    // 3. Searching for term (memcmp)
    // 4. Possibly copying overflow into prebuffer (malloc/realloc, memcpy)
    //
    // Reading into prebuffer first:
    // 1. Possibly resizing prebuffer (malloc/realloc)
    // 2. Filling buffer (read)
    // 3. Searching for term (memcmp)
    // 4. Copying underflow into packet buffer (malloc/realloc, memcpy)
    // 5. Removing underflow from prebuffer (memmove)
    //
    // Comparing the performance of the two we can see that reading
    // data into the prebuffer first is slower due to the extra memove.
    //
    // However:
    // The implementation of NSMutableData is open source via core foundation's CFMutableData.
    // Decreasing the length of a mutable data object doesn't cause a realloc.
    // In other words, the capacity of a mutable data object can grow, but doesn't shrink.
    //
    // This means the prebuffer will rarely need a realloc.
    // The packet buffer, on the other hand, may often need a realloc.
    // This is especially true if we are the buffer owner.
    // Furthermore, if we are constantly realloc'ing the packet buffer,
    // and then moving the overflow into the prebuffer,
    // then we're consistently over-allocating memory for each term read.
    // And now we get into a bit of a tradeoff between speed and memory utilization.
    //
    // The end result is that the two perform very similarly.
    // And we can answer the original question very simply by another means.
    //
    // If we can read all the data directly into the packet's buffer without resizing it first,
    // then we do so. Otherwise we use the prebuffer.
    
    if (shouldPreBufferPtr)
    {
        NSUInteger buffSize = [buffer length];
        NSUInteger buffUsed = startOffset + bytesDone;
        
        if ((buffSize - buffUsed) >= result)
            *shouldPreBufferPtr = NO;
        else
            *shouldPreBufferPtr = YES;
    }
    
    return result;
}

/**
 * For read packets with a set terminator,
 * returns the amount of data that can be read from the given preBuffer,
 * without going over a terminator or the maxLength.
 *
 * It is assumed the terminator has not already been read.
**/
- (NSUInteger)readLengthForTermWithPreBuffer:(iOSAsyncSocketPreBuffer *)preBuffer found:(BOOL *)foundPtr
{
    NSAssert(term != nil, @"This method does not apply to non-term reads");
    NSAssert([preBuffer availableBytes] > 0, @"Invoked with empty pre buffer!");
    
    // We know that the terminator, as a whole, doesn't exist in our own buffer.
    // But it is possible that a _portion_ of it exists in our buffer.
    // So we're going to look for the terminator starting with a portion of our own buffer.
    //
    // Example:
    //
    // term length      = 3 bytes
    // bytesDone        = 5 bytes
    // preBuffer length = 5 bytes
    //
    // If we append the preBuffer to our buffer,
    // it would look like this:
    //
    // ---------------------
    // |B|B|B|B|B|P|P|P|P|P|
    // ---------------------
    //
    // So we start our search here:
    //
    // ---------------------
    // |B|B|B|B|B|P|P|P|P|P|
    // -------^-^-^---------
    //
    // And move forwards...
    //
    // ---------------------
    // |B|B|B|B|B|P|P|P|P|P|
    // ---------^-^-^-------
    //
    // Until we find the terminator or reach the end.
    //
    // ---------------------
    // |B|B|B|B|B|P|P|P|P|P|
    // ---------------^-^-^-
    
    BOOL found = NO;
    
    NSUInteger termLength = [term length];
    NSUInteger preBufferLength = [preBuffer availableBytes];
    
    if ((bytesDone + preBufferLength) < termLength)
    {
        // Not enough data for a full term sequence yet
        return preBufferLength;
    }
    
    NSUInteger maxPreBufferLength;
    if (maxLength > 0) {
        maxPreBufferLength = MIN(preBufferLength, (maxLength - bytesDone));
        
        // Note: maxLength >= termLength
    }
    else {
        maxPreBufferLength = preBufferLength;
    }
    
    uint8_t seq[termLength];
    const void *termBuf = [term bytes];
    
    NSUInteger bufLen = MIN(bytesDone, (termLength - 1));
    uint8_t *buf = (uint8_t *)[buffer mutableBytes] + startOffset + bytesDone - bufLen;
    
    NSUInteger preLen = termLength - bufLen;
    const uint8_t *pre = [preBuffer readBuffer];
    
    NSUInteger loopCount = bufLen + maxPreBufferLength - termLength + 1; // Plus one. See example above.
    
    NSUInteger result = maxPreBufferLength;
    
    NSUInteger i;
    for (i = 0; i < loopCount; i++)
    {
        if (bufLen > 0)
        {
            // Combining bytes from buffer and preBuffer
            
            memcpy(seq, buf, bufLen);
            memcpy(seq + bufLen, pre, preLen);
            
            if (memcmp(seq, termBuf, termLength) == 0)
            {
                result = preLen;
                found = YES;
                break;
            }
            
            buf++;
            bufLen--;
            preLen++;
        }
        else
        {
            // Comparing directly from preBuffer
            
            if (memcmp(pre, termBuf, termLength) == 0)
            {
                NSUInteger preOffset = pre - [preBuffer readBuffer]; // pointer arithmetic
                
                result = preOffset + termLength;
                found = YES;
                break;
            }
            
            pre++;
        }
    }
    
    // There is no need to avoid resizing the buffer in this particular situation.
    
    if (foundPtr) *foundPtr = found;
    return result;
}

/**
 * For read packets with a set terminator, scans the packet buffer for the term.
 * It is assumed the terminator had not been fully read prior to the new bytes.
 *
 * If the term is found, the number of excess bytes after the term are returned.
 * If the term is not found, this method will return -1.
 *
 * Note: A return value of zero means the term was found at the very end.
 *
 * Prerequisites:
 * The given number of bytes have been added to the end of our buffer.
 * Our bytesDone variable has NOT been changed due to the prebuffered bytes.
**/
- (NSInteger)searchForTermAfterPreBuffering:(ssize_t)numBytes
{
    NSAssert(term != nil, @"This method does not apply to non-term reads");
    
    // The implementation of this method is very similar to the above method.
    // See the above method for a discussion of the algorithm used here.
    
    uint8_t *buff = [buffer mutableBytes];
    NSUInteger buffLength = bytesDone + numBytes;
    
    const void *termBuff = [term bytes];
    NSUInteger termLength = [term length];
    
    // Note: We are dealing with unsigned integers,
    // so make sure the math doesn't go below zero.
    
    NSUInteger i = ((buffLength - numBytes) >= termLength) ? (buffLength - numBytes - termLength + 1) : 0;
    
    while (i + termLength <= buffLength)
    {
        uint8_t *subBuffer = buff + startOffset + i;
        
        if (memcmp(subBuffer, termBuff, termLength) == 0)
        {
            return buffLength - (i + termLength);
        }
        
        i++;
    }
    
    return -1;
}


@end

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark -
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * The GCDAsyncWritePacket encompasses the instructions for any given write.
**/
@interface GCDAsyncWritePacket : NSObject
{
  @public
    NSData *buffer;
    NSUInteger bytesDone;
    long tag;
    NSTimeInterval timeout;
}
- (instancetype)initWithData:(NSData *)d timeout:(NSTimeInterval)t tag:(long)i NS_DESIGNATED_INITIALIZER;
@end

@implementation GCDAsyncWritePacket

// Cover the superclass' designated initializer
- (instancetype)init NS_UNAVAILABLE
{
    NSAssert(0, @"Use the designated initializer");
    return nil;
}

- (instancetype)initWithData:(NSData *)d timeout:(NSTimeInterval)t tag:(long)i
{
    if((self = [super init]))
    {
        buffer = d; // Retain not copy. For performance as documented in header file.
        bytesDone = 0;
        timeout = t;
        tag = i;
    }
    return self;
}


@end

@implementation iOSAsyncSocket
{
    uint32_t flags;
    
    __weak id<iOSAsyncSocketDelegate> delegate;
    dispatch_queue_t delegateQueue;
    
    int socket4FD;
    
    dispatch_queue_t socketQueue;
    
    dispatch_source_t accept4Source;
    
    dispatch_source_t readSource;
    dispatch_source_t writeSource;
    
    dispatch_source_t readTimer;
    dispatch_source_t writeTimer;
    
    NSMutableArray *readQueue;
    NSMutableArray *writeQueue;
    
    GCDAsyncReadPacket *currentRead;
    GCDAsyncWritePacket *currentWrite;
    
    unsigned long socketFDBytesAvailable;
    
    iOSAsyncSocketPreBuffer *preBuffer;
    
    void *IsOnSocketQueueOrTargetQueueKey;
}

- (instancetype)init
{
    return [self initWithDelegate:nil delegateQueue:NULL socketQueue:NULL];
}

- (instancetype)initWithDelegate:(id<iOSAsyncSocketDelegate>)aDelegate delegateQueue:(dispatch_queue_t)dq
{
    return [self initWithDelegate:aDelegate delegateQueue:dq socketQueue:NULL];
}

- (instancetype)initWithDelegate:(id<iOSAsyncSocketDelegate>)aDelegate delegateQueue:(dispatch_queue_t)dq socketQueue:(dispatch_queue_t)sq
{
    if((self = [super init]))
    {
        delegate = aDelegate;
        delegateQueue = dq;
        
        #if !OS_OBJECT_USE_OBJC
        if (dq) dispatch_retain(dq);
        #endif
        
        socket4FD = SOCKET_NULL;
//        socket6FD = SOCKET_NULL;
//        socketUN = SOCKET_NULL;
//        socketUrl = nil;
//        stateIndex = 0;
        
        if (sq)
        {
            NSAssert(sq != dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0),
                     @"The given socketQueue parameter must not be a concurrent queue.");
            NSAssert(sq != dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
                     @"The given socketQueue parameter must not be a concurrent queue.");
            NSAssert(sq != dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                     @"The given socketQueue parameter must not be a concurrent queue.");
            
            socketQueue = sq;
            #if !OS_OBJECT_USE_OBJC
            dispatch_retain(sq);
            #endif
        }
        else
        {
            socketQueue = dispatch_queue_create([iOSAsyncSocketQueueName UTF8String], NULL);
        }
        
        // The dispatch_queue_set_specific() and dispatch_get_specific() functions take a "void *key" parameter.
        // From the documentation:
        //
        // > Keys are only compared as pointers and are never dereferenced.
        // > Thus, you can use a pointer to a static variable for a specific subsystem or
        // > any other value that allows you to identify the value uniquely.
        //
        // We're just going to use the memory address of an ivar.
        // Specifically an ivar that is explicitly named for our purpose to make the code more readable.
        //
        // However, it feels tedious (and less readable) to include the "&" all the time:
        // dispatch_get_specific(&IsOnSocketQueueOrTargetQueueKey)
        //
        // So we're going to make it so it doesn't matter if we use the '&' or not,
        // by assigning the value of the ivar to the address of the ivar.
        // Thus: IsOnSocketQueueOrTargetQueueKey == &IsOnSocketQueueOrTargetQueueKey;
        
        IsOnSocketQueueOrTargetQueueKey = &IsOnSocketQueueOrTargetQueueKey;
        
        void *nonNullUnusedPointer = (__bridge void *)self;
        dispatch_queue_set_specific(socketQueue, IsOnSocketQueueOrTargetQueueKey, nonNullUnusedPointer, NULL);
        
        readQueue = [[NSMutableArray alloc] initWithCapacity:5];
        currentRead = nil;
        
        writeQueue = [[NSMutableArray alloc] initWithCapacity:5];
        currentWrite = nil;
        
        preBuffer = [[iOSAsyncSocketPreBuffer alloc] initWithCapacity:(1024 * 4)];
//        alternateAddressDelay = 0.3;
    }
    return self;
}

- (void)dealloc
{
    LogInfo(@"%@ - %@ (start)", THIS_METHOD, self);
    
    // Set dealloc flag.
    // This is used by closeWithError to ensure we don't accidentally retain ourself.
    flags |= kDealloc;
    
    if (dispatch_get_specific(IsOnSocketQueueOrTargetQueueKey))
    {
        [self closeWithError:nil];
    }
    else
    {
        dispatch_sync(socketQueue, ^{
            [self closeWithError:nil];
        });
    }
    
    delegate = nil;
    
    #if !OS_OBJECT_USE_OBJC
    if (delegateQueue) dispatch_release(delegateQueue);
    #endif
    delegateQueue = NULL;
    
    #if !OS_OBJECT_USE_OBJC
    if (socketQueue) dispatch_release(socketQueue);
    #endif
    socketQueue = NULL;
    
    LogInfo(@"%@ - %@ (finish)", THIS_METHOD, self);
}


#pragma mark Accepting
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (BOOL)acceptOnPort:(uint16_t)port error:(NSError **)errPtr
{
    return [self acceptOnInterface:nil port:port error:errPtr];
}

- (BOOL)acceptOnInterface:(NSString *)inInterface port:(uint16_t)port error:(NSError **)errPtr
{
    LogTrace();
    
    // Just in-case interface parameter is immutable.
    NSString *interface = [inInterface copy];
    
    __block BOOL result = NO;
    __block NSError *err = nil;
    
    // CreateSocket Block
    // This block will be invoked within the dispatch block below.
    
    int(^createSocket)(int, NSData*) = ^int (int domain, NSData *interfaceAddr) {
        
        int socketFD = socket(domain, SOCK_STREAM, 0);
        
        if (socketFD == SOCKET_NULL)
        {
            NSString *reason = @"Error in socket() function";
            err = [self errorWithErrno:errno reason:reason];
            
            return SOCKET_NULL;
        }
        
        int status;
        
        // Set socket options
        
        status = fcntl(socketFD, F_SETFL, O_NONBLOCK);
        if (status == -1)
        {
            NSString *reason = @"Error enabling non-blocking IO on socket (fcntl)";
            err = [self errorWithErrno:errno reason:reason];
            
            LogVerbose(@"close(socketFD)");
            close(socketFD);
            return SOCKET_NULL;
        }
        
        int reuseOn = 1;
        status = setsockopt(socketFD, SOL_SOCKET, SO_REUSEADDR, &reuseOn, sizeof(reuseOn));
        if (status == -1)
        {
            NSString *reason = @"Error enabling address reuse (setsockopt)";
            err = [self errorWithErrno:errno reason:reason];
            
            LogVerbose(@"close(socketFD)");
            close(socketFD);
            return SOCKET_NULL;
        }
        
        // Bind socket
        
        status = bind(socketFD, (const struct sockaddr *)[interfaceAddr bytes], (socklen_t)[interfaceAddr length]);
        if (status == -1)
        {
            NSString *reason = @"Error in bind() function";
            err = [self errorWithErrno:errno reason:reason];
            
            LogVerbose(@"close(socketFD)");
            close(socketFD);
            return SOCKET_NULL;
        }
        
        // Listen
        
        status = listen(socketFD, 1024);
        if (status == -1)
        {
            NSString *reason = @"Error in listen() function";
            err = [self errorWithErrno:errno reason:reason];
            
            LogVerbose(@"close(socketFD)");
            close(socketFD);
            return SOCKET_NULL;
        }
        
        return socketFD;
    };
    
    // Create dispatch block and run on socketQueue
    
    dispatch_block_t block = ^{ @autoreleasepool {
        
        if (self->delegate == nil) // Must have delegate set
        {
            NSString *msg = @"Attempting to accept without a delegate. Set a delegate first.";
            err = [self badConfigError:msg];
            
            return_from_block;
        }
        
        if (self->delegateQueue == NULL) // Must have delegate queue set
        {
            NSString *msg = @"Attempting to accept without a delegate queue. Set a delegate queue first.";
            err = [self badConfigError:msg];
            
            return_from_block;
        }
        
        BOOL isIPv4Disabled = NO;//(self->config & kIPv4Disabled) ? YES : NO;
        BOOL isIPv6Disabled = NO;//(self->config & kIPv6Disabled) ? YES : NO;
//
//        if (isIPv4Disabled && isIPv6Disabled) // Must have IPv4 or IPv6 enabled
//        {
//            NSString *msg = @"Both IPv4 and IPv6 have been disabled. Must enable at least one protocol first.";
//            err = [self badConfigError:msg];
//
//            return_from_block;
//        }
        
        if (![self isDisconnected]) // Must be disconnected
        {
            NSString *msg = @"Attempting to accept while connected or accepting connections. Disconnect first.";
            err = [self badConfigError:msg];
            
            return_from_block;
        }
        
        // Clear queues (spurious read/write requests post disconnect)
        [self->readQueue removeAllObjects];
        [self->writeQueue removeAllObjects];
        
        // Resolve interface from description
        
        NSMutableData *interface4 = nil;
        NSMutableData *interface6 = nil;
        
        [self getInterfaceAddress4:&interface4 address6:&interface6 fromDescription:interface port:port];
        
        if ((interface4 == nil) && (interface6 == nil))
        {
            NSString *msg = @"Unknown interface. Specify valid interface by name (e.g. \"en1\") or IP address.";
            err = [self badParamError:msg];
            
            return_from_block;
        }
        
        if (isIPv4Disabled && (interface6 == nil))
        {
            NSString *msg = @"IPv4 has been disabled and specified interface doesn't support IPv6.";
            err = [self badParamError:msg];
            
            return_from_block;
        }
        
        if (isIPv6Disabled && (interface4 == nil))
        {
            NSString *msg = @"IPv6 has been disabled and specified interface doesn't support IPv4.";
            err = [self badParamError:msg];
            
            return_from_block;
        }
        
        BOOL enableIPv4 = !isIPv4Disabled && (interface4 != nil);
//        BOOL enableIPv6 = !isIPv6Disabled && (interface6 != nil);
        
        // Create sockets, configure, bind, and listen
        
        if (enableIPv4)
        {
            LogVerbose(@"Creating IPv4 socket");
            self->socket4FD = createSocket(AF_INET, interface4);
            
            if (self->socket4FD == SOCKET_NULL)
            {
                return_from_block;
            }
        }
        
//        if (enableIPv6)
//        {
//            LogVerbose(@"Creating IPv6 socket");
//
//            if (enableIPv4 && (port == 0))
//            {
//                // No specific port was specified, so we allowed the OS to pick an available port for us.
//                // Now we need to make sure the IPv6 socket listens on the same port as the IPv4 socket.
//
//                struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)[interface6 mutableBytes];
//                addr6->sin6_port = htons([self localPort4]);
//            }
//
//            self->socket6FD = createSocket(AF_INET6, interface6);
//
//            if (self->socket6FD == SOCKET_NULL)
//            {
//                if (self->socket4FD != SOCKET_NULL)
//                {
//                    LogVerbose(@"close(socket4FD)");
//                    close(self->socket4FD);
//                    self->socket4FD = SOCKET_NULL;
//                }
//
//                return_from_block;
//            }
//        }
        
        // Create accept sources
        
        if (enableIPv4)
        {
            self->accept4Source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, self->socket4FD, 0, self->socketQueue);
            
            int socketFD = self->socket4FD;
            dispatch_source_t acceptSource = self->accept4Source;
            
            __weak iOSAsyncSocket *weakSelf = self;
            
            dispatch_source_set_event_handler(self->accept4Source, ^{ @autoreleasepool {
            #pragma clang diagnostic push
            #pragma clang diagnostic warning "-Wimplicit-retain-self"
                
                __strong iOSAsyncSocket *strongSelf = weakSelf;
                if (strongSelf == nil) return_from_block;
                
                LogVerbose(@"event4Block");
                
                unsigned long i = 0;
                unsigned long numPendingConnections = dispatch_source_get_data(acceptSource);
                
                LogVerbose(@"numPendingConnections: %lu", numPendingConnections);
                
                while ([strongSelf doAccept:socketFD] && (++i < numPendingConnections));
                
            #pragma clang diagnostic pop
            }});
            
            
            dispatch_source_set_cancel_handler(self->accept4Source, ^{
            #pragma clang diagnostic push
            #pragma clang diagnostic warning "-Wimplicit-retain-self"
                
                #if !OS_OBJECT_USE_OBJC
                LogVerbose(@"dispatch_release(accept4Source)");
                dispatch_release(acceptSource);
                #endif
                
                LogVerbose(@"close(socket4FD)");
                close(socketFD);
            
            #pragma clang diagnostic pop
            });
            
            LogVerbose(@"dispatch_resume(accept4Source)");
            dispatch_resume(self->accept4Source);
        }
        
//        if (enableIPv6)
//        {
//            self->accept6Source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, self->socket6FD, 0, self->socketQueue);
//
//            int socketFD = self->socket6FD;
//            dispatch_source_t acceptSource = self->accept6Source;
//
//            __weak iOSAsyncSocket *weakSelf = self;
//
//            dispatch_source_set_event_handler(self->accept6Source, ^{ @autoreleasepool {
//            #pragma clang diagnostic push
//            #pragma clang diagnostic warning "-Wimplicit-retain-self"
//
//                __strong iOSAsyncSocket *strongSelf = weakSelf;
//                if (strongSelf == nil) return_from_block;
//
//                LogVerbose(@"event6Block");
//
//                unsigned long i = 0;
//                unsigned long numPendingConnections = dispatch_source_get_data(acceptSource);
//
//                LogVerbose(@"numPendingConnections: %lu", numPendingConnections);
//
//                while ([strongSelf doAccept:socketFD] && (++i < numPendingConnections));
//
//            #pragma clang diagnostic pop
//            }});
//
//            dispatch_source_set_cancel_handler(self->accept6Source, ^{
//            #pragma clang diagnostic push
//            #pragma clang diagnostic warning "-Wimplicit-retain-self"
//
//                #if !OS_OBJECT_USE_OBJC
//                LogVerbose(@"dispatch_release(accept6Source)");
//                dispatch_release(acceptSource);
//                #endif
//
//                LogVerbose(@"close(socket6FD)");
//                close(socketFD);
//
//            #pragma clang diagnostic pop
//            });
//
//            LogVerbose(@"dispatch_resume(accept6Source)");
//            dispatch_resume(self->accept6Source);
//        }
        
        self->flags |= kSocketStarted;
        
        result = YES;
    }};
    
    if (dispatch_get_specific(IsOnSocketQueueOrTargetQueueKey))
        block();
    else
        dispatch_sync(socketQueue, block);
    
    if (result == NO)
    {
        LogInfo(@"Error in accept: %@", err);
        
        if (errPtr)
            *errPtr = err;
    }
    
    return result;
}

- (BOOL)doAccept:(int)parentSocketFD
{
    LogTrace();
    
    int socketType;
    int childSocketFD;
    NSData *childSocketAddress;
    
    if (parentSocketFD == socket4FD)
    {
        socketType = 0;
        
        struct sockaddr_in addr;
        socklen_t addrLen = sizeof(addr);
        
        childSocketFD = accept(parentSocketFD, (struct sockaddr *)&addr, &addrLen);
        
        if (childSocketFD == -1)
        {
            LogWarn(@"Accept failed with error: %@", [self errnoError]);
            return NO;
        }
        
        childSocketAddress = [NSData dataWithBytes:&addr length:addrLen];
    }
//    else if (parentSocketFD == socket6FD)
//    {
//        socketType = 1;
//
//        struct sockaddr_in6 addr;
//        socklen_t addrLen = sizeof(addr);
//
//        childSocketFD = accept(parentSocketFD, (struct sockaddr *)&addr, &addrLen);
//
//        if (childSocketFD == -1)
//        {
//            LogWarn(@"Accept failed with error: %@", [self errnoError]);
//            return NO;
//        }
//
//        childSocketAddress = [NSData dataWithBytes:&addr length:addrLen];
//    }
//    else // if (parentSocketFD == socketUN)
//    {
//        socketType = 2;
//
//        struct sockaddr_un addr;
//        socklen_t addrLen = sizeof(addr);
//
//        childSocketFD = accept(parentSocketFD, (struct sockaddr *)&addr, &addrLen);
//
//        if (childSocketFD == -1)
//        {
//            LogWarn(@"Accept failed with error: %@", [self errnoError]);
//            return NO;
//        }
//
//        childSocketAddress = [NSData dataWithBytes:&addr length:addrLen];
//    }
    
    // Enable non-blocking IO on the socket
    
    int result = fcntl(childSocketFD, F_SETFL, O_NONBLOCK);
    if (result == -1)
    {
        LogWarn(@"Error enabling non-blocking IO on accepted socket (fcntl)");
        LogVerbose(@"close(childSocketFD)");
        close(childSocketFD);
        return NO;
    }
    
    // Prevent SIGPIPE signals
    
    int nosigpipe = 1;
    setsockopt(childSocketFD, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, sizeof(nosigpipe));
    
    // Notify delegate
    
    if (delegateQueue)
    {
        __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;
        
        dispatch_async(delegateQueue, ^{ @autoreleasepool {
            
            // Query delegate for custom socket queue
            
            dispatch_queue_t childSocketQueue = NULL;
            
            if ([theDelegate respondsToSelector:@selector(newSocketQueueForConnectionFromAddress:onSocket:)])
            {
                childSocketQueue = [theDelegate newSocketQueueForConnectionFromAddress:childSocketAddress
                                                                              onSocket:self];
            }
            
            // Create iOSAsyncSocket instance for accepted socket
            
            iOSAsyncSocket *acceptedSocket = [[[self class] alloc] initWithDelegate:theDelegate
                                                                      delegateQueue:self->delegateQueue
                                                                        socketQueue:childSocketQueue];
            
            if (socketType == 0)
                acceptedSocket->socket4FD = childSocketFD;
//            else if (socketType == 1)
//                acceptedSocket->socket6FD = childSocketFD;
//            else
//                acceptedSocket->socketUN = childSocketFD;
            
            acceptedSocket->flags = (kSocketStarted | kConnected);
            
            // Setup read and write sources for accepted socket
            
            dispatch_async(acceptedSocket->socketQueue, ^{ @autoreleasepool {
                
                [acceptedSocket setupReadAndWriteSourcesForNewlyConnectedSocket:childSocketFD];
            }});
            
            // Notify delegate
            
            if ([theDelegate respondsToSelector:@selector(socket:didAcceptNewSocket:)])
            {
                [theDelegate socket:self didAcceptNewSocket:acceptedSocket];
            }
            
            // Release the socket queue returned from the delegate (it was retained by acceptedSocket)
            #if !OS_OBJECT_USE_OBJC
            if (childSocketQueue) dispatch_release(childSocketQueue);
            #endif
            
            // The accepted socket should have been retained by the delegate.
            // Otherwise it gets properly released when exiting the block.
        }});
    }
    
    return YES;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Connecting
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Disconnecting
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (void)closeWithError:(NSError *)error
{
    LogTrace();
    NSAssert(dispatch_get_specific(IsOnSocketQueueOrTargetQueueKey), @"Must be dispatched on socketQueue");
    
    [self endConnectTimeout];
    
    if (currentRead != nil)  [self endCurrentRead];
    if (currentWrite != nil) [self endCurrentWrite];
    
    [readQueue removeAllObjects];
    [writeQueue removeAllObjects];
    
    [preBuffer reset];
    
    #if TARGET_OS_IPHONE
//    {
//        if (readStream || writeStream)
//        {
//            [self removeStreamsFromRunLoop];
//
//            if (readStream)
//            {
//                CFReadStreamSetClient(readStream, kCFStreamEventNone, NULL, NULL);
//                CFReadStreamClose(readStream);
//                CFRelease(readStream);
//                readStream = NULL;
//            }
//            if (writeStream)
//            {
//                CFWriteStreamSetClient(writeStream, kCFStreamEventNone, NULL, NULL);
//                CFWriteStreamClose(writeStream);
//                CFRelease(writeStream);
//                writeStream = NULL;
//            }
//        }
//    }
    #endif
    
//    [sslPreBuffer reset];
//    sslErrCode = lastSSLHandshakeError = noErr;
//
//    if (sslContext)
//    {
//        // Getting a linker error here about the SSLx() functions?
//        // You need to add the Security Framework to your application.
//
//        SSLClose(sslContext);
//
//        #if TARGET_OS_IPHONE || (__MAC_OS_X_VERSION_MIN_REQUIRED >= 1080)
//        CFRelease(sslContext);
//        #else
//        SSLDisposeContext(sslContext);
//        #endif
//
//        sslContext = NULL;
//    }
    
    // For some crazy reason (in my opinion), cancelling a dispatch source doesn't
    // invoke the cancel handler if the dispatch source is paused.
    // So we have to unpause the source if needed.
    // This allows the cancel handler to be run, which in turn releases the source and closes the socket.
    
    if (!accept4Source  && !readSource && !writeSource)//&& !accept6Source && !acceptUNSource
    {
        LogVerbose(@"manually closing close");

        if (socket4FD != SOCKET_NULL)
        {
            LogVerbose(@"close(socket4FD)");
            close(socket4FD);
            socket4FD = SOCKET_NULL;
        }

//        if (socket6FD != SOCKET_NULL)
//        {
//            LogVerbose(@"close(socket6FD)");
//            close(socket6FD);
//            socket6FD = SOCKET_NULL;
//        }
//
//        if (socketUN != SOCKET_NULL)
//        {
//            LogVerbose(@"close(socketUN)");
//            close(socketUN);
//            socketUN = SOCKET_NULL;
//            unlink(socketUrl.path.fileSystemRepresentation);
//            socketUrl = nil;
//        }
    }
    else
    {
        if (accept4Source)
        {
            LogVerbose(@"dispatch_source_cancel(accept4Source)");
            dispatch_source_cancel(accept4Source);
            
            // We never suspend accept4Source
            
            accept4Source = NULL;
        }
        
//        if (accept6Source)
//        {
//            LogVerbose(@"dispatch_source_cancel(accept6Source)");
//            dispatch_source_cancel(accept6Source);
//
//            // We never suspend accept6Source
//
//            accept6Source = NULL;
//        }
//
//        if (acceptUNSource)
//        {
//            LogVerbose(@"dispatch_source_cancel(acceptUNSource)");
//            dispatch_source_cancel(acceptUNSource);
//
//            // We never suspend acceptUNSource
//
//            acceptUNSource = NULL;
//        }
    
        if (readSource)
        {
            LogVerbose(@"dispatch_source_cancel(readSource)");
            dispatch_source_cancel(readSource);
            
            [self resumeReadSource];
            
            readSource = NULL;
        }
        
        if (writeSource)
        {
            LogVerbose(@"dispatch_source_cancel(writeSource)");
            dispatch_source_cancel(writeSource);
            
            [self resumeWriteSource];
            
            writeSource = NULL;
        }
        
        // The sockets will be closed by the cancel handlers of the corresponding source
        
        socket4FD = SOCKET_NULL;
//        socket6FD = SOCKET_NULL;
//        socketUN = SOCKET_NULL;
    }
    
    // If the client has passed the connect/accept method, then the connection has at least begun.
    // Notify delegate that it is now ending.
    BOOL shouldCallDelegate = (flags & kSocketStarted) ? YES : NO;
    BOOL isDeallocating = (flags & kDealloc) ? YES : NO;
    
    // Clear stored socket info and all flags (config remains as is)
    socketFDBytesAvailable = 0;
    flags = 0;
//    sslWriteCachedLength = 0;
    
    if (shouldCallDelegate)
    {
        __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;
        __strong id theSelf = isDeallocating ? nil : self;
        
        if (delegateQueue && [theDelegate respondsToSelector: @selector(socketDidDisconnect:withError:)])
        {
            dispatch_async(delegateQueue, ^{ @autoreleasepool {
                
                [theDelegate socketDidDisconnect:theSelf withError:error];
            }});
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Utilities
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Finds the address of an interface description.
 * An inteface description may be an interface name (en0, en1, lo0) or corresponding IP (192.168.4.34).
 *
 * The interface description may optionally contain a port number at the end, separated by a colon.
 * If a non-zero port parameter is provided, any port number in the interface description is ignored.
 *
 * The returned value is a 'struct sockaddr' wrapped in an NSMutableData object.
**/
- (void)getInterfaceAddress4:(NSMutableData **)interfaceAddr4Ptr
                    address6:(NSMutableData **)interfaceAddr6Ptr
             fromDescription:(NSString *)interfaceDescription
                        port:(uint16_t)port
{
    NSMutableData *addr4 = nil;
    NSMutableData *addr6 = nil;
    
    NSString *interface = nil;
    
    NSArray *components = [interfaceDescription componentsSeparatedByString:@":"];
    if ([components count] > 0)
    {
        NSString *temp = [components objectAtIndex:0];
        if ([temp length] > 0)
        {
            interface = temp;
        }
    }
    if ([components count] > 1 && port == 0)
    {
        NSString *temp = [components objectAtIndex:1];
        long portL = strtol([temp UTF8String], NULL, 10);
        
        if (portL > 0 && portL <= UINT16_MAX)
        {
            port = (uint16_t)portL;
        }
    }
    
    if (interface == nil)
    {
        // ANY address
        
        struct sockaddr_in sockaddr4;
        memset(&sockaddr4, 0, sizeof(sockaddr4));
        
        sockaddr4.sin_len         = sizeof(sockaddr4);
        sockaddr4.sin_family      = AF_INET;
        sockaddr4.sin_port        = htons(port);
        sockaddr4.sin_addr.s_addr = htonl(INADDR_ANY);
        
        struct sockaddr_in6 sockaddr6;
        memset(&sockaddr6, 0, sizeof(sockaddr6));
        
        sockaddr6.sin6_len       = sizeof(sockaddr6);
        sockaddr6.sin6_family    = AF_INET6;
        sockaddr6.sin6_port      = htons(port);
        sockaddr6.sin6_addr      = in6addr_any;
        
        addr4 = [NSMutableData dataWithBytes:&sockaddr4 length:sizeof(sockaddr4)];
        addr6 = [NSMutableData dataWithBytes:&sockaddr6 length:sizeof(sockaddr6)];
    }
    else if ([interface isEqualToString:@"localhost"] || [interface isEqualToString:@"loopback"])
    {
        // LOOPBACK address
        
        struct sockaddr_in sockaddr4;
        memset(&sockaddr4, 0, sizeof(sockaddr4));
        
        sockaddr4.sin_len         = sizeof(sockaddr4);
        sockaddr4.sin_family      = AF_INET;
        sockaddr4.sin_port        = htons(port);
        sockaddr4.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        
        struct sockaddr_in6 sockaddr6;
        memset(&sockaddr6, 0, sizeof(sockaddr6));
        
        sockaddr6.sin6_len       = sizeof(sockaddr6);
        sockaddr6.sin6_family    = AF_INET6;
        sockaddr6.sin6_port      = htons(port);
        sockaddr6.sin6_addr      = in6addr_loopback;
        
        addr4 = [NSMutableData dataWithBytes:&sockaddr4 length:sizeof(sockaddr4)];
        addr6 = [NSMutableData dataWithBytes:&sockaddr6 length:sizeof(sockaddr6)];
    }
    else
    {
        const char *iface = [interface UTF8String];
        
        struct ifaddrs *addrs;
        const struct ifaddrs *cursor;
        
        if ((getifaddrs(&addrs) == 0))
        {
            cursor = addrs;
            while (cursor != NULL)
            {
                if ((addr4 == nil) && (cursor->ifa_addr->sa_family == AF_INET))
                {
                    // IPv4
                    
                    struct sockaddr_in nativeAddr4;
                    memcpy(&nativeAddr4, cursor->ifa_addr, sizeof(nativeAddr4));
                    
                    if (strcmp(cursor->ifa_name, iface) == 0)
                    {
                        // Name match
                        
                        nativeAddr4.sin_port = htons(port);
                        
                        addr4 = [NSMutableData dataWithBytes:&nativeAddr4 length:sizeof(nativeAddr4)];
                    }
                    else
                    {
                        char ip[INET_ADDRSTRLEN];
                        
                        const char *conversion = inet_ntop(AF_INET, &nativeAddr4.sin_addr, ip, sizeof(ip));
                        
                        if ((conversion != NULL) && (strcmp(ip, iface) == 0))
                        {
                            // IP match
                            
                            nativeAddr4.sin_port = htons(port);
                            
                            addr4 = [NSMutableData dataWithBytes:&nativeAddr4 length:sizeof(nativeAddr4)];
                        }
                    }
                }
                else if ((addr6 == nil) && (cursor->ifa_addr->sa_family == AF_INET6))
                {
                    // IPv6
                    
                    struct sockaddr_in6 nativeAddr6;
                    memcpy(&nativeAddr6, cursor->ifa_addr, sizeof(nativeAddr6));
                    
                    if (strcmp(cursor->ifa_name, iface) == 0)
                    {
                        // Name match
                        
                        nativeAddr6.sin6_port = htons(port);
                        
                        addr6 = [NSMutableData dataWithBytes:&nativeAddr6 length:sizeof(nativeAddr6)];
                    }
                    else
                    {
                        char ip[INET6_ADDRSTRLEN];
                        
                        const char *conversion = inet_ntop(AF_INET6, &nativeAddr6.sin6_addr, ip, sizeof(ip));
                        
                        if ((conversion != NULL) && (strcmp(ip, iface) == 0))
                        {
                            // IP match
                            
                            nativeAddr6.sin6_port = htons(port);
                            
                            addr6 = [NSMutableData dataWithBytes:&nativeAddr6 length:sizeof(nativeAddr6)];
                        }
                    }
                }
                
                cursor = cursor->ifa_next;
            }
            
            freeifaddrs(addrs);
        }
    }
    
    if (interfaceAddr4Ptr) *interfaceAddr4Ptr = addr4;
    if (interfaceAddr6Ptr) *interfaceAddr6Ptr = addr6;
}

- (void)setupReadAndWriteSourcesForNewlyConnectedSocket:(int)socketFD
{
    readSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, socketFD, 0, socketQueue);
    writeSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_WRITE, socketFD, 0, socketQueue);
    
    // Setup event handlers
    
    __weak iOSAsyncSocket *weakSelf = self;
    
    dispatch_source_set_event_handler(readSource, ^{ @autoreleasepool {
    #pragma clang diagnostic push
    #pragma clang diagnostic warning "-Wimplicit-retain-self"
        
        __strong iOSAsyncSocket *strongSelf = weakSelf;
        if (strongSelf == nil) return_from_block;
        
        LogVerbose(@"readEventBlock");
        
        strongSelf->socketFDBytesAvailable = dispatch_source_get_data(strongSelf->readSource);
        LogVerbose(@"socketFDBytesAvailable: %lu", strongSelf->socketFDBytesAvailable);
        
        if (strongSelf->socketFDBytesAvailable > 0)
            [strongSelf doReadData];
        else
            [strongSelf doReadEOF];
        
    #pragma clang diagnostic pop
    }});
    
    dispatch_source_set_event_handler(writeSource, ^{ @autoreleasepool {
    #pragma clang diagnostic push
    #pragma clang diagnostic warning "-Wimplicit-retain-self"
        
        __strong iOSAsyncSocket *strongSelf = weakSelf;
        if (strongSelf == nil) return_from_block;
        
        LogVerbose(@"writeEventBlock");
        
        strongSelf->flags |= kSocketCanAcceptBytes;
        [strongSelf doWriteData];
        
    #pragma clang diagnostic pop
    }});
    
    // Setup cancel handlers
    
    __block int socketFDRefCount = 2;
    
    #if !OS_OBJECT_USE_OBJC
    dispatch_source_t theReadSource = readSource;
    dispatch_source_t theWriteSource = writeSource;
    #endif
    
    dispatch_source_set_cancel_handler(readSource, ^{
    #pragma clang diagnostic push
    #pragma clang diagnostic warning "-Wimplicit-retain-self"
        
        LogVerbose(@"readCancelBlock");
        
        #if !OS_OBJECT_USE_OBJC
        LogVerbose(@"dispatch_release(readSource)");
        dispatch_release(theReadSource);
        #endif
        
        if (--socketFDRefCount == 0)
        {
            LogVerbose(@"close(socketFD)");
            close(socketFD);
        }
        
    #pragma clang diagnostic pop
    });
    
    dispatch_source_set_cancel_handler(writeSource, ^{
    #pragma clang diagnostic push
    #pragma clang diagnostic warning "-Wimplicit-retain-self"
        
        LogVerbose(@"writeCancelBlock");
        
        #if !OS_OBJECT_USE_OBJC
        LogVerbose(@"dispatch_release(writeSource)");
        dispatch_release(theWriteSource);
        #endif
        
        if (--socketFDRefCount == 0)
        {
            LogVerbose(@"close(socketFD)");
            close(socketFD);
        }
        
    #pragma clang diagnostic pop
    });
    
    // We will not be able to read until data arrives.
    // But we should be able to write immediately.
    
    socketFDBytesAvailable = 0;
    flags &= ~kReadSourceSuspended;
    
    LogVerbose(@"dispatch_resume(readSource)");
    dispatch_resume(readSource);
    
    flags |= kSocketCanAcceptBytes;
    flags |= kWriteSourceSuspended;
}

- (void)suspendReadSource
{
    if (!(flags & kReadSourceSuspended))
    {
        LogVerbose(@"dispatch_suspend(readSource)");
        
        dispatch_suspend(readSource);
        flags |= kReadSourceSuspended;
    }
}

- (void)resumeReadSource
{
    if (flags & kReadSourceSuspended)
    {
        LogVerbose(@"dispatch_resume(readSource)");
        
        dispatch_resume(readSource);
        flags &= ~kReadSourceSuspended;
    }
}

- (void)suspendWriteSource
{
    if (!(flags & kWriteSourceSuspended))
    {
        LogVerbose(@"dispatch_suspend(writeSource)");
        
        dispatch_suspend(writeSource);
        flags |= kWriteSourceSuspended;
    }
}

- (void)resumeWriteSource
{
    if (flags & kWriteSourceSuspended)
    {
        LogVerbose(@"dispatch_resume(writeSource)");
        
        dispatch_resume(writeSource);
        flags &= ~kWriteSourceSuspended;
    }
}

- (BOOL)usingCFStreamForTLS
{
    #if TARGET_OS_IPHONE
    
    if ((flags & kSocketSecure) && (flags & kUsingCFStreamForTLS))
    {
        // The startTLS method was given the iOSAsyncSocketUseCFStreamForTLS flag.
        
        return YES;
    }
    
    #endif
    
    return NO;
}

- (void)endConnectTimeout
{
    LogTrace();
    
//    if (connectTimer)
//    {
//        dispatch_source_cancel(connectTimer);
//        connectTimer = NULL;
//    }
//
//    // Increment stateIndex.
//    // This will prevent us from processing results from any related background asynchronous operations.
//    //
//    // Note: This should be called from close method even if connectTimer is NULL.
//    // This is because one might disconnect a socket prior to a successful connection which had no timeout.
//
//    stateIndex++;
//
//    if (connectInterface4)
//    {
//        connectInterface4 = nil;
//    }
//    if (connectInterface6)
//    {
//        connectInterface6 = nil;
//    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Reading
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (void)readDataWithTimeout:(NSTimeInterval)timeout tag:(long)tag
{
    [self readDataWithTimeout:timeout buffer:nil bufferOffset:0 maxLength:0 tag:tag];
}

- (void)readDataWithTimeout:(NSTimeInterval)timeout
                     buffer:(NSMutableData *)buffer
               bufferOffset:(NSUInteger)offset
                        tag:(long)tag
{
    [self readDataWithTimeout:timeout buffer:buffer bufferOffset:offset maxLength:0 tag:tag];
}

- (void)readDataWithTimeout:(NSTimeInterval)timeout
                     buffer:(NSMutableData *)buffer
               bufferOffset:(NSUInteger)offset
                  maxLength:(NSUInteger)length
                        tag:(long)tag
{
    if (offset > [buffer length]) {
        LogWarn(@"Cannot read: offset > [buffer length]");
        return;
    }
    
    GCDAsyncReadPacket *packet = [[GCDAsyncReadPacket alloc] initWithData:buffer
                                                              startOffset:offset
                                                                maxLength:length
                                                                  timeout:timeout
                                                               readLength:0
                                                               terminator:nil
                                                                      tag:tag];
    
    dispatch_async(socketQueue, ^{ @autoreleasepool {
        
        LogTrace();
        
        if ((self->flags & kSocketStarted) && !(self->flags & kForbidReadsWrites))
        {
            [self->readQueue addObject:packet];
            [self maybeDequeueRead];
        }
    }});
    
    // Do not rely on the block being run in order to release the packet,
    // as the queue might get released without the block completing.
}

/**
 * This method starts a new read, if needed.
 *
 * It is called when:
 * - a user requests a read
 * - after a read request has finished (to handle the next request)
 * - immediately after the socket opens to handle any pending requests
 *
 * This method also handles auto-disconnect post read/write completion.
**/
- (void)maybeDequeueRead
{
    LogTrace();
    NSAssert(dispatch_get_specific(IsOnSocketQueueOrTargetQueueKey), @"Must be dispatched on socketQueue");
    
    // If we're not currently processing a read AND we have an available read stream
    if ((currentRead == nil) && (flags & kConnected))
    {
        if ([readQueue count] > 0)
        {
            // Dequeue the next object in the write queue
            currentRead = [readQueue objectAtIndex:0];
            [readQueue removeObjectAtIndex:0];
            
            
//            if ([currentRead isKindOfClass:[GCDAsyncSpecialPacket class]])
//            {
//                LogVerbose(@"Dequeued GCDAsyncSpecialPacket");
//
//                // Attempt to start TLS
//                flags |= kStartingReadTLS;
//
//                // This method won't do anything unless both kStartingReadTLS and kStartingWriteTLS are set
//                [self maybeStartTLS];
//            }
//            else
            {
                LogVerbose(@"Dequeued GCDAsyncReadPacket");
                
                // Setup read timer (if needed)
                [self setupReadTimerWithTimeout:currentRead->timeout];
                
                // Immediately read, if possible
                [self doReadData];
            }
        }
        else if (flags & kDisconnectAfterReads)
        {
            if (flags & kDisconnectAfterWrites)
            {
                if (([writeQueue count] == 0) && (currentWrite == nil))
                {
                    [self closeWithError:nil];
                }
            }
            else
            {
                [self closeWithError:nil];
            }
        }
//        else if (flags & kSocketSecure)
//        {
//            [self flushSSLBuffers];
//
//            // Edge case:
//            //
//            // We just drained all data from the ssl buffers,
//            // and all known data from the socket (socketFDBytesAvailable).
//            //
//            // If we didn't get any data from this process,
//            // then we may have reached the end of the TCP stream.
//            //
//            // Be sure callbacks are enabled so we're notified about a disconnection.
//
//            if ([preBuffer availableBytes] == 0)
//            {
//                if ([self usingCFStreamForTLS]) {
//                    // Callbacks never disabled
//                }
//                else {
//                    [self resumeReadSource];
//                }
//            }
//        }
    }
}

- (void)doReadData
{
    LogTrace();
    
    // This method is called on the socketQueue.
    // It might be called directly, or via the readSource when data is available to be read.
    
    if ((currentRead == nil) || (flags & kReadsPaused))
    {
        LogVerbose(@"No currentRead or kReadsPaused");
        
        // Unable to read at this time
        
//        if (flags & kSocketSecure)
//        {
//            // Here's the situation:
//            //
//            // We have an established secure connection.
//            // There may not be a currentRead, but there might be encrypted data sitting around for us.
//            // When the user does get around to issuing a read, that encrypted data will need to be decrypted.
//            //
//            // So why make the user wait?
//            // We might as well get a head start on decrypting some data now.
//            //
//            // The other reason we do this has to do with detecting a socket disconnection.
//            // The SSL/TLS protocol has it's own disconnection handshake.
//            // So when a secure socket is closed, a "goodbye" packet comes across the wire.
//            // We want to make sure we read the "goodbye" packet so we can properly detect the TCP disconnection.
//
//            [self flushSSLBuffers];
//        }
        
        if ([self usingCFStreamForTLS])
        {
            // CFReadStream only fires once when there is available data.
            // It won't fire again until we've invoked CFReadStreamRead.
        }
        else
        {
            // If the readSource is firing, we need to pause it
            // or else it will continue to fire over and over again.
            //
            // If the readSource is not firing,
            // we want it to continue monitoring the socket.
            
            if (socketFDBytesAvailable > 0)
            {
                [self suspendReadSource];
            }
        }
        return;
    }
    
    BOOL hasBytesAvailable = NO;
    unsigned long estimatedBytesAvailable = 0;
    
//    if ([self usingCFStreamForTLS])
//    {
//        #if TARGET_OS_IPHONE
//
//        // Requested CFStream, rather than SecureTransport, for TLS (via iOSAsyncSocketUseCFStreamForTLS)
//
//        estimatedBytesAvailable = 0;
//        if ((flags & kSecureSocketHasBytesAvailable) && CFReadStreamHasBytesAvailable(readStream))
//            hasBytesAvailable = YES;
//        else
//            hasBytesAvailable = NO;
//
//        #endif
//    }
//    else
    {
        estimatedBytesAvailable = socketFDBytesAvailable;
        
//        if (flags & kSocketSecure)
//        {
//            // There are 2 buffers to be aware of here.
//            //
//            // We are using SecureTransport, a TLS/SSL security layer which sits atop TCP.
//            // We issue a read to the SecureTranport API, which in turn issues a read to our SSLReadFunction.
//            // Our SSLReadFunction then reads from the BSD socket and returns the encrypted data to SecureTransport.
//            // SecureTransport then decrypts the data, and finally returns the decrypted data back to us.
//            //
//            // The first buffer is one we create.
//            // SecureTransport often requests small amounts of data.
//            // This has to do with the encypted packets that are coming across the TCP stream.
//            // But it's non-optimal to do a bunch of small reads from the BSD socket.
//            // So our SSLReadFunction reads all available data from the socket (optimizing the sys call)
//            // and may store excess in the sslPreBuffer.
//
//            estimatedBytesAvailable += [sslPreBuffer availableBytes];
//
//            // The second buffer is within SecureTransport.
//            // As mentioned earlier, there are encrypted packets coming across the TCP stream.
//            // SecureTransport needs the entire packet to decrypt it.
//            // But if the entire packet produces X bytes of decrypted data,
//            // and we only asked SecureTransport for X/2 bytes of data,
//            // it must store the extra X/2 bytes of decrypted data for the next read.
//            //
//            // The SSLGetBufferedReadSize function will tell us the size of this internal buffer.
//            // From the documentation:
//            //
//            // "This function does not block or cause any low-level read operations to occur."
//
//            size_t sslInternalBufSize = 0;
//            SSLGetBufferedReadSize(sslContext, &sslInternalBufSize);
//
//            estimatedBytesAvailable += sslInternalBufSize;
//        }
        
        hasBytesAvailable = (estimatedBytesAvailable > 0);
    }
    
    if ((hasBytesAvailable == NO) && ([preBuffer availableBytes] == 0))
    {
        LogVerbose(@"No data available to read...");
        
        // No data available to read.
        
        if (![self usingCFStreamForTLS])
        {
            // Need to wait for readSource to fire and notify us of
            // available data in the socket's internal read buffer.
            
            [self resumeReadSource];
        }
        return;
    }
    
//    if (flags & kStartingReadTLS)
//    {
//        LogVerbose(@"Waiting for SSL/TLS handshake to complete");
//
//        // The readQueue is waiting for SSL/TLS handshake to complete.
//
//        if (flags & kStartingWriteTLS)
//        {
//            if ([self usingSecureTransportForTLS] && lastSSLHandshakeError == errSSLWouldBlock)
//            {
//                // We are in the process of a SSL Handshake.
//                // We were waiting for incoming data which has just arrived.
//
//                [self ssl_continueSSLHandshake];
//            }
//        }
//        else
//        {
//            // We are still waiting for the writeQueue to drain and start the SSL/TLS process.
//            // We now know data is available to read.
//
//            if (![self usingCFStreamForTLS])
//            {
//                // Suspend the read source or else it will continue to fire nonstop.
//
//                [self suspendReadSource];
//            }
//        }
//
//        return;
//    }
    
    BOOL done        = NO;  // Completed read operation
    NSError *error   = nil; // Error occurred
    
    NSUInteger totalBytesReadForCurrentRead = 0;
    
    //
    // STEP 1 - READ FROM PREBUFFER
    //
    
    if ([preBuffer availableBytes] > 0)
    {
        // There are 3 types of read packets:
        //
        // 1) Read all available data.
        // 2) Read a specific length of data.
        // 3) Read up to a particular terminator.
        
        NSUInteger bytesToCopy;
        
        if (currentRead->term != nil)
        {
            // Read type #3 - read up to a terminator
            
            bytesToCopy = [currentRead readLengthForTermWithPreBuffer:preBuffer found:&done];
        }
        else
        {
            // Read type #1 or #2
            
            bytesToCopy = [currentRead readLengthForNonTermWithHint:[preBuffer availableBytes]];
        }
        
        // Make sure we have enough room in the buffer for our read.
        
        [currentRead ensureCapacityForAdditionalDataOfLength:bytesToCopy];
        
        // Copy bytes from prebuffer into packet buffer
        
        uint8_t *buffer = (uint8_t *)[currentRead->buffer mutableBytes] + currentRead->startOffset +
                                                                          currentRead->bytesDone;
        
        memcpy(buffer, [preBuffer readBuffer], bytesToCopy);
        
        // Remove the copied bytes from the preBuffer
        [preBuffer didRead:bytesToCopy];
        
        LogVerbose(@"copied(%lu) preBufferLength(%zu)", (unsigned long)bytesToCopy, [preBuffer availableBytes]);
        
        // Update totals
        
        currentRead->bytesDone += bytesToCopy;
        totalBytesReadForCurrentRead += bytesToCopy;
        
        // Check to see if the read operation is done
        
        if (currentRead->readLength > 0)
        {
            // Read type #2 - read a specific length of data
            
            done = (currentRead->bytesDone == currentRead->readLength);
        }
        else if (currentRead->term != nil)
        {
            // Read type #3 - read up to a terminator
            
            // Our 'done' variable was updated via the readLengthForTermWithPreBuffer:found: method
            
            if (!done && currentRead->maxLength > 0)
            {
                // We're not done and there's a set maxLength.
                // Have we reached that maxLength yet?
                
                if (currentRead->bytesDone >= currentRead->maxLength)
                {
                    error = [self readMaxedOutError];
                }
            }
        }
        else
        {
            // Read type #1 - read all available data
            //
            // We're done as soon as
            // - we've read all available data (in prebuffer and socket)
            // - we've read the maxLength of read packet.
            
            done = ((currentRead->maxLength > 0) && (currentRead->bytesDone == currentRead->maxLength));
        }
        
    }
    
    //
    // STEP 2 - READ FROM SOCKET
    //
    
    BOOL socketEOF = (flags & kSocketHasReadEOF) ? YES : NO;  // Nothing more to read via socket (end of file)
    BOOL waiting   = !done && !error && !socketEOF && !hasBytesAvailable; // Ran out of data, waiting for more
    
    if (!done && !error && !socketEOF && hasBytesAvailable)
    {
        NSAssert(([preBuffer availableBytes] == 0), @"Invalid logic");
        
        BOOL readIntoPreBuffer = NO;
        uint8_t *buffer = NULL;
        size_t bytesRead = 0;
        
//        if (flags & kSocketSecure)
//        {
//            if ([self usingCFStreamForTLS])
//            {
//                #if TARGET_OS_IPHONE
//
//                // Using CFStream, rather than SecureTransport, for TLS
//
//                NSUInteger defaultReadLength = (1024 * 32);
//
//                NSUInteger bytesToRead = [currentRead optimalReadLengthWithDefault:defaultReadLength
//                                                                   shouldPreBuffer:&readIntoPreBuffer];
//
//                // Make sure we have enough room in the buffer for our read.
//                //
//                // We are either reading directly into the currentRead->buffer,
//                // or we're reading into the temporary preBuffer.
//
//                if (readIntoPreBuffer)
//                {
//                    [preBuffer ensureCapacityForWrite:bytesToRead];
//
//                    buffer = [preBuffer writeBuffer];
//                }
//                else
//                {
//                    [currentRead ensureCapacityForAdditionalDataOfLength:bytesToRead];
//
//                    buffer = (uint8_t *)[currentRead->buffer mutableBytes]
//                           + currentRead->startOffset
//                           + currentRead->bytesDone;
//                }
//
//                // Read data into buffer
//
//                CFIndex result = CFReadStreamRead(readStream, buffer, (CFIndex)bytesToRead);
//                LogVerbose(@"CFReadStreamRead(): result = %i", (int)result);
//
//                if (result < 0)
//                {
//                    error = (__bridge_transfer NSError *)CFReadStreamCopyError(readStream);
//                }
//                else if (result == 0)
//                {
//                    socketEOF = YES;
//                }
//                else
//                {
//                    waiting = YES;
//                    bytesRead = (size_t)result;
//                }
//
//                // We only know how many decrypted bytes were read.
//                // The actual number of bytes read was likely more due to the overhead of the encryption.
//                // So we reset our flag, and rely on the next callback to alert us of more data.
//                flags &= ~kSecureSocketHasBytesAvailable;
//
//                #endif
//            }
//            else
//            {
//                // Using SecureTransport for TLS
//                //
//                // We know:
//                // - how many bytes are available on the socket
//                // - how many encrypted bytes are sitting in the sslPreBuffer
//                // - how many decypted bytes are sitting in the sslContext
//                //
//                // But we do NOT know:
//                // - how many encypted bytes are sitting in the sslContext
//                //
//                // So we play the regular game of using an upper bound instead.
//
//                NSUInteger defaultReadLength = (1024 * 32);
//
//                if (defaultReadLength < estimatedBytesAvailable) {
//                    defaultReadLength = estimatedBytesAvailable + (1024 * 16);
//                }
//
//                NSUInteger bytesToRead = [currentRead optimalReadLengthWithDefault:defaultReadLength
//                                                                   shouldPreBuffer:&readIntoPreBuffer];
//
//                if (bytesToRead > SIZE_MAX) { // NSUInteger may be bigger than size_t
//                    bytesToRead = SIZE_MAX;
//                }
//
//                // Make sure we have enough room in the buffer for our read.
//                //
//                // We are either reading directly into the currentRead->buffer,
//                // or we're reading into the temporary preBuffer.
//
//                if (readIntoPreBuffer)
//                {
//                    [preBuffer ensureCapacityForWrite:bytesToRead];
//
//                    buffer = [preBuffer writeBuffer];
//                }
//                else
//                {
//                    [currentRead ensureCapacityForAdditionalDataOfLength:bytesToRead];
//
//                    buffer = (uint8_t *)[currentRead->buffer mutableBytes]
//                           + currentRead->startOffset
//                           + currentRead->bytesDone;
//                }
//
//                // The documentation from Apple states:
//                //
//                //     "a read operation might return errSSLWouldBlock,
//                //      indicating that less data than requested was actually transferred"
//                //
//                // However, starting around 10.7, the function will sometimes return noErr,
//                // even if it didn't read as much data as requested. So we need to watch out for that.
//
//                OSStatus result;
//                do
//                {
//                    void *loop_buffer = buffer + bytesRead;
//                    size_t loop_bytesToRead = (size_t)bytesToRead - bytesRead;
//                    size_t loop_bytesRead = 0;
//
//                    result = SSLRead(sslContext, loop_buffer, loop_bytesToRead, &loop_bytesRead);
//                    LogVerbose(@"read from secure socket = %u", (unsigned)loop_bytesRead);
//
//                    bytesRead += loop_bytesRead;
//
//                } while ((result == noErr) && (bytesRead < bytesToRead));
//
//
//                if (result != noErr)
//                {
//                    if (result == errSSLWouldBlock)
//                        waiting = YES;
//                    else
//                    {
//                        if (result == errSSLClosedGraceful || result == errSSLClosedAbort)
//                        {
//                            // We've reached the end of the stream.
//                            // Handle this the same way we would an EOF from the socket.
//                            socketEOF = YES;
//                            sslErrCode = result;
//                        }
//                        else
//                        {
//                            error = [self sslError:result];
//                        }
//                    }
//                    // It's possible that bytesRead > 0, even if the result was errSSLWouldBlock.
//                    // This happens when the SSLRead function is able to read some data,
//                    // but not the entire amount we requested.
//
//                    if (bytesRead <= 0)
//                    {
//                        bytesRead = 0;
//                    }
//                }
//
//                // Do not modify socketFDBytesAvailable.
//                // It will be updated via the SSLReadFunction().
//            }
//        }
//        else
        {
            // Normal socket operation
            
            NSUInteger bytesToRead;
            
            // There are 3 types of read packets:
            //
            // 1) Read all available data.
            // 2) Read a specific length of data.
            // 3) Read up to a particular terminator.
            
            if (currentRead->term != nil)
            {
                // Read type #3 - read up to a terminator
                
                bytesToRead = [currentRead readLengthForTermWithHint:estimatedBytesAvailable
                                                     shouldPreBuffer:&readIntoPreBuffer];
            }
            else
            {
                // Read type #1 or #2
                
                bytesToRead = [currentRead readLengthForNonTermWithHint:estimatedBytesAvailable];
            }
            
            if (bytesToRead > SIZE_MAX) { // NSUInteger may be bigger than size_t (read param 3)
                bytesToRead = SIZE_MAX;
            }
            
            // Make sure we have enough room in the buffer for our read.
            //
            // We are either reading directly into the currentRead->buffer,
            // or we're reading into the temporary preBuffer.
            
            if (readIntoPreBuffer)
            {
                [preBuffer ensureCapacityForWrite:bytesToRead];
                
                buffer = [preBuffer writeBuffer];
            }
            else
            {
                [currentRead ensureCapacityForAdditionalDataOfLength:bytesToRead];
                
                buffer = (uint8_t *)[currentRead->buffer mutableBytes]
                       + currentRead->startOffset
                       + currentRead->bytesDone;
            }
            
            // Read data into buffer
            
            int socketFD = socket4FD;//(socket4FD != SOCKET_NULL) ? socket4FD : (socket6FD != SOCKET_NULL) ? socket6FD : socketUN;
            
            ssize_t result = read(socketFD, buffer, (size_t)bytesToRead);
            LogVerbose(@"read from socket = %i", (int)result);
            
            if (result < 0)
            {
                if (errno == EWOULDBLOCK)
                    waiting = YES;
                else
                    error = [self errorWithErrno:errno reason:@"Error in read() function"];
                
                socketFDBytesAvailable = 0;
            }
            else if (result == 0)
            {
                socketEOF = YES;
                socketFDBytesAvailable = 0;
            }
            else
            {
                bytesRead = result;
                
                if (bytesRead < bytesToRead)
                {
                    // The read returned less data than requested.
                    // This means socketFDBytesAvailable was a bit off due to timing,
                    // because we read from the socket right when the readSource event was firing.
                    socketFDBytesAvailable = 0;
                }
                else
                {
                    if (socketFDBytesAvailable <= bytesRead)
                        socketFDBytesAvailable = 0;
                    else
                        socketFDBytesAvailable -= bytesRead;
                }
                
                if (socketFDBytesAvailable == 0)
                {
                    waiting = YES;
                }
            }
        }
        
        if (bytesRead > 0)
        {
            // Check to see if the read operation is done
            
            if (currentRead->readLength > 0)
            {
                // Read type #2 - read a specific length of data
                //
                // Note: We should never be using a prebuffer when we're reading a specific length of data.
                
                NSAssert(readIntoPreBuffer == NO, @"Invalid logic");
                
                currentRead->bytesDone += bytesRead;
                totalBytesReadForCurrentRead += bytesRead;
                
                done = (currentRead->bytesDone == currentRead->readLength);
            }
            else if (currentRead->term != nil)
            {
                // Read type #3 - read up to a terminator
                
                if (readIntoPreBuffer)
                {
                    // We just read a big chunk of data into the preBuffer
                    
                    [preBuffer didWrite:bytesRead];
                    LogVerbose(@"read data into preBuffer - preBuffer.length = %zu", [preBuffer availableBytes]);
                    
                    // Search for the terminating sequence
                    
                    NSUInteger bytesToCopy = [currentRead readLengthForTermWithPreBuffer:preBuffer found:&done];
                    LogVerbose(@"copying %lu bytes from preBuffer", (unsigned long)bytesToCopy);
                    
                    // Ensure there's room on the read packet's buffer
                    
                    [currentRead ensureCapacityForAdditionalDataOfLength:bytesToCopy];
                    
                    // Copy bytes from prebuffer into read buffer
                    
                    uint8_t *readBuf = (uint8_t *)[currentRead->buffer mutableBytes] + currentRead->startOffset
                                                                                     + currentRead->bytesDone;
                    
                    memcpy(readBuf, [preBuffer readBuffer], bytesToCopy);
                    
                    // Remove the copied bytes from the prebuffer
                    [preBuffer didRead:bytesToCopy];
                    LogVerbose(@"preBuffer.length = %zu", [preBuffer availableBytes]);
                    
                    // Update totals
                    currentRead->bytesDone += bytesToCopy;
                    totalBytesReadForCurrentRead += bytesToCopy;
                    
                    // Our 'done' variable was updated via the readLengthForTermWithPreBuffer:found: method above
                }
                else
                {
                    // We just read a big chunk of data directly into the packet's buffer.
                    // We need to move any overflow into the prebuffer.
                    
                    NSInteger overflow = [currentRead searchForTermAfterPreBuffering:bytesRead];
                    
                    if (overflow == 0)
                    {
                        // Perfect match!
                        // Every byte we read stays in the read buffer,
                        // and the last byte we read was the last byte of the term.
                        
                        currentRead->bytesDone += bytesRead;
                        totalBytesReadForCurrentRead += bytesRead;
                        done = YES;
                    }
                    else if (overflow > 0)
                    {
                        // The term was found within the data that we read,
                        // and there are extra bytes that extend past the end of the term.
                        // We need to move these excess bytes out of the read packet and into the prebuffer.
                        
                        NSInteger underflow = bytesRead - overflow;
                        
                        // Copy excess data into preBuffer
                        
                        LogVerbose(@"copying %ld overflow bytes into preBuffer", (long)overflow);
                        [preBuffer ensureCapacityForWrite:overflow];
                        
                        uint8_t *overflowBuffer = buffer + underflow;
                        memcpy([preBuffer writeBuffer], overflowBuffer, overflow);
                        
                        [preBuffer didWrite:overflow];
                        LogVerbose(@"preBuffer.length = %zu", [preBuffer availableBytes]);
                        
                        // Note: The completeCurrentRead method will trim the buffer for us.
                        
                        currentRead->bytesDone += underflow;
                        totalBytesReadForCurrentRead += underflow;
                        done = YES;
                    }
                    else
                    {
                        // The term was not found within the data that we read.
                        
                        currentRead->bytesDone += bytesRead;
                        totalBytesReadForCurrentRead += bytesRead;
                        done = NO;
                    }
                }
                
                if (!done && currentRead->maxLength > 0)
                {
                    // We're not done and there's a set maxLength.
                    // Have we reached that maxLength yet?
                    
                    if (currentRead->bytesDone >= currentRead->maxLength)
                    {
                        error = [self readMaxedOutError];
                    }
                }
            }
            else
            {
                // Read type #1 - read all available data
                
                if (readIntoPreBuffer)
                {
                    // We just read a chunk of data into the preBuffer
                    
                    [preBuffer didWrite:bytesRead];
                    
                    // Now copy the data into the read packet.
                    //
                    // Recall that we didn't read directly into the packet's buffer to avoid
                    // over-allocating memory since we had no clue how much data was available to be read.
                    //
                    // Ensure there's room on the read packet's buffer
                    
                    [currentRead ensureCapacityForAdditionalDataOfLength:bytesRead];
                    
                    // Copy bytes from prebuffer into read buffer
                    
                    uint8_t *readBuf = (uint8_t *)[currentRead->buffer mutableBytes] + currentRead->startOffset
                                                                                     + currentRead->bytesDone;
                    
                    memcpy(readBuf, [preBuffer readBuffer], bytesRead);
                    
                    // Remove the copied bytes from the prebuffer
                    [preBuffer didRead:bytesRead];
                    
                    // Update totals
                    currentRead->bytesDone += bytesRead;
                    totalBytesReadForCurrentRead += bytesRead;
                }
                else
                {
                    currentRead->bytesDone += bytesRead;
                    totalBytesReadForCurrentRead += bytesRead;
                }
                
                done = YES;
            }
            
        } // if (bytesRead > 0)
        
    } // if (!done && !error && !socketEOF && hasBytesAvailable)
    
    
    if (!done && currentRead->readLength == 0 && currentRead->term == nil)
    {
        // Read type #1 - read all available data
        //
        // We might arrive here if we read data from the prebuffer but not from the socket.
        
        done = (totalBytesReadForCurrentRead > 0);
    }
    
    // Check to see if we're done, or if we've made progress
    
    if (done)
    {
        [self completeCurrentRead];
        
        if (!error && (!socketEOF || [preBuffer availableBytes] > 0))
        {
            [self maybeDequeueRead];
        }
    }
    else if (totalBytesReadForCurrentRead > 0)
    {
        // We're not done read type #2 or #3 yet, but we have read in some bytes
        //
        // We ensure that `waiting` is set in order to resume the readSource (if it is suspended). It is
        // possible to reach this point and `waiting` not be set, if the current read's length is
        // sufficiently large. In that case, we may have read to some upperbound successfully, but
        // that upperbound could be smaller than the desired length.
        waiting = YES;

        __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;
        
        if (delegateQueue && [theDelegate respondsToSelector:@selector(socket:didReadPartialDataOfLength:tag:)])
        {
            long theReadTag = currentRead->tag;
            
            dispatch_async(delegateQueue, ^{ @autoreleasepool {
                
                [theDelegate socket:self didReadPartialDataOfLength:totalBytesReadForCurrentRead tag:theReadTag];
            }});
        }
    }
    
    // Check for errors
    
    if (error)
    {
        [self closeWithError:error];
    }
    else if (socketEOF)
    {
        [self doReadEOF];
    }
    else if (waiting)
    {
        if (![self usingCFStreamForTLS])
        {
            // Monitor the socket for readability (if we're not already doing so)
            [self resumeReadSource];
        }
    }
    
    // Do not add any code here without first adding return statements in the error cases above.
}

- (void)doReadEOF
{
    LogTrace();
    
    // This method may be called more than once.
    // If the EOF is read while there is still data in the preBuffer,
    // then this method may be called continually after invocations of doReadData to see if it's time to disconnect.
    
    flags |= kSocketHasReadEOF;
    
//    if (flags & kSocketSecure)
//    {
//        // If the SSL layer has any buffered data, flush it into the preBuffer now.
//
//        [self flushSSLBuffers];
//    }
    
    BOOL shouldDisconnect = NO;
    NSError *error = nil;
    
//    if ((flags & kStartingReadTLS) || (flags & kStartingWriteTLS))
//    {
//        // We received an EOF during or prior to startTLS.
//        // The SSL/TLS handshake is now impossible, so this is an unrecoverable situation.
//
//        shouldDisconnect = YES;
//
//        if ([self usingSecureTransportForTLS])
//        {
//            error = [self sslError:errSSLClosedAbort];
//        }
//    }
//    else
    if (flags & kReadStreamClosed)
    {
        // The preBuffer has already been drained.
        // The config allows half-duplex connections.
        // We've previously checked the socket, and it appeared writeable.
        // So we marked the read stream as closed and notified the delegate.
        //
        // As per the half-duplex contract, the socket will be closed when a write fails,
        // or when the socket is manually closed.
        
        shouldDisconnect = NO;
    }
    else if ([preBuffer availableBytes] > 0)
    {
        LogVerbose(@"Socket reached EOF, but there is still data available in prebuffer");
        
        // Although we won't be able to read any more data from the socket,
        // there is existing data that has been prebuffered that we can read.
        
        shouldDisconnect = NO;
    }
//    else if (config & kAllowHalfDuplexConnection)
//    {
//        // We just received an EOF (end of file) from the socket's read stream.
//        // This means the remote end of the socket (the peer we're connected to)
//        // has explicitly stated that it will not be sending us any more data.
//        //
//        // Query the socket to see if it is still writeable. (Perhaps the peer will continue reading data from us)
//
//        int socketFD = (socket4FD != SOCKET_NULL) ? socket4FD : (socket6FD != SOCKET_NULL) ? socket6FD : socketUN;
//
//        struct pollfd pfd[1];
//        pfd[0].fd = socketFD;
//        pfd[0].events = POLLOUT;
//        pfd[0].revents = 0;
//
//        poll(pfd, 1, 0);
//
//        if (pfd[0].revents & POLLOUT)
//        {
//            // Socket appears to still be writeable
//
//            shouldDisconnect = NO;
//            flags |= kReadStreamClosed;
//
//            // Notify the delegate that we're going half-duplex
//
//            __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;
//
//            if (delegateQueue && [theDelegate respondsToSelector:@selector(socketDidCloseReadStream:)])
//            {
//                dispatch_async(delegateQueue, ^{ @autoreleasepool {
//
//                    [theDelegate socketDidCloseReadStream:self];
//                }});
//            }
//        }
//        else
//        {
//            shouldDisconnect = YES;
//        }
//    }
    else
    {
        shouldDisconnect = YES;
    }
    
    
    if (shouldDisconnect)
    {
        if (error == nil)
        {
//            if ([self usingSecureTransportForTLS])
//            {
//                if (sslErrCode != noErr && sslErrCode != errSSLClosedGraceful)
//                {
//                    error = [self sslError:sslErrCode];
//                }
//                else
//                {
//                    error = [self connectionClosedError];
//                }
//            }
//            else
            {
                error = [self connectionClosedError];
            }
        }
        [self closeWithError:error];
    }
    else
    {
        if (![self usingCFStreamForTLS])
        {
            // Suspend the read source (if needed)
            
            [self suspendReadSource];
        }
    }
}

- (void)completeCurrentRead
{
    LogTrace();
    
    NSAssert(currentRead, @"Trying to complete current read when there is no current read.");
    
    
    NSData *result = nil;
    
    if (currentRead->bufferOwner)
    {
        // We created the buffer on behalf of the user.
        // Trim our buffer to be the proper size.
        [currentRead->buffer setLength:currentRead->bytesDone];
        
        result = currentRead->buffer;
    }
    else
    {
        // We did NOT create the buffer.
        // The buffer is owned by the caller.
        // Only trim the buffer if we had to increase its size.
        
        if ([currentRead->buffer length] > currentRead->originalBufferLength)
        {
            NSUInteger readSize = currentRead->startOffset + currentRead->bytesDone;
            NSUInteger origSize = currentRead->originalBufferLength;
            
            NSUInteger buffSize = MAX(readSize, origSize);
            
            [currentRead->buffer setLength:buffSize];
        }
        
        uint8_t *buffer = (uint8_t *)[currentRead->buffer mutableBytes] + currentRead->startOffset;
        
        result = [NSData dataWithBytesNoCopy:buffer length:currentRead->bytesDone freeWhenDone:NO];
    }
    
    __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;

    if (delegateQueue && [theDelegate respondsToSelector:@selector(socket:didReadData:withTag:)])
    {
        GCDAsyncReadPacket *theRead = currentRead; // Ensure currentRead retained since result may not own buffer
        
        dispatch_async(delegateQueue, ^{ @autoreleasepool {
            
            [theDelegate socket:self didReadData:result withTag:theRead->tag];
        }});
    }
    
    [self endCurrentRead];
}

- (void)endCurrentRead
{
    if (readTimer)
    {
        dispatch_source_cancel(readTimer);
        readTimer = NULL;
    }
    
    currentRead = nil;
}

- (void)setupReadTimerWithTimeout:(NSTimeInterval)timeout
{
    if (timeout >= 0.0)
    {
        readTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, socketQueue);
        
        __weak iOSAsyncSocket *weakSelf = self;
        
        dispatch_source_set_event_handler(readTimer, ^{ @autoreleasepool {
        #pragma clang diagnostic push
        #pragma clang diagnostic warning "-Wimplicit-retain-self"
            
            __strong iOSAsyncSocket *strongSelf = weakSelf;
            if (strongSelf == nil) return_from_block;
            
            [strongSelf doReadTimeout];
            
        #pragma clang diagnostic pop
        }});
        
        #if !OS_OBJECT_USE_OBJC
        dispatch_source_t theReadTimer = readTimer;
        dispatch_source_set_cancel_handler(readTimer, ^{
        #pragma clang diagnostic push
        #pragma clang diagnostic warning "-Wimplicit-retain-self"
            
            LogVerbose(@"dispatch_release(readTimer)");
            dispatch_release(theReadTimer);
            
        #pragma clang diagnostic pop
        });
        #endif
        
        dispatch_time_t tt = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC));
        
        dispatch_source_set_timer(readTimer, tt, DISPATCH_TIME_FOREVER, 0);
        dispatch_resume(readTimer);
    }
}

- (void)doReadTimeout
{
    // This is a little bit tricky.
    // Ideally we'd like to synchronously query the delegate about a timeout extension.
    // But if we do so synchronously we risk a possible deadlock.
    // So instead we have to do so asynchronously, and callback to ourselves from within the delegate block.
    
    flags |= kReadsPaused;
    
    __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;

    if (delegateQueue && [theDelegate respondsToSelector:@selector(socket:shouldTimeoutReadWithTag:elapsed:bytesDone:)])
    {
        GCDAsyncReadPacket *theRead = currentRead;
        
        dispatch_async(delegateQueue, ^{ @autoreleasepool {
            
            NSTimeInterval timeoutExtension = 0.0;
            
            timeoutExtension = [theDelegate socket:self shouldTimeoutReadWithTag:theRead->tag
                                                                         elapsed:theRead->timeout
                                                                       bytesDone:theRead->bytesDone];
            
            dispatch_async(self->socketQueue, ^{ @autoreleasepool {
                
                [self doReadTimeoutWithExtension:timeoutExtension];
            }});
        }});
    }
    else
    {
        [self doReadTimeoutWithExtension:0.0];
    }
}

- (void)doReadTimeoutWithExtension:(NSTimeInterval)timeoutExtension
{
    if (currentRead)
    {
        if (timeoutExtension > 0.0)
        {
            currentRead->timeout += timeoutExtension;
            
            // Reschedule the timer
            dispatch_time_t tt = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeoutExtension * NSEC_PER_SEC));
            dispatch_source_set_timer(readTimer, tt, DISPATCH_TIME_FOREVER, 0);
            
            // Unpause reads, and continue
            flags &= ~kReadsPaused;
            [self doReadData];
        }
        else
        {
            LogVerbose(@"ReadTimeout");
            
            [self closeWithError:[self readTimeoutError]];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Writing
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (void)writeData:(NSData *)data withTimeout:(NSTimeInterval)timeout tag:(long)tag
{
    if ([data length] == 0) return;
    
    GCDAsyncWritePacket *packet = [[GCDAsyncWritePacket alloc] initWithData:data timeout:timeout tag:tag];
    
    dispatch_async(socketQueue, ^{ @autoreleasepool {
        
        LogTrace();
        
        if ((self->flags & kSocketStarted) && !(self->flags & kForbidReadsWrites))
        {
            [self->writeQueue addObject:packet];
            [self maybeDequeueWrite];
        }
    }});
    
    // Do not rely on the block being run in order to release the packet,
    // as the queue might get released without the block completing.
}

/**
 * Conditionally starts a new write.
 *
 * It is called when:
 * - a user requests a write
 * - after a write request has finished (to handle the next request)
 * - immediately after the socket opens to handle any pending requests
 *
 * This method also handles auto-disconnect post read/write completion.
**/
- (void)maybeDequeueWrite
{
    LogTrace();
    NSAssert(dispatch_get_specific(IsOnSocketQueueOrTargetQueueKey), @"Must be dispatched on socketQueue");
    
    
    // If we're not currently processing a write AND we have an available write stream
    if ((currentWrite == nil) && (flags & kConnected))
    {
        if ([writeQueue count] > 0)
        {
            // Dequeue the next object in the write queue
            currentWrite = [writeQueue objectAtIndex:0];
            [writeQueue removeObjectAtIndex:0];
            
            
//            if ([currentWrite isKindOfClass:[GCDAsyncSpecialPacket class]])
//            {
//                LogVerbose(@"Dequeued GCDAsyncSpecialPacket");
//
//                // Attempt to start TLS
//                flags |= kStartingWriteTLS;
//
//                // This method won't do anything unless both kStartingReadTLS and kStartingWriteTLS are set
//                [self maybeStartTLS];
//            }
//            else
            {
                LogVerbose(@"Dequeued GCDAsyncWritePacket");
                
                // Setup write timer (if needed)
                [self setupWriteTimerWithTimeout:currentWrite->timeout];
                
                // Immediately write, if possible
                [self doWriteData];
            }
        }
        else if (flags & kDisconnectAfterWrites)
        {
            if (flags & kDisconnectAfterReads)
            {
                if (([readQueue count] == 0) && (currentRead == nil))
                {
                    [self closeWithError:nil];
                }
            }
            else
            {
                [self closeWithError:nil];
            }
        }
    }
}

- (void)doWriteData
{
    LogTrace();
    
    // This method is called by the writeSource via the socketQueue
    
    if ((currentWrite == nil) || (flags & kWritesPaused))
    {
        LogVerbose(@"No currentWrite or kWritesPaused");
        
        // Unable to write at this time
        
        if ([self usingCFStreamForTLS])
        {
            // CFWriteStream only fires once when there is available data.
            // It won't fire again until we've invoked CFWriteStreamWrite.
        }
        else
        {
            // If the writeSource is firing, we need to pause it
            // or else it will continue to fire over and over again.
            
            if (flags & kSocketCanAcceptBytes)
            {
                [self suspendWriteSource];
            }
        }
        return;
    }
    
    if (!(flags & kSocketCanAcceptBytes))
    {
        LogVerbose(@"No space available to write...");
        
        // No space available to write.
        
        if (![self usingCFStreamForTLS])
        {
            // Need to wait for writeSource to fire and notify us of
            // available space in the socket's internal write buffer.
            
            [self resumeWriteSource];
        }
        return;
    }
    
    if (flags & kStartingWriteTLS)
    {
        LogVerbose(@"Waiting for SSL/TLS handshake to complete");
        
        // The writeQueue is waiting for SSL/TLS handshake to complete.
        
//        if (flags & kStartingReadTLS)
//        {
//            if ([self usingSecureTransportForTLS] && lastSSLHandshakeError == errSSLWouldBlock)
//            {
//                // We are in the process of a SSL Handshake.
//                // We were waiting for available space in the socket's internal OS buffer to continue writing.
//
//                [self ssl_continueSSLHandshake];
//            }
//        }
//        else
//        {
//            // We are still waiting for the readQueue to drain and start the SSL/TLS process.
//            // We now know we can write to the socket.
//
//            if (![self usingCFStreamForTLS])
//            {
//                // Suspend the write source or else it will continue to fire nonstop.
//
//                [self suspendWriteSource];
//            }
//        }
        
        return;
    }
    
    // Note: This method is not called if currentWrite is a GCDAsyncSpecialPacket (startTLS packet)
    
    BOOL waiting = NO;
    NSError *error = nil;
    size_t bytesWritten = 0;
    
//    if (flags & kSocketSecure)
//    {
//        if ([self usingCFStreamForTLS])
//        {
//            #if TARGET_OS_IPHONE
//
//            //
//            // Writing data using CFStream (over internal TLS)
//            //
//
//            const uint8_t *buffer = (const uint8_t *)[currentWrite->buffer bytes] + currentWrite->bytesDone;
//
//            NSUInteger bytesToWrite = [currentWrite->buffer length] - currentWrite->bytesDone;
//
//            if (bytesToWrite > SIZE_MAX) // NSUInteger may be bigger than size_t (write param 3)
//            {
//                bytesToWrite = SIZE_MAX;
//            }
//
//            CFIndex result = CFWriteStreamWrite(writeStream, buffer, (CFIndex)bytesToWrite);
//            LogVerbose(@"CFWriteStreamWrite(%lu) = %li", (unsigned long)bytesToWrite, result);
//
//            if (result < 0)
//            {
//                error = (__bridge_transfer NSError *)CFWriteStreamCopyError(writeStream);
//            }
//            else
//            {
//                bytesWritten = (size_t)result;
//
//                // We always set waiting to true in this scenario.
//                // CFStream may have altered our underlying socket to non-blocking.
//                // Thus if we attempt to write without a callback, we may end up blocking our queue.
//                waiting = YES;
//            }
//
//            #endif
//        }
//        else
//        {
//            // We're going to use the SSLWrite function.
//            //
//            // OSStatus SSLWrite(SSLContextRef context, const void *data, size_t dataLength, size_t *processed)
//            //
//            // Parameters:
//            // context     - An SSL session context reference.
//            // data        - A pointer to the buffer of data to write.
//            // dataLength  - The amount, in bytes, of data to write.
//            // processed   - On return, the length, in bytes, of the data actually written.
//            //
//            // It sounds pretty straight-forward,
//            // but there are a few caveats you should be aware of.
//            //
//            // The SSLWrite method operates in a non-obvious (and rather annoying) manner.
//            // According to the documentation:
//            //
//            //   Because you may configure the underlying connection to operate in a non-blocking manner,
//            //   a write operation might return errSSLWouldBlock, indicating that less data than requested
//            //   was actually transferred. In this case, you should repeat the call to SSLWrite until some
//            //   other result is returned.
//            //
//            // This sounds perfect, but when our SSLWriteFunction returns errSSLWouldBlock,
//            // then the SSLWrite method returns (with the proper errSSLWouldBlock return value),
//            // but it sets processed to dataLength !!
//            //
//            // In other words, if the SSLWrite function doesn't completely write all the data we tell it to,
//            // then it doesn't tell us how many bytes were actually written. So, for example, if we tell it to
//            // write 256 bytes then it might actually write 128 bytes, but then report 0 bytes written.
//            //
//            // You might be wondering:
//            // If the SSLWrite function doesn't tell us how many bytes were written,
//            // then how in the world are we supposed to update our parameters (buffer & bytesToWrite)
//            // for the next time we invoke SSLWrite?
//            //
//            // The answer is that SSLWrite cached all the data we told it to write,
//            // and it will push out that data next time we call SSLWrite.
//            // If we call SSLWrite with new data, it will push out the cached data first, and then the new data.
//            // If we call SSLWrite with empty data, then it will simply push out the cached data.
//            //
//            // For this purpose we're going to break large writes into a series of smaller writes.
//            // This allows us to report progress back to the delegate.
//
//            OSStatus result;
//
//            BOOL hasCachedDataToWrite = (sslWriteCachedLength > 0);
//            BOOL hasNewDataToWrite = YES;
//
//            if (hasCachedDataToWrite)
//            {
//                size_t processed = 0;
//
//                result = SSLWrite(sslContext, NULL, 0, &processed);
//
//                if (result == noErr)
//                {
//                    bytesWritten = sslWriteCachedLength;
//                    sslWriteCachedLength = 0;
//
//                    if ([currentWrite->buffer length] == (currentWrite->bytesDone + bytesWritten))
//                    {
//                        // We've written all data for the current write.
//                        hasNewDataToWrite = NO;
//                    }
//                }
//                else
//                {
//                    if (result == errSSLWouldBlock)
//                    {
//                        waiting = YES;
//                    }
//                    else
//                    {
//                        error = [self sslError:result];
//                    }
//
//                    // Can't write any new data since we were unable to write the cached data.
//                    hasNewDataToWrite = NO;
//                }
//            }
//
//            if (hasNewDataToWrite)
//            {
//                const uint8_t *buffer = (const uint8_t *)[currentWrite->buffer bytes]
//                                                        + currentWrite->bytesDone
//                                                        + bytesWritten;
//
//                NSUInteger bytesToWrite = [currentWrite->buffer length] - currentWrite->bytesDone - bytesWritten;
//
//                if (bytesToWrite > SIZE_MAX) // NSUInteger may be bigger than size_t (write param 3)
//                {
//                    bytesToWrite = SIZE_MAX;
//                }
//
//                size_t bytesRemaining = bytesToWrite;
//
//                BOOL keepLooping = YES;
//                while (keepLooping)
//                {
//                    const size_t sslMaxBytesToWrite = 32768;
//                    size_t sslBytesToWrite = MIN(bytesRemaining, sslMaxBytesToWrite);
//                    size_t sslBytesWritten = 0;
//
//                    result = SSLWrite(sslContext, buffer, sslBytesToWrite, &sslBytesWritten);
//
//                    if (result == noErr)
//                    {
//                        buffer += sslBytesWritten;
//                        bytesWritten += sslBytesWritten;
//                        bytesRemaining -= sslBytesWritten;
//
//                        keepLooping = (bytesRemaining > 0);
//                    }
//                    else
//                    {
//                        if (result == errSSLWouldBlock)
//                        {
//                            waiting = YES;
//                            sslWriteCachedLength = sslBytesToWrite;
//                        }
//                        else
//                        {
//                            error = [self sslError:result];
//                        }
//
//                        keepLooping = NO;
//                    }
//
//                } // while (keepLooping)
//
//            } // if (hasNewDataToWrite)
//        }
//    }
//    else
    {
        //
        // Writing data directly over raw socket
        //
        
        int socketFD = socket4FD;//(socket4FD != SOCKET_NULL) ? socket4FD : (socket6FD != SOCKET_NULL) ? socket6FD : socketUN;
        
        const uint8_t *buffer = (const uint8_t *)[currentWrite->buffer bytes] + currentWrite->bytesDone;
        
        NSUInteger bytesToWrite = [currentWrite->buffer length] - currentWrite->bytesDone;
        
        if (bytesToWrite > SIZE_MAX) // NSUInteger may be bigger than size_t (write param 3)
        {
            bytesToWrite = SIZE_MAX;
        }
        
        ssize_t result = write(socketFD, buffer, (size_t)bytesToWrite);
        LogVerbose(@"wrote to socket = %zd", result);
        
        // Check results
        if (result < 0)
        {
            if (errno == EWOULDBLOCK)
            {
                waiting = YES;
            }
            else
            {
                error = [self errorWithErrno:errno reason:@"Error in write() function"];
            }
        }
        else
        {
            bytesWritten = result;
        }
    }
    
    // We're done with our writing.
    // If we explictly ran into a situation where the socket told us there was no room in the buffer,
    // then we immediately resume listening for notifications.
    //
    // We must do this before we dequeue another write,
    // as that may in turn invoke this method again.
    //
    // Note that if CFStream is involved, it may have maliciously put our socket in blocking mode.
    
    if (waiting)
    {
        flags &= ~kSocketCanAcceptBytes;
        
        if (![self usingCFStreamForTLS])
        {
            [self resumeWriteSource];
        }
    }
    
    // Check our results
    
    BOOL done = NO;
    
    if (bytesWritten > 0)
    {
        // Update total amount read for the current write
        currentWrite->bytesDone += bytesWritten;
        LogVerbose(@"currentWrite->bytesDone = %lu", (unsigned long)currentWrite->bytesDone);
        
        // Is packet done?
        done = (currentWrite->bytesDone == [currentWrite->buffer length]);
    }
    
    if (done)
    {
        [self completeCurrentWrite];
        
        if (!error)
        {
            dispatch_async(socketQueue, ^{ @autoreleasepool{
                
                [self maybeDequeueWrite];
            }});
        }
    }
    else
    {
        // We were unable to finish writing the data,
        // so we're waiting for another callback to notify us of available space in the lower-level output buffer.
        
        if (!waiting && !error)
        {
            // This would be the case if our write was able to accept some data, but not all of it.
            
            flags &= ~kSocketCanAcceptBytes;
            
            if (![self usingCFStreamForTLS])
            {
                [self resumeWriteSource];
            }
        }
        
        if (bytesWritten > 0)
        {
            // We're not done with the entire write, but we have written some bytes
            
            __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;

            if (delegateQueue && [theDelegate respondsToSelector:@selector(socket:didWritePartialDataOfLength:tag:)])
            {
                long theWriteTag = currentWrite->tag;
                
                dispatch_async(delegateQueue, ^{ @autoreleasepool {
                    
                    [theDelegate socket:self didWritePartialDataOfLength:bytesWritten tag:theWriteTag];
                }});
            }
        }
    }
    
    // Check for errors
    
    if (error)
    {
        [self closeWithError:[self errorWithErrno:errno reason:@"Error in write() function"]];
    }
    
    // Do not add any code here without first adding a return statement in the error case above.
}

- (void)completeCurrentWrite
{
    LogTrace();
    
    NSAssert(currentWrite, @"Trying to complete current write when there is no current write.");
    

    __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;
    
    if (delegateQueue && [theDelegate respondsToSelector:@selector(socket:didWriteDataWithTag:)])
    {
        long theWriteTag = currentWrite->tag;
        
        dispatch_async(delegateQueue, ^{ @autoreleasepool {
            
            [theDelegate socket:self didWriteDataWithTag:theWriteTag];
        }});
    }
    
    [self endCurrentWrite];
}

- (void)endCurrentWrite
{
    if (writeTimer)
    {
        dispatch_source_cancel(writeTimer);
        writeTimer = NULL;
    }
    
    currentWrite = nil;
}

- (void)setupWriteTimerWithTimeout:(NSTimeInterval)timeout
{
    if (timeout >= 0.0)
    {
        writeTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, socketQueue);
        
        __weak iOSAsyncSocket *weakSelf = self;
        
        dispatch_source_set_event_handler(writeTimer, ^{ @autoreleasepool {
        #pragma clang diagnostic push
        #pragma clang diagnostic warning "-Wimplicit-retain-self"
            
            __strong iOSAsyncSocket *strongSelf = weakSelf;
            if (strongSelf == nil) return_from_block;
            
            [strongSelf doWriteTimeout];
            
        #pragma clang diagnostic pop
        }});
        
        #if !OS_OBJECT_USE_OBJC
        dispatch_source_t theWriteTimer = writeTimer;
        dispatch_source_set_cancel_handler(writeTimer, ^{
        #pragma clang diagnostic push
        #pragma clang diagnostic warning "-Wimplicit-retain-self"
            
            LogVerbose(@"dispatch_release(writeTimer)");
            dispatch_release(theWriteTimer);
            
        #pragma clang diagnostic pop
        });
        #endif
        
        dispatch_time_t tt = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC));
        
        dispatch_source_set_timer(writeTimer, tt, DISPATCH_TIME_FOREVER, 0);
        dispatch_resume(writeTimer);
    }
}

- (void)doWriteTimeout
{
    // This is a little bit tricky.
    // Ideally we'd like to synchronously query the delegate about a timeout extension.
    // But if we do so synchronously we risk a possible deadlock.
    // So instead we have to do so asynchronously, and callback to ourselves from within the delegate block.
    
    flags |= kWritesPaused;
    
    __strong id<iOSAsyncSocketDelegate> theDelegate = delegate;

    if (delegateQueue && [theDelegate respondsToSelector:@selector(socket:shouldTimeoutWriteWithTag:elapsed:bytesDone:)])
    {
        GCDAsyncWritePacket *theWrite = currentWrite;
        
        dispatch_async(delegateQueue, ^{ @autoreleasepool {
            
            NSTimeInterval timeoutExtension = 0.0;
            
            timeoutExtension = [theDelegate socket:self shouldTimeoutWriteWithTag:theWrite->tag
                                                                          elapsed:theWrite->timeout
                                                                        bytesDone:theWrite->bytesDone];
            
            dispatch_async(self->socketQueue, ^{ @autoreleasepool {
                
                [self doWriteTimeoutWithExtension:timeoutExtension];
            }});
        }});
    }
    else
    {
        [self doWriteTimeoutWithExtension:0.0];
    }
}

- (void)doWriteTimeoutWithExtension:(NSTimeInterval)timeoutExtension
{
    if (currentWrite)
    {
        if (timeoutExtension > 0.0)
        {
            currentWrite->timeout += timeoutExtension;
            
            // Reschedule the timer
            dispatch_time_t tt = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeoutExtension * NSEC_PER_SEC));
            dispatch_source_set_timer(writeTimer, tt, DISPATCH_TIME_FOREVER, 0);
            
            // Unpause writes, and continue
            flags &= ~kWritesPaused;
            [self doWriteData];
        }
        else
        {
            LogVerbose(@"WriteTimeout");
            
            [self closeWithError:[self writeTimeoutError]];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Diagnostics
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (BOOL)isDisconnected
{
    __block BOOL result = NO;
    
    dispatch_block_t block = ^{
        result = (self->flags & kSocketStarted) ? NO : YES;
    };
    
    if (dispatch_get_specific(IsOnSocketQueueOrTargetQueueKey))
        block();
    else
        dispatch_sync(socketQueue, block);
    
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark Errors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

- (NSError *)badConfigError:(NSString *)errMsg
{
    NSDictionary *userInfo = @{NSLocalizedDescriptionKey : errMsg};
    
    return [NSError errorWithDomain:iOSAsyncSocketErrorDomain code:iOSAsyncSocketBadConfigError userInfo:userInfo];
}

- (NSError *)badParamError:(NSString *)errMsg
{
    NSDictionary *userInfo = @{NSLocalizedDescriptionKey : errMsg};
    
    return [NSError errorWithDomain:iOSAsyncSocketErrorDomain code:iOSAsyncSocketBadParamError userInfo:userInfo];
}

- (NSError *)errorWithErrno:(int)err reason:(NSString *)reason
{
    NSString *errMsg = [NSString stringWithUTF8String:strerror(err)];
    NSDictionary *userInfo = @{NSLocalizedDescriptionKey : errMsg,
                               NSLocalizedFailureReasonErrorKey : reason};
    
    return [NSError errorWithDomain:NSPOSIXErrorDomain code:err userInfo:userInfo];
}

/**
 * Returns a standard AsyncSocket maxed out error.
**/
- (NSError *)readMaxedOutError
{
    NSString *errMsg = NSLocalizedStringWithDefaultValue(@"iOSAsyncSocketReadMaxedOutError",
                                                         @"iOSAsyncSocket", [NSBundle mainBundle],
                                                         @"Read operation reached set maximum length", nil);
    
    NSDictionary *info = @{NSLocalizedDescriptionKey : errMsg};
    
    return [NSError errorWithDomain:iOSAsyncSocketErrorDomain code:iOSAsyncSocketReadMaxedOutError userInfo:info];
}

/**
 * Returns a standard AsyncSocket write timeout error.
**/
- (NSError *)readTimeoutError
{
    NSString *errMsg = NSLocalizedStringWithDefaultValue(@"iOSAsyncSocketReadTimeoutError",
                                                         @"iOSAsyncSocket", [NSBundle mainBundle],
                                                         @"Read operation timed out", nil);
    
    NSDictionary *userInfo = @{NSLocalizedDescriptionKey : errMsg};
    
    return [NSError errorWithDomain:iOSAsyncSocketErrorDomain code:iOSAsyncSocketReadTimeoutError userInfo:userInfo];
}

/**
 * Returns a standard AsyncSocket write timeout error.
**/
- (NSError *)writeTimeoutError
{
    NSString *errMsg = NSLocalizedStringWithDefaultValue(@"iOSAsyncSocketWriteTimeoutError",
                                                         @"iOSAsyncSocket", [NSBundle mainBundle],
                                                         @"Write operation timed out", nil);
    
    NSDictionary *userInfo = @{NSLocalizedDescriptionKey : errMsg};
    
    return [NSError errorWithDomain:iOSAsyncSocketErrorDomain code:iOSAsyncSocketWriteTimeoutError userInfo:userInfo];
}

- (NSError *)connectionClosedError
{
    NSString *errMsg = NSLocalizedStringWithDefaultValue(@"iOSAsyncSocketClosedError",
                                                         @"iOSAsyncSocket", [NSBundle mainBundle],
                                                         @"Socket closed by remote peer", nil);
    
    NSDictionary *userInfo = @{NSLocalizedDescriptionKey : errMsg};
    
    return [NSError errorWithDomain:iOSAsyncSocketErrorDomain code:iOSAsyncSocketClosedError userInfo:userInfo];
}


@end
