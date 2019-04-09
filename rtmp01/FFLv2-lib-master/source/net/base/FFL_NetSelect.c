#include "internalSocket.h"
#include <net/FFL_Net.h>
#include "internalLogConfig.h"


int32_t FFL_socketSelect(const NetFD *fdList, int8_t *flagList, size_t fdNum, int64_t timeoutUs) {
    struct timeval tv;
    NetFD maxfd = 64;
    fd_set fdset;
    size_t i = 0;	
    int status = 0;
    int socketError = 0;
    
    if (timeoutUs > 0) {
        tv.tv_sec = (long)(timeoutUs / (1000 * 1000));
        tv.tv_usec = (long)(timeoutUs % (1000 * 1000));
    }
    else {
        tv.tv_sec = -1;
        tv.tv_usec = -1;
    }
    
#if WIN32
    if (fdNum > 64) {
        return FFL_ERROR_SOCKET_SELECT;
    }
#else
    maxfd = 0;
    for (i = 0; i < fdNum; i++) {
        if (fdList[i] > maxfd) {
            maxfd=fdList[i];
        }
    }
    maxfd+=1;
#endif
    
    FD_ZERO(&fdset);
    for ( i = 0; i < fdNum; i++) {
        FD_SET(fdList[i], &fdset);
        flagList[i] = 0;
    }
    
    status = select(maxfd, &fdset, 0, 0, (timeoutUs == 0 ? NULL : (&tv)));
    if (status < 0) {
        INTERNAL_FFL_LOG_WARNING("FFL_socketSelect error=%d",SOCKET_ERRNO());
#if WIN32
        return FFL_ERROR_SOCKET_SELECT;
#else
        socketError = SOCKET_ERRNO();
        if (socketError == EINTR) {
            //
            //  当做超时处理，可能其他信号触发了这
            //
            return 0;
        }
#endif
    }
    
    if (status > 0) {
        for ( i = 0; i < fdNum; i++) {
            if (FD_ISSET(fdList[i], &fdset))
                flagList[i] = 1;
        }
    }
    return status;
}


/************************************************************************
*  pull模式                                                                   
************************************************************************/

/*
int32_t FFL_socketPoll(const NetFD *fdList, int8_t *flagList, size_t fdNum, int64_t timeoutUs) {
	struct pollfd fds[64];
	for (size_t i = 0; i < fdNum; i++) {
		fds[i].fd = fdList[i];
		fds[i].events = POLLIN;
		fds[i].revents = 0;
		flagList[i] = 0;
	}

	int timeoutmsec = -1;
	if (timeoutUs >= 0) {
		timeoutmsec = (int)(timeoutUs / 1000);
	}

	
	int status = poll(&(fds[0]), fdNum, timeoutmsec);
	if (status < 0) {
		if (errno == EINTR)
			return 0;
		return -1;
	}

	if (status > 0) {
		for (size_t i = 0; i < fdNum; i++) {
			if (fds[i].revents)
				flagList[i] = 1;
		}
	}
	return status;
}
*/