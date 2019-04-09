#if ANDROID
#include <time.h>
#else
#include <sys/time.h>
#endif
static int gStarted=0;
static struct timeval gStartUs;
static time_t gStartTime;


static void internalTimeInit()
{
	if (gStarted == 0) 
	{
		gettimeofday(&gStartUs, NULL);
		gStarted = 1;
        time(&gStartTime);
	}
}

static int64_t gLastNowus=0;
int64_t internalGetUs(){
	struct timeval now;
	internalTimeInit();

	int64_t nowUs =gLastNowus;
	if(gettimeofday(&now, NULL)==0) {
		int64_t a= (int64_t) now.tv_sec * 1000 * 1000 +now.tv_usec;
		int64_t b= (int64_t) gStartUs.tv_sec * 1000 * 1000 +gStartUs.tv_usec;
		nowUs = a-b;
		gLastNowus=nowUs;
	}else{

	}
	return nowUs;
}

void internalGetTimeString(int64_t current,char* s, int32_t bufSize) {
	time_t ts = gStartTime + (current / 1000000);
	struct tm* t = 0;
	t = localtime(&ts);
    int tus = current % 1000000;
//#ifdef FFL_TIME_USE_SHORT_STRING
//	sprintf(s,
//		"%02d%02d%02d:%03d:%03d",
//		t->tm_hour,
//		t->tm_min,
//		t->tm_sec,
//		tus / 1000, tus % 1000);
//#else
	sprintf(s,
		"%4d%02d%02d%02d%02d%02d:%03d:%03d",
		t->tm_year + 1900,
		t->tm_mon + 1,
		t->tm_mday,
		t->tm_hour,
		t->tm_min,
		t->tm_sec,
		tus / 1000, tus % 1000);
//#endif
}
