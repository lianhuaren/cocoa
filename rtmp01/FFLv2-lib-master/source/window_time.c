#include <windows.h>
#include<time.h>
/*
 * gStarted：是否初始化了
 * gFreq   ：cpu频率参数
 * gStartUs：初始化的时候的初始值
 */
static int gStarted=0;
static LARGE_INTEGER gFreq;
static LARGE_INTEGER gStartUs;
static time_t gStartTime;

static void internalTimeInit()
{
	if (gStarted == 0) 
	{
		QueryPerformanceFrequency(&gFreq);
		QueryPerformanceCounter(&gStartUs);
		gStarted = 1;
		time(&gStartTime);
	}
}

static int64_t internalGetUs()
{
	LARGE_INTEGER now;
	double dt=0;
	internalTimeInit();	
	QueryPerformanceCounter(&now);

	dt = (now.QuadPart - gStartUs.QuadPart) / (double)gFreq.QuadPart;
	return (int64_t)(dt * 1000000);
}

static void internalGetTimeString(int64_t current,char* s,int32_t bufSize) {
	time_t ts = gStartTime +(current / 1000000);
	int tus=current % 1000000;

	struct tm t = {0};
	localtime_s(&t,&ts);
//#ifdef FFL_TIME_USE_SHORT_STRING
//	sprintf(s,
//		"%02d-%02d%02d%02d:%03d:%03d",
//		t.tm_mday,
//		t.tm_hour,
//		t.tm_min,
//		t.tm_sec,
//		tus / 1000, tus % 1000);
//#else
	sprintf_s(s, bufSize - 1,
		"%4d%02d%02d%02d%02d%02d:%03d:%03d",
		t.tm_year + 1900,
		t.tm_mon + 1,
		t.tm_mday,
		t.tm_hour,
		t.tm_min,
		t.tm_sec,
		tus / 1000, tus % 1000);
//#endif
	
}


