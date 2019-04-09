#include <FFL_Lib.h>

class LeakObject {
public:
	LeakObject() {
		mId = 0;
	}

	int mId;
};

static void testLeak2() {
	FFL_mallocz(10);
}

static void testLeak() {
	new LeakObject();
	testLeak2();
}

//
//  编译FFL库的时候需要打开宏
//  FFL_config.h
//  #define  CHECK_FOR_MEMORY_LEAKS 1
//
int FFL_main() {
	char exePath[1024] = {};
	char exeName[1024] = {};
	FFL_getCurrentProcessPath(exePath, 1023, exeName);

	FFL::String path(exePath);
	//path = exePath;
	path += "memoryLeank.log";

	if (0) {
		//
		//  需要需要检测，则开启这个函数，当有上次标记的泄漏的时候会触发assert中断
		//
		FFL_checkMemoryLeak(path.string());
	}


	testLeak();
	FFL_dumpMemoryLeakFile(path.string());
	return 0;
}
