
#include <FFL_Netlib.hpp>
#include <net/websocket/FFL_WebSocketServer.hpp>

static void myExit(const char* args, void* userdata) {
	printf("cmd:myExit\n");

	FFL::ConsoleLoop* console =(FFL::ConsoleLoop*) userdata;
	FFL::WebSocketServer* webServer=(FFL::WebSocketServer*)console->getUserdata();
	webServer->stop();
	console->stop();
}

static CmdOption  myCmdOption[] = {	
	{ "exit",0,myExit,"exit process" },
	{ "quit",0,myExit,"exit process" },
	{ 0,0,0,0 }
};


int FFL_main() {	
	FFL_socketInit();
	
	FFL::WebSocketServer webServer("127.0.0.1",8800,NULL);	
	webServer.start(new FFL::ModuleThread("websocket"));

	FFL::ConsoleLoop console;
	console.setUserdata(&webServer);
	console.registeCommand(myCmdOption, 2);
	console.start(NULL);
	console.dumpUsage();
	console.eventLoop(NULL);	

	return 0;
}
