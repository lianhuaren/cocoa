// uvw01.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

//#include <iostream>
//#include <uvw.hpp>
//int main()
//{
//    std::cout << "Hello World!\n";
//}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

#include <uvw.hpp>
//#include "uvw/src/uvw.hpp"
#include <memory>
#include "HttpServer.h"
//
//void listen(uvw::Loop& loop) {
//    //std::shared_ptr<uvw::TcpHandle> handleSrv;
//    //handleSrv = loop.resource<uvw::TcpHandle>();
//    //handleSrv->bind("0.0.0.0", port);
//    //handleSrv->listen();
//
//
//    std::shared_ptr<uvw::TcpHandle> tcp = loop.resource<uvw::TcpHandle>();
//
//    tcp->once<uvw::ListenEvent>([](const uvw::ListenEvent&, uvw::TcpHandle& srv) {
//        std::shared_ptr<uvw::TcpHandle> client = srv.loop().resource<uvw::TcpHandle>();
//
//        client->on<uvw::CloseEvent>([ptr = srv.shared_from_this()](const uvw::CloseEvent&, uvw::TcpHandle&) { ptr->close(); });
//        client->on<uvw::EndEvent>([](const uvw::EndEvent&, uvw::TcpHandle& client) { client.close(); });
//
//        srv.accept(*client);
//        client->read();
//        });
//
//    tcp->bind("127.0.0.1", 4242);
//    tcp->listen();
//}
//
//void conn(uvw::Loop& loop) {
//    auto tcp = loop.resource<uvw::TcpHandle>();
//
//    tcp->on<uvw::ErrorEvent>([](const uvw::ErrorEvent&, uvw::TcpHandle&) { /* handle errors */ });
//
//    tcp->once<uvw::ConnectEvent>([](const uvw::ConnectEvent&, uvw::TcpHandle& tcp) {
//        auto dataWrite = std::unique_ptr<char[]>(new char[2]{ 'b', 'c' });
//        tcp.write(std::move(dataWrite), 2);
//        tcp.close();
//        });
//
//    tcp->connect(std::string{ "127.0.0.1" }, 4242);
//}
//
//int main() {
//    auto loop = uvw::Loop::getDefault();
//    listen(*loop);
//    conn(*loop);
//    loop->run();
//}

#include "HttpServer.h"
int main() {
    auto loop = uvw::Loop::getDefault();
    
    HttpServer::Instance(8800);

    loop->run();
}
