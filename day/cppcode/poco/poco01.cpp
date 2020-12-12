1.poco

poco只能用c++模板技术,acceptor的回调只能new ServiceHandler(socket, *_pReactor)来实现。
而且绑定类成员函数，需要使用一个模板参数，可读可写又要用一个模板参数

NObserver<EchoServiceHandler, ReadableNotification>(*this, &EchoServiceHandler::onSocketReadable)


			// set-up a server socket
			ServerSocket svs(port);
			// set-up a SocketReactor...
			SocketReactor reactor;
			// ... and a SocketAcceptor
			SocketAcceptor<EchoServiceHandler> acceptor(svs, reactor);

	EchoServiceHandler(StreamSocket& socket, SocketReactor& reactor):
		_socket(socket),
		_reactor(reactor),
		_fifoIn(BUFFER_SIZE, true),
		_fifoOut(BUFFER_SIZE, true)
	{
		Application& app = Application::instance();
		app.logger().information("Connection from " + socket.peerAddress().toString());

		_reactor.addEventHandler(_socket, NObserver<EchoServiceHandler, ReadableNotification>(*this, &EchoServiceHandler::onSocketReadable));
		_reactor.addEventHandler(_socket, NObserver<EchoServiceHandler, ShutdownNotification>(*this, &EchoServiceHandler::onSocketShutdown));

		_fifoOut.readable += delegate(this, &EchoServiceHandler::onFIFOOutReadable);
		_fifoIn.writable += delegate(this, &EchoServiceHandler::onFIFOInWritable);
	}

	virtual ServiceHandler* createServiceHandler(StreamSocket& socket)
		/// Create and initialize a new ServiceHandler instance.
		///
		/// Subclasses can override this method.
	{
		return new ServiceHandler(socket, *_pReactor);
	}

void SocketNotifier::addObserver(SocketReactor* pReactor, const Poco::AbstractObserver& observer)
{
	_nc.addObserver(observer);
	if (observer.accepts(pReactor->_pReadableNotification))
		_events.insert(pReactor->_pReadableNotification.get());
	else if (observer.accepts(pReactor->_pWritableNotification))
		_events.insert(pReactor->_pWritableNotification.get());
	else if (observer.accepts(pReactor->_pErrorNotification))
		_events.insert(pReactor->_pErrorNotification.get());
	else if (observer.accepts(pReactor->_pTimeoutNotification))
		_events.insert(pReactor->_pTimeoutNotification.get());
}

template <class C, class N>
class Observer: public AbstractObserver

	bool accepts(Notification* pNf) const
	{
		return dynamic_cast<N*>(pNf) != 0;
	}



2.muduo

boost::function又能绑定普通函数，又能绑定类成员函数。


    EventLoop loop;

    TcpServer server(&loop, listenAddr, "PingPong");

    server.setConnectionCallback(onConnection);
    server.setMessageCallback(onMessage);


void onConnection(const TcpConnectionPtr& conn)
{
  if (conn->connected())
  {
    conn->setTcpNoDelay(true);
  }
}

void onMessage(const TcpConnectionPtr& conn, Buffer* buf, Timestamp)
{
  conn->send(buf);
}



TcpConnection::TcpConnection(EventLoop* loop,
                             const string& nameArg,
                             int sockfd,
                             const InetAddress& localAddr,
                             const InetAddress& peerAddr)
  : loop_(CHECK_NOTNULL(loop)),
    name_(nameArg),
    state_(kConnecting),
    socket_(new Socket(sockfd)),
    channel_(new Channel(loop, sockfd)),
    localAddr_(localAddr),
    peerAddr_(peerAddr),
    highWaterMark_(64*1024*1024)
{
  channel_->setReadCallback(
      boost::bind(&TcpConnection::handleRead, this, _1));
  channel_->setWriteCallback(
      boost::bind(&TcpConnection::handleWrite, this));
  channel_->setCloseCallback(
      boost::bind(&TcpConnection::handleClose, this));
  channel_->setErrorCallback(
      boost::bind(&TcpConnection::handleError, this));
  LOG_DEBUG << "TcpConnection::ctor[" <<  name_ << "] at " << this
            << " fd=" << sockfd;
  socket_->setKeepAlive(true);
}

class Channel : boost::noncopyable
{
 public:
  typedef boost::function<void()> EventCallback;
  typedef boost::function<void(Timestamp)> ReadEventCallback;

  Channel(EventLoop* loop, int fd);
  ~Channel();

  void handleEvent(Timestamp receiveTime);
  void setReadCallback(const ReadEventCallback& cb)
  { readCallback_ = cb; }
  void setWriteCallback(const EventCallback& cb)
  { writeCallback_ = cb; }
  void setCloseCallback(const EventCallback& cb)
  { closeCallback_ = cb; }
  void setErrorCallback(const EventCallback& cb)
  { errorCallback_ = cb; }
