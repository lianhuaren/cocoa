'use strict';

var isChannelReady = false;
var isInitiator = false;
var isStarted = false;
var localStream;
var pc;
var remoteStream;
var turnReady;

var pcConfig = {
  'iceServers': [{
    'urls': 'stun:stun.l.google.com:19302'
  }]
};

// Set up audio and video regardless of what devices are present.
var sdpConstraints = {
  offerToReceiveAudio: true,
  offerToReceiveVideo: true
};

/////////////////////////////////////////////

var room = 'foo';
// Could prompt for room name:
// room = prompt('Enter room name:');

// var socket = io.connect();

//检查浏览器是否支持WebSocket
if (window.WebSocket) {
    console.log('This browser supports WebSocket');
} else {
    console.log('This browser does not supports WebSocket');
}

/*WebSocket*/
var url = 'ws://localhost:8080/marco2';
var sock = new WebSocket(url);
var myUserId = 0;

sock.onopen = function (ev) {
    console.log("正在建立连接...");
    
    var msg = {};
    msg.type = 'create or join';
    msg.content = room;
    msg.group = 0;

    sendMessageCus(msg);
};

sock.onmessage = function (ev) {
    console.log("接收并处理消息：" + ev.data);
    var msg = JSON.parse(ev.data) || {};

    if (msg.group == 0) {
      var content = msg.content;

      if (msg.type == 'created') {
         
         myUserId = content.userId;
         console.log('created: ' + room + ', userId: ' + content.userId);

         isInitiator = true;
      } else if (msg.type == 'full') {
          console.log('Room ' + room + ' is full');
      } else if (msg.type == 'join') {
          console.log('Another peer made a request to join room ' + room);
          console.log('This peer is the initiator of room ' + room + '!');
          isInitiator = true;
          isChannelReady = true;
      } else if (msg.type == 'joined') {
        myUserId = content.userId;

        console.log('joined: ' + room + ', userId: ' + content.userId);
        isChannelReady = true;
        
      } else if (msg.type == 'userLeave') {
        console.log('on userLeave: ' + msg.content);
        if (isStarted) {
          handleRemoteHangup();
        }
      }
    } else if(msg.group ==1) {
      if (msg.type == 'message') {
        var message = msg.content;
        console.log('Client received message:', message);
        if (message === 'got user media') {
          maybeStart();
        } else if (message === 'bye') {
          if (isStarted) {
            handleRemoteHangup();
          };
        } else {
            message = JSON.parse(message);
            if (message.type === 'offer') {
              if (!isInitiator && !isStarted) {
                maybeStart();
              }
              pc.setRemoteDescription(new RTCSessionDescription(message));
              doAnswer();
            } else if (message.type === 'answer' && isStarted) {
              pc.setRemoteDescription(new RTCSessionDescription(message));
            } else if (message.type === 'candidate' && isStarted) {
              var candidate = new RTCIceCandidate({
                sdpMLineIndex: message.label,
                candidate: message.candidate
              });
              pc.addIceCandidate(candidate);
            }
        }
      }
      else if (msg.type == 'draw') {
        var content = msg.content;
        panel.onMessage(content);

      }

    }


};

sock.onclose = function (ev) {
    console.log("连接关闭...");
};

function sayMarco() {
    console.log('Sending Marco !');
    sock.send("Marco!")
}


if (room !== '') {
  // socket.emit('create or join', room);
  console.log('Attempted to create or  join room', room);
}

// socket.on('created', function(room) {
//   console.log('Created room ' + room);
//   isInitiator = true;
// });

// socket.on('full', function(room) {
//   console.log('Room ' + room + ' is full');
// });

//   console.log('This peer is the initiator of room ' + room + '!');
// socket.on('join', function (room){
//   console.log('Another peer made a request to join room ' + room);
//   isChannelReady = true;
// });

// socket.on('joined', function(room) {
//   console.log('joined: ' + room);
//   isChannelReady = true;
// });

// socket.on('log', function(array) {
//   console.log.apply(console, array);
// });

////////////////////////////////////////////////
function sendMessageCus(message) {
  console.log('Client sending message: ', JSON.stringify(message));

  sock.send(JSON.stringify(message));
}

function sendMessage(message) {
  // sendMessageInner('message',1, message);
  var msg = {};
  msg.type = 'message';//
  msg.group = 1;
  msg.content = message;
  if (typeof(message) === 'object') {
    //message是客户自定义群发的，所以都发送string
    msg.content = JSON.stringify(message);
  } 

  sendMessageInner(msg);
}

function sendMessageDraw(message) {
  // sendMessageInner('draw',1, message);
  message.userid = myUserId;

  var msg = {};
  msg.type = 'draw';//
  msg.group = 1;
  msg.content = message;

  sendMessageInner(msg);
}

function sendMessageInner(msg) {
  console.log('Client sending message: ', msg);
  // socket.emit('message', message);

  // var msg = {};
  // msg.type = type;//
  // msg.group = group;
  // if (typeof(msg.content) === 'object') {
  //   msg.content = JSON.stringify(msg.content);
  // } 
  // else {
  //   msg.content = message;
  // }
  var message = JSON.stringify(msg);
  // if (message.length > 1024) 
  {
    console.log("====message length:"+message.length);
  };
  sock.send(message);
}


// // This client receives a message
// socket.on('message', function(message) {
//   console.log('Client received message:', message);
//   if (message === 'got user media') {
//     maybeStart();
//   } else if (message.type === 'offer') {
//     if (!isInitiator && !isStarted) {
//       maybeStart();
//     }
//     pc.setRemoteDescription(new RTCSessionDescription(message));
//     doAnswer();
//   } else if (message.type === 'answer' && isStarted) {
//     pc.setRemoteDescription(new RTCSessionDescription(message));
//   } else if (message.type === 'candidate' && isStarted) {
//     var candidate = new RTCIceCandidate({
//       sdpMLineIndex: message.label,
//       candidate: message.candidate
//     });
//     pc.addIceCandidate(candidate);
//   } else if (message === 'bye' && isStarted) {
//     handleRemoteHangup();
//   }
// });

////////////////////////////////////////////////////

var localVideo = document.querySelector('#localVideo');
var remoteVideo = document.querySelector('#remoteVideo');

navigator.mediaDevices.getUserMedia({
  audio: false,
  video: true
})
.then(gotStream)
.catch(function(e) {
  alert('getUserMedia() error: ' + e.name);
});

function gotStream(stream) {
  console.log('Adding local stream.');
  localStream = stream;
  localVideo.srcObject = stream;
  sendMessage('got user media');

    // var msg = {};
    // msg.type = 'message';
    // msg.content = 'got user media';
    // msg.group = 1;

    // sendMessageCus(msg);

  if (isInitiator) {
    maybeStart();
  }
}

var constraints = {
  video: true
};

console.log('Getting user media with constraints', constraints);

if (location.hostname !== 'localhost') {
  requestTurn(
    'https://computeengineondemand.appspot.com/turn?username=41784574&key=4080218913'
  );
}

function maybeStart() {
  console.log('>>>>>>> maybeStart() ', isStarted, localStream, isChannelReady);
  if (!isStarted && typeof localStream !== 'undefined' && isChannelReady) {
    console.log('>>>>>> creating peer connection');
    createPeerConnection();
    pc.addStream(localStream);
    isStarted = true;
    console.log('isInitiator', isInitiator);
    if (isInitiator) {
      doCall();
    }
  }
}

window.onbeforeunload = function() {
  sendMessage('bye');
  // var msg = {};
  // msg.type = 'message';
  // msg.content = 'bye';
  // msg.group = 1;

  // sendMessageCus(msg);
};


/////////////////////////////////////////////////////////

function createPeerConnection() {
  try {
    pc = new RTCPeerConnection(null);
    pc.onicecandidate = handleIceCandidate;
    pc.onaddstream = handleRemoteStreamAdded;
    pc.onremovestream = handleRemoteStreamRemoved;
    console.log('Created RTCPeerConnnection');
  } catch (e) {
    console.log('Failed to create PeerConnection, exception: ' + e.message);
    alert('Cannot create RTCPeerConnection object.');
    return;
  }
}

function handleIceCandidate(event) {
  console.log('icecandidate event: ', event);
  if (event.candidate) {
    sendMessage({
      type: 'candidate',
      label: event.candidate.sdpMLineIndex,
      id: event.candidate.sdpMid,
      candidate: event.candidate.candidate
    });

    // var msg = {};
    // msg.type = 'message';
    // msg.content = JSON.stringify({
    //   type: 'candidate',
    //   label: event.candidate.sdpMLineIndex,
    //   id: event.candidate.sdpMid,
    //   candidate: event.candidate.candidate
    // });
    // msg.group = 1;

    // sendMessageCus(msg);

  } else {
    console.log('End of candidates.');
  }
}

function handleCreateOfferError(event) {
  console.log('createOffer() error: ', event);
}

function doCall() {
  console.log('Sending offer to peer');
  pc.createOffer(setLocalAndSendMessage, handleCreateOfferError);
}

function doAnswer() {
  console.log('Sending answer to peer.');
  pc.createAnswer().then(
    setLocalAndSendMessage,
    onCreateSessionDescriptionError
  );
}

function setLocalAndSendMessage(sessionDescription) {
  pc.setLocalDescription(sessionDescription);
  console.log('setLocalAndSendMessage sending message', sessionDescription);
  sendMessage(sessionDescription);

    // var msg = {};
    // msg.type = 'message';
    // msg.content = JSON.stringify(sessionDescription);
    // msg.group = 1;

    // sendMessageCus(msg);
}

function onCreateSessionDescriptionError(error) {
  trace('Failed to create session description: ' + error.toString());
}

function requestTurn(turnURL) {
  var turnExists = false;
  for (var i in pcConfig.iceServers) {
    if (pcConfig.iceServers[i].urls.substr(0, 5) === 'turn:') {
      turnExists = true;
      turnReady = true;
      break;
    }
  }
  if (!turnExists) {
    console.log('Getting TURN server from ', turnURL);
    // No TURN server. Get one from computeengineondemand.appspot.com:
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4 && xhr.status === 200) {
        var turnServer = JSON.parse(xhr.responseText);
        console.log('Got TURN server: ', turnServer);
        pcConfig.iceServers.push({
          'urls': 'turn:' + turnServer.username + '@' + turnServer.turn,
          'credential': turnServer.password
        });
        turnReady = true;
      }
    };
    xhr.open('GET', turnURL, true);
    xhr.send();
  }
}

function handleRemoteStreamAdded(event) {
  console.log('Remote stream added.');
  remoteStream = event.stream;
  remoteVideo.srcObject = remoteStream;
}

function handleRemoteStreamRemoved(event) {
  console.log('Remote stream removed. Event: ', event);
}

function hangup() {
  console.log('Hanging up.');
  stop();
  sendMessage('bye');
  // var msg = {};
  // msg.type = 'message';
  // msg.content = 'bye';
  // msg.group = 1;

  // sendMessageCus(msg);
}

function handleRemoteHangup() {
  console.log('Session terminated.');
  stop();
  isInitiator = false;
}

function stop() {
  isStarted = false;
  pc.close();
  pc = null;
}
