//检查浏览器是否支持WebSocket
if (window.WebSocket) {
    console.log('This browser supports WebSocket');
} else {
    console.log('This browser does not supports WebSocket');
}

document.cookie = "userName"+"="+"111";

var strCookies = document.cookie;
console.log("cookie:"+strCookies)

/*--正文--*/
var count = 0;

/*WebSocket*/
var url = 'ws://localhost:8080/marco2';
var sock = new WebSocket(url);


sock.onopen = function (ev) {
    console.log("正在建立连接...");
    sayMarco();
};

sock.onmessage = function (ev) {
    console.log("接收并处理消息：" + ev.data);
    if (count == 10) {
        sock.close();
    }
    setTimeout(
        function () {
            sayMarco();
            count++;
        }, 2000);
};

sock.onclose = function (ev) {
    console.log("连接关闭...");
};

function sayMarco() {
    console.log('Sending Marco !');
    sock.send("Marco!")
}
