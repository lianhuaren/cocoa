/*
 * @Author: kael 
 * @Date: 2018-12-20 13:53:53 
 * @Last Modified by:   kael 
 * @Last Modified time: 2018-12-20 13:53:53 
 */

// https://zhuanlan.zhihu.com/p/37350346

let http = require("http");
const hostname = "127.0.0.1"; // 或者是localhost
const port = "8800";

class BitBuffer {
  // 构造函数传一个Buffer对象
  constructor(buffer) {
    this.buffer = buffer;
  }
  // 获取第offset个位的内容
  _getBit(offset) {
    let byteIndex = offset / 8 >> 0,
      byteOffset = offset % 8;
    // readUInt8可以读取第n个字节的数据
    // 取出这个数的第m位即可
    let num = this.buffer.readUInt8(byteIndex) & (1 << (7 - byteOffset));
    return num >> (7 - byteOffset);
  }

  getBit(offset, len = 1) {
    let result = 0;
    for (let i = 0; i < len; i++) {
      result += this._getBit(offset + i) << (len - i - 1);
    }
    return result;
  }

  getMaskingKey(offset) {
    const BYTE_COUNT = 4;
    let masks = [];
    for (let i = 0; i < BYTE_COUNT; i++) {
      masks.push(this.getBit(offset + i * 8, 8));
    }
    return masks;
  }

  getXorString(byteOffset, byteCount, maskingKeys) {
    let text = '';
    for (let i = 0; i < byteCount; i++) {
      let j = i % 4;
      // 通过异或得到原始的utf-8编码
      let transformedByte = this.buffer.readUInt8(byteOffset + i) ^ maskingKeys[j];
      // 把编码值转成对应的字符
      text += String.fromCharCode(transformedByte);
    }
    return text;
  }
}

// 创建一个http服务
let server = http.createServer((req, res) => {
  // 收到请求
  console.log("recv request");
  console.log(req.headers);
  // 进行响应，发送数据
  // res.write('hello, world');
  // res.end();
});

// 开始监听
server.listen(port, hostname, () => {
  // 启动成功
  console.log(`Server running at ${hostname}:${port}`);
});

const crypto = require('crypto');

function sha1(data) {
  let generator = crypto.createHash('sha1');
  generator.update(data)
  return generator.digest('hex')
}

// 协议升级
server.on("upgrade", (request, socket, head) => {
  // console.log('kael', { request, socket, head });
  // 取出浏览器发送的key值
  let secKey = request.headers['sec-websocket-key'];
  console.log('secKey', secKey);
  // RFC 6455规定的全局标志符(GUID)
  const UNIQUE_IDENTIFIER = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
  // 计算sha1和base64值
  let shaValue = sha1(secKey + UNIQUE_IDENTIFIER);
  let base64Value = Buffer.from(shaValue, 'hex').toString('base64');
  console.log('secApt', base64Value);
  socket.write('HTTP/1.1 101 Web Socket Protocol Handshake\r\n' +
    'Upgrade: WebSocket\r\n' +
    'Connection: Upgrade\r\n' +
    `Sec-WebSocket-Accept: ${base64Value}\r\n` +
    '\r\n');

  socket.on('data', buffer => {
    console.log('buffer len = ', buffer.length);
    let bitBuffer = new BitBuffer(buffer);
    let maskFlag = bitBuffer.getBit(8);
    let opcode = bitBuffer.getBit(4, 4);
    let payloadLen = bitBuffer.getBit(9, 7);
    console.log('maskFlag = ' + maskFlag);
    console.log('opcode = ' + opcode);
    console.log('payloadLen = ' + payloadLen);
    let maskKeys = bitBuffer.getMaskingKey(16);
    console.log('maskKey = ' + maskKeys);
    let payloadText = bitBuffer.getXorString(48 / 8, payloadLen, maskKeys);
    console.log('payloadText = ' + payloadText);
  });
});