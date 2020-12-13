;(function(window, $, html2canvas){

var getMouseOffset;
(function() {
    var elementOffset = function(elem) {
        var x, y;
        x = 0;
        y = 0;
        while (true) {
            x += elem.offsetLeft;
            y += elem.offsetTop;
            if (!(elem = elem.offsetParent)) {
                break;
            }
        }
        return {
            x: x,
            y: y
        };
    };
    //event是jquery中的event对象 原生event有兼容问题
    getMouseOffset = function(elem, event) {
        var offset = elementOffset(elem);
        return {
            left: event.pageX - offset.x,
            top: event.pageY - offset.y
        };
    };
})();
var drag = function(selector, option) {
    var $els = $(selector),
        dragging = option.dragging || $.noop,
        dragend = option.dragend || $.noop,
        dragcontinue = option.dragcontinue || $.noop,
        dragpause = option.dragpause || $.noop,
        dragstart = option.dragstart || $.noop;
    $els.each(function(){
        var mouse = $.proxy(function(e){
            return getMouseOffset(this, e);
        } ,this);
        var $this = $(this);
        var mousestate = {
            down: false
        };
        $this.mousedown(function(e){
            if (mousestate.down == true) {
                //如果鼠标在离开绘图区域后弹起，再次进入绘图区域，不进行重新计算，而是继续之前的绘图
                return false;
            } else {
                mousestate.down = true;
                mousestate.downOffset = mouse(e);
                dragstart.call(this, mousestate.downOffset);
            }

        });
        $this.mouseup(function(e){
            mousestate.down = false;
            if(mousestate.move == true){
                dragend.call(this, mousestate.downOffset, mouse(e));
                mousestate.move = false;
            }
        });
        $this.mouseout(function (e) {
            if(!mousestate.down) return;
            dragpause.call(this, mouse(e));
            mousestate.out = true;
        });
        var move = function(e){
            if(!mousestate.down) return;
            if(mousestate.out === true) {
                //如果鼠标在离开绘图区域后弹起，再次进入绘图区域，不进行重新计算，而是继续之前的绘图
                dragcontinue.call(this, mouse(e));
            }
            mousestate.out = false;
            mousestate.move = true;
            dragging.call(this, mousestate.downOffset, mouse(e));
        }
        $this.mousemove(move);
    });
};
var util = {};
(function() {
    //两点间的距离
    util.len = function (start, end) {
        var w = end.left-start.left
        var h = end.top-start.top;
        return Math.sqrt(Math.pow(w, 2) + Math.pow(h, 2));
    };
    util.throttle = function(func, wait, options) {
        var context, args, result;
        var timeout = null;
        var previous = 0;
        options || (options = {});
        var later = function() {
            previous = options.leading === false ? 0 : _.now();
            timeout = null;
            result = func.apply(context, args);
            context = args = null;
        };
        return function() {
            var now = util.now();
            if (!previous && options.leading === false) previous = now;
            var remaining = wait - (now - previous);
            context = this;
            args = arguments;
            if (remaining <= 0) {
                clearTimeout(timeout);
                timeout = null;
                previous = now;
                result = func.apply(context, args);
                context = args = null;
            } else if (!timeout && options.trailing !== false) {
                timeout = setTimeout(later, remaining);
            }
            return result;
        };
    };
    util.uuid = function uuid() {
        function S4() {
            return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
        }
        return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
    };    
    util.now = Date.now || function() { return new Date().getTime(); };
})();
var line = function(ctx1, ctx2, that) {
    var canvas1 = ctx1.canvas,
        canvas2 = ctx2.canvas;
    return {
        ing: function (start, end) {
            ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
            ctx2.beginPath();
            ctx2.moveTo(start.left, start.top);
            ctx2.lineTo(end.left, end.top);
            ctx2.stroke();
        },
        drawMy: function (message) {
            var start = message.start;
            var end = message.end;
            var color = message.color;
            var lineWidth = message.lineWidth;
            {
                ctx1.save();
                ctx1.strokeStyle = color;
                ctx1.lineWidth = lineWidth;
                ctx1.beginPath();
                ctx1.moveTo(start.left, start.top);
                ctx1.lineTo(end.left, end.top);
                ctx1.stroke();
                ctx1.restore();             
            }
        },
        onMessage: function (message) {
            this.drawMy(message);
        },
        end: function (start, end) {
            ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
            var color = ctx1.strokeStyle;
            var lineWidth = ctx1.lineWidth;
            // var draw = function () {
            //     ctx1.save();
            //     ctx1.strokeStyle = color;
            //     ctx1.lineWidth = lineWidth;
            //     ctx1.beginPath();
            //     ctx1.moveTo(start.left, start.top);
            //     ctx1.lineTo(end.left, end.top);
            //     ctx1.stroke();
            //     ctx1.restore();             
            // }
            // draw();
            // that.stack.push(draw);
            // that.rstack = [];
            {
                var msg = {};
                msg.uuid = util.uuid();
                msg.type = 'line';
                msg.color = color;
                msg.lineWidth = lineWidth;
                msg.start = start;
                msg.end = end;

                this.drawMy(msg);
                that.stack.push(msg);
                that.rstack = [];
                sendMessageDraw(msg);
            }               
        }
    }
};
var curve = function(ctx1, ctx2, that) {
    var canvas1 = ctx1.canvas,
        canvas2 = ctx2.canvas;
    var points = [];
    var index=0;
    return {
        start: function (position) {
            ctx2.beginPath();
            ctx2.lineJoin = 'round';
            ctx2.moveTo(position.left, position.top);
            // points.push(position);
            points.push([position.left, position.top]);
        },
        ing: function (start, end) {
            ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
            ctx2.lineTo(end.left, end.top);
            if (index++%2 == 0) 
            {
                // points.push(end);
                points.push([end.left, end.top]);
            };
            
            ctx2.stroke();
        },
        drawMy: function (message) {
            var points = message.points;
            var stackPoints = points.slice(0);
            var color = message.color;
            var lineWidth = message.lineWidth;
            {
                ctx1.save();
                ctx1.strokeStyle = color;
                ctx1.lineWidth = lineWidth;
                ctx1.beginPath();
                $.each(stackPoints, function(i, p){
                    if (i===0) {
                        ctx1.moveTo(p[0], p[1]);
                        // ctx1.moveTo(p.left, p.top);
                    } else {
                        // ctx1.lineTo(p.left, p.top);
                        ctx1.lineTo(p[0], p[1]);
                    }
                });
                ctx1.stroke();
                ctx1.restore();
            }
        },
        onMessage: function (message) {
            this.drawMy(message);
        },        
        end: function () {
            ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
            //回退或前进功能中重绘时会重新读取该数组，而points是动态变化的，所以拷贝一份出来
            var stackPoints = points.slice(0);
            var color = ctx1.strokeStyle;
            var lineWidth = ctx1.lineWidth;
            // var draw = function () {
            //     ctx1.save();
            //     ctx1.strokeStyle = color;
            //     ctx1.lineWidth = lineWidth;
            //     ctx1.beginPath();
            //     $.each(stackPoints, function(i, p){
            //         if (i===0) {
            //             ctx1.moveTo(p.left, p.top);
            //         } else {
            //             ctx1.lineTo(p.left, p.top);
            //         }
            //     });
            //     ctx1.stroke();
            //     ctx1.restore();
            // }
            // draw();
            // that.stack.push(draw);
            // that.rstack = [];
            // points = [];
            {
                var msg = {};
                msg.uuid = util.uuid();
                msg.type = 'curve';
                msg.color = color;
                msg.lineWidth = lineWidth;
                msg.points = points;

                this.drawMy(msg);
                that.stack.push(msg);
                that.rstack = [];
                points = [];
                sendMessageDraw(msg);
            }    
        }
    };
};
var round;
(function() {
    var len = util.len;
    round = function(ctx1, ctx2, that) {
        var canvas1 = ctx1.canvas,
            canvas2 = ctx2.canvas;
        return {
            ing: function (start, end) {
                ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
                ctx2.beginPath();
                ctx2.arc( start.left + (end.left-start.left)/2, start.top + (end.top-start.top)/2, len(start, end)/2, 0, 2*Math.PI );
                ctx2.stroke();
            },
            drawMy: function (message) {
                var start = message.start;
                var end = message.end;
                var color = message.color;
                var lineWidth = message.lineWidth;
                {
                    ctx1.save();
                    ctx1.lineWidth = lineWidth;
                    ctx1.strokeStyle = color;
                    ctx1.beginPath();
                    ctx1.arc( start.left + (end.left-start.left)/2, start.top + (end.top-start.top)/2, len(start, end)/2, 0, 2*Math.PI );
                    ctx1.stroke();
                    ctx1.restore();
                }
            },
            onMessage: function (message) {
                this.drawMy(message);
            },

            end: function (start, end) {
                ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
                var color = ctx1.strokeStyle;
                var lineWidth = ctx1.lineWidth;
                // var draw = function () {
                //     ctx1.save();
                //     ctx1.lineWidth = lineWidth;
                //     ctx1.strokeStyle = color;
                //     ctx1.beginPath();
                //     ctx1.arc( start.left + (end.left-start.left)/2, start.top + (end.top-start.top)/2, len(start, end)/2, 0, 2*Math.PI );
                //     ctx1.stroke();
                //     ctx1.restore();
                // }
                // draw();
                // that.stack.push(draw);
                // that.rstack = [];
                {
                    var msg = {};
                    msg.uuid = util.uuid();
                    msg.type = 'round';
                    msg.color = color;
                    msg.lineWidth = lineWidth;
                    msg.start = start;
                    msg.end = end;

                    this.drawMy(msg);
                    that.stack.push(msg);
                    that.rstack = [];

                    sendMessageDraw(msg);
                }                  
            }
        }
    }
})();
var rect = function(ctx1, ctx2, that) {
    var canvas1 = ctx1.canvas,
        canvas2 = ctx2.canvas;
    return {
        ing: function (start, end) {
            ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
            ctx2.strokeRect(start.left, start.top, end.left - start.left, end.top - start.top);
        },
        drawMy: function (message) {
            var start = message.start;
            var end = message.end;
            var color = message.color;
            var lineWidth = message.lineWidth;
            {
                ctx1.save();
                ctx1.lineWidth = lineWidth;
                ctx1.strokeStyle = color;
                ctx1.strokeRect(start.left, start.top, end.left - start.left, end.top - start.top);
                ctx1.restore();
            }
        },
        onMessage: function (message) {
            this.drawMy(message);
        },
        end: function (start, end) {
            ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
            var color = ctx1.strokeStyle;
            var lineWidth = ctx1.lineWidth;
            // var draw = function () {
            //     ctx1.save();
            //     ctx1.lineWidth = lineWidth;
            //     ctx1.strokeStyle = color;
            //     ctx1.strokeRect(start.left, start.top, end.left - start.left, end.top - start.top);
            //     ctx1.restore();
            // }
            // draw();
            // that.stack.push(draw);
            // that.rstack = [];
                {
                    var msg = {};
                    msg.uuid = util.uuid();
                    msg.type = 'rect';
                    msg.color = color;
                    msg.lineWidth = lineWidth;
                    msg.start = start;
                    msg.end = end;

                    this.drawMy(msg);
                    that.stack.push(msg);
                    that.rstack = [];

                    sendMessageDraw(msg);
                }   
        }
    }
};
var ease = function(ctx1, ctx2, that) {
    var canvas1 = ctx1.canvas,
        canvas2 = ctx2.canvas,
        easeFn = [];
    return {
        ing: function (start, end) {
            function c2(){
                ctx2.clearRect(0,0,canvas2.width,canvas2.height);
                ctx2.beginPath();
                ctx2.arc(Math.floor(end.left), Math.floor(end.top), 10, 0, 2*Math.PI);
                ctx2.stroke();
            }
            c2 = util.throttle(c2, 140);
            c2();
            var draw = function () {
                ctx1.globalCompositeOperation = "destination-out";  //鼠标覆盖区域不显示
                ctx1.beginPath();
                ctx1.arc(Math.floor(end.left), Math.floor(end.top), 10, 0, 2*Math.PI, true);
                ctx1.closePath();
                ctx1.fill();
                ctx1.globalCompositeOperation = "source-over";
            }
            draw();
            easeFn.push(draw);
        },
        end: function (start, end) {
            that.ctx2.clearRect(0,0,that.canvas2.width,that.canvas2.height);
            var stackFn = easeFn.slice(0);
            that.stack.push(function () {
                $.each(stackFn, function () {
                    this();
                });
            });
            that.rstack = [];
            that.easeFn = [];
        }
    };
}
var arrow = function(ctx1, ctx2, that) {
    var arrow = function(ctx2, start, end) {
        ctx2.save()
        ctx2.beginPath()
        ctx2.moveTo(start.left, start.top)
        ctx2.lineTo(end.left, end.top)
        ctx2.fillStyle = ctx2.strokeStyle

        var atan = Math.atan((end.top - start.top)/(end.left - start.left));
        var rotate = Math.PI/2 + atan
        if (end.left < start.left) {
            rotate =  rotate - Math.PI
        }

        ctx2.translate(end.left, end.top)
        ctx2.rotate(rotate)

        var angle = 30/180*Math.PI/2
        var width = 15 * Math.tan(angle)
        ctx2.moveTo(0, 0)
        ctx2.lineTo(width, 15)
        ctx2.lineTo(-width, 15)

        ctx2.stroke()
        ctx2.fill()
        ctx2.restore()
    }
    return {
        start: function (position) {

        },
        ing: function (start, end) {
            ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height)
            arrow(ctx2, start, end)
        },
        drawMy: function (message) {
            var start = message.start;
            var end = message.end;
            var color = message.color;
            var lineWidth = message.lineWidth;
            {
                ctx1.strokeStyle = color
                arrow(ctx1, start, end)
            }
        },
        onMessage: function (message) {
            this.drawMy(message);
        },        
        end: function (start, end) {
            ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height)
            var color = ctx1.strokeStyle;
            var lineWidth = ctx1.lineWidth;
            // var color = that.color
            // var draw = function() {
            //     ctx1.strokeStyle = color
            //     arrow(ctx1, start, end)
            // }
            // draw()
            // that.stack.push(draw)
            // that.rstack = [];
            {
                var msg = {};
                msg.uuid = util.uuid();
                msg.type = 'arrow';
                msg.color = color;
                msg.lineWidth = lineWidth;
                msg.start = start;
                msg.end = end;

                this.drawMy(msg);
                that.stack.push(msg);
                that.rstack = [];

                sendMessageDraw(msg);
            }  
        }
    }
};
var DrawBoard;
(function() {
    var ident = function() {
        return false;
    };
    /**
     * 为canvas添加绘图功能
     * @param canvas1和canvas2是两个重叠的canvas标签 canvas2在canvas1上面
     * 对于ie，不支持canvas，canvas1和canvas2是excanvas初始化的canvas对象
     */
    DrawBoard = function(option) {
        var that = this;
        this.option = option || {};
        this.type = option.type || 'rect';
        this.lineWidth = option.lineWidth || 1;
        this.color = option.color || 'rgb(0, 0, 0)'
        this.stack = []; this.rstack = [];
        var canvas1 = this.canvas1 = document.createElement('canvas'), $canvas1 = this.$canvas1 = $(canvas1),
            canvas2 = this.canvas2 = document.createElement('canvas'), $canvas2 = this.$canvas2 = $(canvas2),
            $canvases = $canvas1.add($canvas2),
            $con = $('<div></div>').append($canvas1, $canvas2);
        $con.css({ width: option.width, height: option.height, position: 'relative' });
        canvas1.width = canvas2.width = option.width;
        canvas1.height = canvas2.height = option.height;
        $canvases.css({ position: 'absolute', left: 0, top: 0 });
        $con.appendTo(option.parent);
        if (!canvas1.getContext) {
            if(window.G_vmlCanvasManager){
                canvas1=window.G_vmlCanvasManager.initElement(canvas1);
                canvas2=window.G_vmlCanvasManager.initElement(canvas2);
            } else {
                alert('对不起，您的浏览器不支持canvas!');
            }
        }
        var ctx1 = this.ctx1 = canvas1.getContext('2d');
        var ctx2 = this.ctx2 = canvas2.getContext('2d');
        var option = option || {
            clearBt: null, //清除按钮
            saveBt : null //保存按钮
        };
        ctx1.save();
        ctx2.strokeStyle = this.color;
        ctx1.strokeStyle = this.color;
        ctx2.lineWidth = this.lineWidth;
        ctx1.lineWidth = this.lineWidth;
        var mouse = {};
        $(canvas2).mousemove(function(e){
            mouse = getMouseOffset($(canvas2).get(0), e);
        });
        var draw = {};
        this.draw = draw;
        draw.rect = rect(ctx1, ctx2, that);
        draw.round = round(ctx1, ctx2, that);
        draw.line = line(ctx1, ctx2, that);
        draw.ease = ease(ctx1, ctx2, that);
        draw.curve = curve(ctx1, ctx2, that);

        ctx1.canvas = canvas1;
        ctx2.canvas = canvas2;
        draw.arrow = arrow(ctx1, ctx2, that);
        drag(canvas2, {
            dragstart: function (position) {
                that.refuseSelection();
                (draw[that.type].start || $.noop)(position);
            },
            dragcontinue: function (position) {
                that.refuseSelection();
                (draw[that.type]["continue"] || $.noop)(position);
            },
            dragpause: function (position) {
                that.allowSelection();
                (draw[that.type].pause || $.noop)(position);
            },
            dragging: function (start, end) {
                draw[that.type].ing(start, end);
            },
            dragend: function (start, end) {
                draw[that.type].end(start, end);
                that.allowSelection();
            }
        });
    };
    DrawBoard.prototype = {
        onMessage: function (message) {
            console.log("drawboard onMessage:" + message);
            if (message.type == "delMessage") {
                this.onDelMessage(message)
            } else if (message.type == "clearMessage") {
                this.onClearMessage(message)
            } else {
                this.draw[message.type].onMessage(message);
                this.stack.push(message);
            }
            
        },        
        drawDel: function (message) {
            this.ctx1.clearRect(0, 0, this.canvas1.width, this.canvas1.height);
            var pop = this.stack.pop();
            if (pop) {
                this.rstack.push(pop);
            }
            this.drawStack();            
        }, 
        onDelMessage: function (message) {
            this.drawDel(message);        
        },         
        //撤消
        back: function () {
            this.ctx1.clearRect(0, 0, this.canvas1.width, this.canvas1.height);
            var pop = this.stack.slice(-1);
            // var pop = this.stack.pop();
            // if (pop) {
            //     this.rstack.push(pop);
            // }
            // this.drawStack();
            if (pop) {
                var msg = {};
                msg.uuid = pop.uuid;
                msg.type = 'delMessage';


                this.drawDel(msg);

                sendMessageDraw(msg);
            }  
        },
        //重做
        forward: function () {
            this.ctx1.clearRect(0, 0, this.canvas1.width, this.canvas1.height);
            var pop = this.rstack.pop();
            if (pop) {
                this.stack.push(pop);

                sendMessageDraw(pop);
            }
            this.drawStack();
        },
        //重绘
        drawStack: function () {
            var that = this;
            $.each(this.stack, function () {
                // this && this();
                var message = this;
                that.draw[message.type].onMessage(message);
            });
        },
        setColor: function (color) {
            this.color = this.ctx1.strokeStyle = this.ctx2.strokeStyle = color;
        },
        setLineWidth: function (lineWidth) {
            this.lineWidth = this.ctx1.lineWidth = this.ctx2.lineWidth = lineWidth;
        },
        refuseSelection: function () {
            $('body').attr('unselectable','on').css({
                '-webkit-user-select': 'none', /* Chrome all / Safari all */
                '-moz-user-select': 'none',     /* Firefox all */
                '-ms-user-select': 'none',      /* IE 10+ */
                /* No support for these yet, use at own risk */
                '-o-user-select': 'none',
                'user-select': 'none'
            }).bind('selectstart', ident);
            this.clearSelection();
        },
        allowSelection: function () {
            $('body').attr('unselectable', 'off').css({
                '-webkit-user-select': 'auto', /* Chrome all / Safari all */
                '-moz-user-select': 'auto',     /* Firefox all */
                '-ms-user-select': 'auto',      /* IE 10+ */
                /* No support for these yet, use at own risk */
                '-o-user-select': 'auto',
                'user-select': 'auto'
            }).unbind('selectstart', ident);
        },
        clearSelection: function () {
            if (document.selection) {
                document.selection.empty();
            } else if (window.getSelection) {
                window.getSelection().removeAllRanges();
            }
        },
        drawClear: function (message) {
            var that = this,
            canvas2 = this.canvas2;
            {
                that.ctx1.clearRect(0, 0, canvas2.width, canvas2.height);
            }
            that.stack = [];     
        },            
        onClearMessage: function (message) {
            this.drawClear(message);        
        },    
        clear: function () {
            var that = this,
                canvas2 = this.canvas2;
            // var draw = function () {
            //     that.ctx1.clearRect(0, 0, canvas2.width, canvas2.height);
            //     that.rstack = [];
            // }
            // draw();
            // this.stack.push(draw);
            {
                var msg = {};
                msg.uuid = util.uuid;
                msg.type = 'clearMessage';


                this.drawClear(msg);

                that.rstack = [];
                sendMessageDraw(msg);
            } 
        },
        save: function(el) {
            if (!window.getComputedStyle) { alert('您的浏览器不支持'); return; }
            if (!window.html2canvas) {
                alert('没有引入 html2canvas.js ，不支持保存绘图功能');
            }
            var data;
            html2canvas($(el)[0] || $(this.option.parent)[0], {
                onrendered: function (canvas) {
                    var data = canvas.toDataURL('image/jpeg');
                    var w = window.open();
                    $(w.document.body).append('<img src="'+data+'" />');
                    w.document.title = '保存绘图';
                }
            });
        }
    }
})();
window.DrawBoard = DrawBoard;
})(window, jQuery, html2canvas);