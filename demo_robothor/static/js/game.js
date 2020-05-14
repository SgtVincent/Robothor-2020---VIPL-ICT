// Create the canvas

var canvas = document.createElement("canvas");
var ctx = canvas.getContext("2d");
canvas.width = 640;
canvas.height = 480;
document.body.appendChild(canvas);


// Handle keyboard controls
var keysDown = {};

addEventListener("keydown", function (e) {
    keysDown[e.keyCode] = true;
}, false);

addEventListener("keyup", function (e) {
    delete keysDown[e.keyCode];
}, false);

var bgImage = new Image();
bgImage.src = "";

// Update game objects
var update = function (modifier) {
    if (38 in keysDown) { // Player holding up
        // hero.y -= hero.speed * modifier;
        var action = 'MoveAhead';
        postData(action);
    }
    if (40 in keysDown) { // Player holding down
        // hero.y += hero.speed * modifier;
        var action = 'MoveBack';
        postData(action);
    }
    if (37 in keysDown) { // Player holding left
        // hero.x -= hero.speed * modifier;
        var action = 'RotateLeft';
        postData(action);
    }
    if (39 in keysDown) { // Player holding right
        // hero.x += hero.speed * modifier;
        var action = 'RotateRight';
        postData(action);
    }

    if (87 in keysDown) { // Player holding right
        // hero.x += hero.speed * modifier;
        var action = 'LookUp';
        postData(action);
    }
    if (83 in keysDown) { // Player holding right
        // hero.x += hero.speed * modifier;
        var action = 'LookDown';
        postData(action);
    }

};

// Draw everything
var render = function () {
    // Score
    ctx.drawImage(bgImage, 0, 0);
    ctx.fillStyle = "rgb(250, 250, 250)";
    ctx.font = "24px Helvetica";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("RoboTHOR: ", 32, 32);
};

// The main game loop
var main = function () {
    var now = Date.now();
    var delta = now - then;

    update(delta / 1000);
    render();

    then = now;

    // Request to do this again ASAP
    requestAnimationFrame(main);
};

function postData(input) {
    $.ajax({
        type: "POST",
        url: "/takeaction",
        data: {action: input},
        success: callbackFunc,
        async: false
    });
}

function callbackFunc(response) {
    var text;
    var res =JSON.parse(response);
    // bgImage.src,  text= response;
    // bgImage.src = res.ImageBytes;
    bgImage.src = 'data:image/png;base64,' + res.ImageBytes;
    // console.log(response);
    document.getElementById("robo_state").textContent=res.robo_state;
    bgImage.setAttribute(
        'src', res.ImageBytes
    );
}


// Cross-browser support for requestAnimationFrame
var w = window;
requestAnimationFrame = w.requestAnimationFrame || w.webkitRequestAnimationFrame || w.msRequestAnimationFrame || w.mozRequestAnimationFrame;

// Let's play this game!
var then = Date.now();
var action = 'GetCurrentFrame';
postData(action);
main();
