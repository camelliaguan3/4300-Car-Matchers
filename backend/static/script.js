var w = window.innerWidth;
var h = window.innerHeight;

const maxW = 1920;
const maxH = 1080;


console.log('hello');
console.log(w);
console.log(h);


for (let e of document.getElementsByClassName("moving-car-image")) {
    e.style.display = "none";
        e.style.visibility = "hidden";
}
