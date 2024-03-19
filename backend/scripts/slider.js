var minslider = document.getElementById("minRange");
var minoutput = document.getElementById("min");
var maxslider = document.getElementById("maxRange");
var maxoutput = document.getElementById("max");
minoutput.innerHTML = minslider.value;
maxoutput.innerHTML = maxslider.value;

// Update the current slider value (each time you drag the slider handle)
minslider.oninput = function () {
  minoutput.innerHTML = this.value;
}

maxslider.oninput = function () {
  maxoutput.innerHTML = this.value;
}