
const WIDTH = 64;
const DIMS = [1, 1, WIDTH, WIDTH];
const MAX_LENGTH = DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3];
const MAX_SIGNED_VALUE = 255.0;

let predictedClass;
let isRunning = false;

const canvas = document.createElement("canvas"),
ctx = canvas.getContext("2d");

document.getElementById("file-in").onchange = function (evt) {
  let target = evt.target || window.event.src,
    files = target.files;

  if (FileReader && files && files.length) {
    isRunning = true;
    var fileReader = new FileReader();
    fileReader.onload = () => onLoadImage(fileReader);
    fileReader.readAsDataURL(files[0]);
  }
};

const target = document.getElementById("target");
window.setInterval(function() {
  if (isRunning) {
    target.innerHTML = `<h3>Loading...`;
  } else if (typeof predictedClass !== 'undefined') {
    target.innerHTML = `<h3>${predictedClass}!`;
  } else {
    target.innerHTML = ``;
  }
}, 500)


function onLoadImage(fileReader) {
  var img = document.getElementById("input-image");
  img.onload = () => handleImage(img, WIDTH);
  img.src = fileReader.result;
}

function handleImage(img, targetWidth) {
  ctx.drawImage(img, 0, 0);
  const resizedImageData = processImage(img, targetWidth);
  const inputTensor = imageDataToTensor(resizedImageData, DIMS);
  run(inputTensor);
}

function processImage(img, width) {
  const canvas = document.createElement("canvas"),
  ctx = canvas.getContext("2d");

  canvas.width = width;
  canvas.height = width;  
  ctx.drawImage(img, 0, 0, width, width); //canvas.width, canvas.height);
  
  const data = ctx.getImageData(0, 0, width, width).data
  const greyScale = [];
  for (let i = 0; i < data.length; i+= 4) {
    greyScale.push((data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114));
  }  
  document.getElementById("canvas-image").src = canvas.toDataURL();
  return greyScale;
}

function imageDataToTensor(data, dims) {
  l = data.length; // length, we need this for the loop
  const float32Data = new Float32Array(MAX_LENGTH); // create the Float32Array for output
  for (i = 0; i < l; i++) {
    float32Data[i] = data[i]; // / MAX_SIGNED_VALUE; // convert to float
  }

  // return ort.Tensor
  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}

function argMax(arr) {
  let max = arr[0];
  let maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }
  return [max, maxIndex];
}

async function run(inputTensor) {
  try {
    const session = await ort.InferenceSession.create('./emotion-ferplus-8.onnx');     
    const feeds = { Input3: inputTensor };

    // feed inputs and run
    const results = await session.run(feeds);
    console.log(results);

    const output = results.Plus692_Output_0;
    const classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger',
      'disgust', 'fear', 'contempt'];
    const [maxValue, maxIndex] = argMax(output.data);
    predictedClass = `${classes[maxIndex]}`;
    isRunning = false;
  } catch (e) {
    console.log(e);
  }
}