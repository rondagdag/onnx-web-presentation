# onnx-web-presentation

Click link to view 
[Presentation Material](ONNX-web-presentation.pdf)

---
## Quick Start - Nodejs Binding

This example is a demonstration of basic usage of ONNX Runtime Node.js binding.

This example contains `package.json` file, which lists "onnxruntime-node" as dependency. To work on your own `package.json`, use command `npm install onnxruntime-node` to install ONNX Runtime Node.js binding.

In this example, we load onnxruntime, load an image, create an inference session with the model we generated from a .net notebook, feed input, get output as result and display to standard console. All functions are called in their basic form.

### Usage

```sh
cd node
npm install
node .
```
---
## Quick Start - Web Browser Binding

This example is a demonstration of basic usage of ONNX Runtime Web binding.

This example contains `package.json` file, uses live-server as dependency. The index.html contains link to ONNX Runtime (ort) library https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js

In this example, we load onnxruntime on browser, load model locally, create an inference session that uses FER+ Emotion Recognition ONNX model , load image, convert to feed input properly, get output as result and display. 

### Usage

```sh
cd browser
npm install
npx light-server -s . -p 8081
```

go to http://localhost:8081/

load an image with face and emotion

the result will be displayed



