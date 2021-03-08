const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const modelUrl =
  'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';
let model;

// load COCO-SSD graph model from TensorFlow Hub
const loadModel = async function () {
  console.log(`loading model from ${modelUrl}`);
  model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });

  return model;
};

// convert image to Tensor
const processInput = function (imgName) {
  console.log(`preprocessing image ${imgName}`);
  const img = fs.readFileSync(path.join('assets', imgName));
  const buf = Buffer.from(img);
  const uint8Array = new Uint8Array(buf);

  return tf.node.decodeImage(uint8Array, 3).expandDims();
};

// run
if (process.argv.length < 3) {
  console.log('please supply an image to the process');
  console.log('node run-tfjs-model.js image.jpg');
} else {
  let imgName = process.argv[2];

  loadModel().then((model) => {
    const inputTensor = processInput(imgName);
    inputTensor.print();
  });
}
