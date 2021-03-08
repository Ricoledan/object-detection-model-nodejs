const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const modelUrl =
  'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';
let model;

// load COCO-SSD graph model from TensorFlow Hub
const loadModel = async function () {
  console.log(`loading model from ${modelUrl}`);
  model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });

  return model;
};

// run
loadModel().then((model) => {
  console.log(model);
});

// convert image to Tensor
const processInput = function () {
  console.log(`preprocessing image ${imagePath}`);
  const image = fs.readFileSync(imagePath);
  const buf = Buffer.from(image);
  const Uint8Array = new Uint8Array(buf);

  return tf.node.decodeImage(Uint8Array, 3).expandDims();
};
