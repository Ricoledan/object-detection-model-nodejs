const tf = require('@tensorflow/tfjs-node');
const modelUrl =
  'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';
let model;

const loadModel = async function () {
  console.log(`loading model from ${modelUrl}`);
  model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });

  return model;
};

loadModel().then((model) => {
  console.log(model);
});
