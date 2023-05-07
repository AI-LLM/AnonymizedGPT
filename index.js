import { Tokenizer } from "huggingface-tokenizers-bindings";

//import * as wasmFeatureDetect from "wasm-feature-detect";

//Setup onnxruntime 
const ort = require('onnxruntime-web');

//requires Cross-Origin-*-policy headers https://web.dev/coop-coep/
/**
const simdResolver = wasmFeatureDetect.simd().then(simdSupported => {
    console.log("simd is supported? "+ simdSupported);
    if (simdSupported) {
      ort.env.wasm.numThreads = 3; 
      ort.env.wasm.simd = true;
    } else {
      ort.env.wasm.numThreads = 1; 
      ort.env.wasm.simd = false;
    }
});
*/

const options = {
  executionProviders: ['wasm'], 
  graphOptimizationLevel: 'all'
};

var downLoadingModel = true;
const model = "./bert_ner_ft.onnx";//"./bert_ner_int8.onnx";
let INPUT = "As a warmup surveyor, I want to be able to log into a warmup system";

async function loadConfig() {
  return fetch('./config.json').then(d => d.json());
}
const conf = loadConfig();
let id2label, zero_class;
conf.then(d => {
  const values = Object.values(d.id2label);
  if (values.includes("O")) {
    zero_class = parseInt(Object.keys(d.id2label).find(key => d.id2label[key] === "O"));
  } else {
    throw new Error("model config is incorrect");
  }      
  id2label = d.id2label;
  console.log(id2label, zero_class);
});


const session = ort.InferenceSession.create(model, options);
session.then(t => { 
  downLoadingModel = false;
  //warmup the VM
  for(var i = 0; i < 1; i++) {
    console.log("Inference warmup " + i);
    lm_inference(INPUT);
  }
});

function create_model_input(encoded) {
  var input_ids = new Array(encoded.length+2);
  var attention_mask = new Array(encoded.length+2);
  var token_type_ids = new Array(encoded.length+2);
  input_ids[0] = BigInt(101);
  attention_mask[0] = BigInt(1);
  token_type_ids[0] = BigInt(0);
  var i = 0;
  for(; i < encoded.length; i++) { 
    input_ids[i+1] = BigInt(encoded[i]);
    attention_mask[i+1] = BigInt(1);
    token_type_ids[i+1] = BigInt(0);
  }
  input_ids[i+1] = BigInt(102);
  attention_mask[i+1] = BigInt(1);
  token_type_ids[i+1] = BigInt(0);
  const sequence_length = input_ids.length;
  input_ids = new ort.Tensor('int64', BigInt64Array.from(input_ids), [1,sequence_length]);
  attention_mask = new ort.Tensor('int64', BigInt64Array.from(attention_mask), [1,sequence_length]);
  token_type_ids = new ort.Tensor('int64', BigInt64Array.from(token_type_ids), [1,sequence_length]);
  return {
    input_ids: input_ids,
    attention_mask: attention_mask,
    token_type_ids:token_type_ids
  }
}
function argmax(tensor){
  function maxIndex(arr) {
    let max = arr[0];
    let maxIndex = 0;
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;//[max, maxIndex];
  }
  let output = [];
  //TODO: handle tensor.dims[0]
  for (let j = 0 ; j < tensor.dims[1]; j++) {
    const s = j* tensor.dims[2];
    const e = s + tensor.dims[2]
    const arr = tensor.data.slice(s, e)
    const i = maxIndex(arr)
    output.push(i)
  }
  return output;
}
function softmax(tensor) {
  const dims = tensor.dims;
  const data = tensor.data;

  if (dims.length !== 3) {
    throw new Error('The tensor must be a 3-dimensional array');
  }

  const numPlanes = dims[0];
  const numRows = dims[1];
  const numCols = dims[2];

  // Initialize an empty 3-dimensional array with the same dimensions
  const softmaxData = Array.from({ length: numPlanes }, () =>
    Array.from({ length: numRows }, () => Array(numCols))
  );

  for (let plane = 0; plane < numPlanes; plane++) {
    for (let row = 0; row < numRows; row++) {
      let max = -Infinity;
      let sum = 0;

      // Find the maximum value in the row
      for (let col = 0; col < numCols; col++) {
        const value = data[plane * numRows * numCols + row * numCols + col];
        if (value > max) {
          max = value;
        }
      }

      // Compute the exponentials and their sum
      for (let col = 0; col < numCols; col++) {
        const exp = Math.exp(data[plane * numRows * numCols + row * numCols + col] - max);
        sum += exp;
        softmaxData[plane][row][col] = exp;
      }

      // Normalize the exponentials to get the probabilities
      for (let col = 0; col < numCols; col++) {
        softmaxData[plane][row][col] /= sum;
      }
    }
  }

  return softmaxData;
}
function getLabeledWords(text, tokens, scores, labels, skippingLabel) {
  const importantWords = [];
  let currentWord = '';
  let currentLabel = null;
  let currentScore = 0;
  let startIndex = 0;
  let endIndex = 0;
  let lastIndex = 0;
  
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const label = labels[i];
    const score = scores[i];

    const subtokenMatch = token.match(/^#+(.*)$/);
    if (subtokenMatch) {
      currentWord += subtokenMatch[1];
      lastIndex = text.indexOf(subtokenMatch[1], lastIndex);
      lastIndex += subtokenMatch[1].length;
      endIndex = lastIndex;
      if (label !== skippingLabel) {
        currentLabel = label;
        currentScore = score;
      }
    }
    else{
      if (currentWord && currentLabel!==skippingLabel) {
        importantWords.push({
          word: currentWord,
          start: startIndex,
          end: endIndex,
          label: currentLabel,
          score: currentScore
        });
      }
      currentWord = token;
      startIndex = text.indexOf(token, lastIndex);
      endIndex = startIndex + token.length;
      lastIndex = endIndex;
      currentLabel = label;
      currentScore = score;
    }    
  }
  if (currentWord && currentLabel!==skippingLabel) {
    importantWords.push({
      word: currentWord,
      start: startIndex,
      end: endIndex,
      label: currentLabel,
      score: currentScore
    });
  }

  return importantWords;
}

async function lm_inference(text) {
  try { 
    let tokenizer = await Tokenizer.from_pretrained("bert-base-cased");
    const encoding = tokenizer.encode(text, false);//pub fn encode(&self, text: &str, add_special_tokens: bool) -> EncodingWasm {
    if(encoding.tokens.length === 0) {
      display(encoding,0.0, "[]");
    }
    const encoded_ids = encoding.input_ids;
    
    const start = performance.now();
    const model_input = create_model_input(encoded_ids);
    console.log(text, encoded_ids, model_input);
    //run_bert_ner_onnx.py's tokens: [1249, 170, 25646,...]
    let onames = (await session).outputNames;
    const output =  await session.then(s => { return s.run(model_input,onames/*['output_0']*/, { logSeverityLevel: 2 })});//if the output name is given it must match to the model, otherwise it does not return without any log
    const duration = (performance.now() - start).toFixed(1);
    //console.log(output);
    // Find the correct tensor in the model output
    let outputTensor;
    for (const key in output) {
      if (output[key].dims && output[key].dims.length === 3 && output[key].dims[0] === model_input.input_ids.dims[0] && output[key].dims[1] === model_input.input_ids.dims[1] && output[key].dims[2] === Object.keys(id2label).length) {
        outputTensor = output[key];
        break;
      }
    }
    console.log(outputTensor);
    /*get labels
        labels = np.argmax(prediction, axis=-1)
        labels = [
            labels[sentence, : sentence_lengths[sentence]]
            for sentence in range(labels.shape[0])
        ]*/
    let labels = argmax(outputTensor);
    labels = labels.slice(1,-1);//skip the prefix 101 and suffix 102 tokens
    //console.log(labels.map(idx=>id2label[idx]))
    /*get confidence score
        conf_scores = [
            self._softmax_(prediction[sentence, : sentence_lengths[sentence]])
            for sentence in range(prediction.shape[0])
        ]*/
    let scores = softmax(outputTensor);
    scores = scores[0].slice(1,-1);//TODO: handle tensor.dims[0]
    /*let tokenIdx=0;
    word_ids.forEach((tokenIDs, index)=>{
      tokenIDs.forEach(tokenId=>{
        const label = id2label[labels[tokenIdx]];
        const score = scores[tokenIdx][labels[tokenIdx]];
        console.log(text.substring(encoded_ids[tokenIdx].startOffset,encoded_ids[tokenIdx].endOffset), label, score)
        tokenIdx++;
      });
    });*/
    let ls = [],ss = [];
    for (let tokenIdx=0;tokenIdx<labels.length;tokenIdx++){
        const label = id2label[labels[tokenIdx]];
        const score = scores[tokenIdx][labels[tokenIdx]];
        console.log(encoding.tokens[tokenIdx], label, score);
        ls.push(label);
        ss.push(score);
    }

    let labeledWords = getLabeledWords(text, encoding.tokens, ss, ls, 'O');
    return display(encoding,duration,JSON.stringify(labeledWords));
  } catch (e) {
    console.error(e);//return display(encoding,0.0,e);
  }
}    

function display(encoding, duration, results) {
    document.getElementById("input").innerHTML = INPUT;
    document.getElementById("tokens").innerHTML =  JSON.stringify(encoding.tokens);
    document.getElementById("input_ids").innerHTML = "[" + encoding.input_ids + "]";
    document.getElementById("results").innerHTML = results;
    document.getElementById("duration").innerHTML = duration;
}
