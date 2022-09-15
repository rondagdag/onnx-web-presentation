// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        const session = await ort.InferenceSession.create('./model.onnx');

        // prepare inputs. a tensor need its corresponding TypedArray as data
        const yearsExperience = Float32Array.from([11.5]);
        const salary = Float32Array.from([0]);
        const tensorYears = new ort.Tensor('float32', yearsExperience, [1, 1]);
        const tensorSalary = new ort.Tensor('float32', salary, [1, 1]);

        // prepare feeds. use model input names as keys.
        const feeds = { yearsExperience: tensorYears, salary: tensorSalary };
        console.log(`feeds: ${JSON.stringify(feeds, null, " ")}`);
        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const dataC = results["Score.output"].data;
        console.log(`data of results["Score.output"]: ${dataC}`);

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();
