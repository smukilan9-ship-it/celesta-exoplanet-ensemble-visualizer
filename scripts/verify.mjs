import { readFile } from "node:fs/promises";

const data = JSON.parse(await readFile(new URL("../data/model-traces.json", import.meta.url), "utf8"));
const classes = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"];

if (JSON.stringify(data.meta.classOrder) !== JSON.stringify(classes)) throw new Error("Class order drifted.");
if (data.objects.length < 6) throw new Error("Expected curated correct, error, and disagreement traces.");

for (const trace of data.objects) {
  if (trace.input.retainedFeatureCount !== 98) throw new Error(`Feature contract drifted: ${trace.objectId}`);
  if (trace.models.lightgbm.leafIndices.length !== 1350) throw new Error(`LightGBM trace is incomplete: ${trace.objectId}`);
  if (trace.models.catboost.leafIndices.length !== 450) throw new Error(`CatBoost trace is incomplete: ${trace.objectId}`);
  const blend = classes.map((label) => trace.ensemble.rawProbabilities[label]);
  if (Math.abs(blend.reduce((sum, value) => sum + value, 0) - 1) > 2e-7) {
    throw new Error(`Invalid blend: ${trace.objectId}`);
  }
}

console.log(`Model traces verified: ${data.objects.length} real held-out objects and 1,800 trees each.`);
