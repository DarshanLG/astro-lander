// Runs an exported policy in the JS port and prints outcome statistics as JSON.
//   node web/parity_test.js web/policies/nominal.json 200 [windPower] [gravity]
globalThis.planck = require(process.env.PLANCK_PATH || "planck");
require("./lander.js");
const fs = require("fs");
const { LunarLander, MLPPolicy } = globalThis.LunarLanderJS;
const [, , policyPath, nArg, windArg, gravArg] = process.argv;
const spec = JSON.parse(fs.readFileSync(policyPath, "utf8"));
const pol = new MLPPolicy(spec);
const n = +(nArg || 100), wind = +(windArg || 0), gravity = +(gravArg || -10);
const env = new LunarLander({ continuous: spec.continuous, gravity, enableWind: wind > 0, windPower: wind, turbulencePower: wind > 0 ? 1.0 : 0 });
const counts = { landed: 0, crashed: 0, out_of_bounds: 0, timeout: 0 };
let onPad = 0, totalRet = 0, solved = 0;
for (let ep = 0; ep < n; ep++) {
  let obs = env.reset(1_000_000 + ep), ret = 0, r;
  do { r = env.step(pol.act(obs).action); obs = r.obs; ret += r.reward; } while (!r.terminated && !r.truncated);
  counts[r.outcome]++; totalRet += ret; if (ret >= 200) solved++;
  if (r.outcome === "landed" && Math.abs(obs[0]) <= 0.2) onPad++;
}
console.log(JSON.stringify({ episodes: n, mean_return: totalRet / n, solved_rate: solved / n, success_rate: onPad / n,
  landed_rate: counts.landed / n, crash_rate: (counts.crashed + counts.out_of_bounds) / n, timeout_rate: counts.timeout / n }));
